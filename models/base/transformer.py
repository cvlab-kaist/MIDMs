import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import Dropout, Module

from models.base.conv_pos_enc import ConvPosEnc
from models.normalization import SPADE


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, expand_ratio=4.0, attention_type="self", spade=False, pos_type="conv"):
        super(TransformerEncoderLayer, self).__init__()

        self.pos_embed = ConvPosEnc(d_model, k=7)

        self.dim = d_model // nhead
        self.nhead = nhead
        self.attention_type = attention_type
        self.pos_type = pos_type

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        # self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * expand_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * expand_ratio), d_model),
        )

        # norm and dropout
        self.spade = spade
        if spade:
            spade_config_str = "spadesyncbatch3x3"
            ic = 3
            self.norm1 = SPADE(spade_config_str, d_model, ic, PONO=True, use_apex=False)
            self.norm2 = SPADE(spade_config_str, d_model, ic, PONO=True, use_apex=False)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward_attention(self, x, source, x_mask=None, source_mask=None):
        bs = x.size(0)
        query, key, value = x, source, source

        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = message.view(bs, -1, self.nhead * self.dim)  # [N, L, C]
        return message

    def forward(self, x, source=None, x_mask=None, source_mask=None, spade_segmap=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        x = self.pos_embed(x) if self.pos_type == "conv" else x
        x_shape = x.shape
        if len(x_shape) == 4:
            x = rearrange(x, "B C H W -> B (H W) C")
        if source is not None and len(source.shape) == 4:
            source = rearrange(source, "B C H W -> B (H W) C")

        # multi-head attention
        if self.spade:
            x2 = self.norm1(rearrange(x, "B (H W) C -> B C H W", W=x_shape[-1]), spade_segmap)
            x2 = rearrange(x2, "B C H W -> B (H W) C")
        else:
            x2 = self.norm1(x)

        if self.attention_type == "self":
            x = x + self.forward_attention(x2, x2, x_mask=x_mask, source_mask=source_mask)
        elif self.attention_type == "cross":
            x = x + self.forward_attention(x2, source, x_mask=x_mask, source_mask=source_mask)

        # feed-forward network
        if self.spade:
            x2 = self.norm2(rearrange(x, "B (H W) C -> B C H W", W=x_shape[-1]), spade_segmap)
            x2 = rearrange(x2, "B C H W -> B (H W) C")
        else:
            x2 = self.norm2(x)
        x = x + self.mlp(x2)

        if len(x_shape) == 4:
            x = rearrange(x, "B (H W) C -> B C H W", W=x_shape[-1])

        return x


class LoFTRPositionEncoding(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model // 2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)]
