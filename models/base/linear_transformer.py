from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from torch.nn.modules.normalization import GroupNorm

from models.base.conv4d import Conv4d


class LinearConv4d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        affinity_dim,
        target_proj_dim,
        input_corr_size=(8, 8, 8, 8),
        kernel_size=(3, 3, 3, 3),
        stride=(2, 2, 2, 2),
        padding=(1, 1, 1, 1),
        group=1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            Conv4d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(group, out_channel),
        )

        output_dim = out_channel * reduce(
            lambda x, y: x * y, [c // s for c, s in zip(input_corr_size[:2], stride[:2])]
        )
        self.linear = nn.Linear(output_dim + affinity_dim, target_proj_dim)

    def forward(self, x, affinity):
        assert len(x.shape) == 6, "input should be in shape B C H_t W_t H_s W_s"
        assert len(affinity.shape) == 4, "affinity should be in shape B C H_s W_s"
        x = self.conv(x)
        x = torch.cat(
            (
                rearrange(x, "B C H_t W_t H_s W_s -> B (H_s W_s) (C H_t W_t)"),
                rearrange(affinity, "B C H W -> B (H W) C"),
            ),
            dim=-1,
        )
        x = self.linear(x)
        return x


class MLPConv4d(nn.Module):
    def __init__(self, in_channel, mlp_ratio=4.0, kernel_size=(3, 3, 3, 3), stride=(2, 2, 2, 2), padding=(1, 1, 1, 1)):
        super().__init__()

        self.mlp = nn.Sequential(
            Conv4d(
                in_channels=in_channel,
                out_channels=int(in_channel * mlp_ratio),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GELU(),
            Conv4d(
                in_channels=int(in_channel * mlp_ratio),
                out_channels=in_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        return self.mlp(x)


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
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

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float("-inf"))

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class TransformerLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        affinity_dim=64,
        target_proj_dim=384,
        nhead=4,
        mlp_ratio=4.0,
        input_corr_size=(8, 8, 8, 8),
        kernel_size=(3, 3, 3, 3),
        stride=(2, 2, 1, 1),
        padding=(1, 1, 1, 1),
        group=1,
    ):
        super().__init__()
        self.nhead = nhead
        self.qk = LinearConv4d(
            in_channel,
            in_channel,
            affinity_dim,
            target_proj_dim * 2,
            input_corr_size,
            kernel_size,
            stride,
            padding,
            group,
        )
        self.v = nn.Sequential(
            Conv4d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=kernel_size,
                stride=(1, 1, 1, 1),
                padding=padding,
            ),
            nn.GroupNorm(group, in_channel),
        )
        self.attn = FullAttention()
        self.mlp = MLPConv4d(in_channel, mlp_ratio, kernel_size, stride=(1, 1, 1, 1), padding=padding)

        self.norm1 = GroupNorm(group, in_channel)
        self.norm2 = GroupNorm(group, in_channel)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, input_corr_size[-2] * input_corr_size[-1], 1, target_proj_dim // nhead)
        )
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, affinity):
        """
        x: B C H_t W_t H_s W_s
        affinity: B affinity_dim H_s W_s
        """
        qkv = self.norm1(x)

        q, k = self.qk(qkv, affinity).chunk(2, dim=-1)  # B (H_s W_s) C
        v = self.v(qkv)  # B C H_t W_t H_s W_s

        q = rearrange(q, "B S (C H) -> B S H C", H=self.nhead) + self.pos_embed
        k = rearrange(k, "B S (C H) -> B S H C", H=self.nhead) + self.pos_embed
        H_s, W_s, H_t, W_t = v.shape[-4:]
        v = rearrange(v, "B (H C) H_t W_t H_s W_s -> B (H_s W_s) H (C H_t W_t)", H=self.nhead)

        attn = rearrange(
            self.attn(q, k, v),
            "B (H_s W_s) H (C H_t W_t) -> B (H C) H_t W_t H_s W_s",
            H_s=H_s,
            W_s=W_s,
            H_t=H_t,
            W_t=W_t,
        )

        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class LayerNormWarper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, "B L ... -> B ... L")
        x = self.norm(x)
        x = rearrange(x, "B ... L -> B L ...")
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        in_channel,
        depth=2,
        affinity_dim=64,
        target_proj_dim=384,
        nhead=4,
        mlp_ratio=4.0,
        input_corr_size=(8, 8, 8, 8),
        kernel_size=(3, 3, 3, 3),
        stride=(2, 2, 1, 1),
        padding=(1, 1, 1, 1),
        group=1,
    ):
        super().__init__()
        assert stride[-1] == 1 and stride[-2] == 1, "stride of source dimension must be 1"

        self.layer = nn.ModuleList(
            [
                TransformerLayer(
                    in_channel,
                    affinity_dim,
                    target_proj_dim,
                    nhead,
                    mlp_ratio,
                    input_corr_size,
                    kernel_size,
                    stride,
                    padding,
                    group,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, affinity):
        for layer in self.layer:
            x = layer(x, affinity)

        return x
