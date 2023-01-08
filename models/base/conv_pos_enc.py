import torch
import torch.nn as nn
from einops import rearrange


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Sequential(nn.ReflectionPad2d(k // 2), nn.Conv2d(dim, dim, k, 1, 0, groups=dim),)

    def forward(self, x, size=(64, 64)):
        x = self.proj(x) + x
        return x
