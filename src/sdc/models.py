from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block_2d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet2D(nn.Module):
    """Minimal 2D U-Net for reproducibility/smoke tests."""
    def __init__(self, in_channels: int, num_classes: int, base: int = 32) -> None:
        super().__init__()
        self.enc1 = conv_block_2d(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block_2d(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block_2d(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block_2d(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block_2d(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block_2d(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block_2d(base*2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

def conv_block_3d(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet3D(nn.Module):
    """Minimal 3D U-Net. Use small base channels for memory."""
    def __init__(self, in_channels: int, num_classes: int, base: int = 16) -> None:
        super().__init__()
        self.enc1 = conv_block_3d(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block_3d(base, base*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block_3d(base*2, base*4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = conv_block_3d(base*4, base*8)

        self.up3 = nn.ConvTranspose3d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block_3d(base*8, base*4)
        self.up2 = nn.ConvTranspose3d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block_3d(base*4, base*2)
        self.up1 = nn.ConvTranspose3d(base*2, base, 2, stride=2)
        self.dec1 = conv_block_3d(base*2, base)

        self.head = nn.Conv3d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)
