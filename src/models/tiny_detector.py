from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, act: str = "relu6") -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = _activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, act: str = "relu6") -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = _activation(act)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = _activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int = 2, act: str = "relu6") -> None:
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.use_residual = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            ConvBNAct(in_ch, hidden, kernel=1, stride=1, act=act),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            _activation(act),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.out_act = _activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_residual:
            y = y + x
        return self.out_act(y)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name in {"hswish", "hard_swish", "hard-swish"}:
        return nn.Hardswish(inplace=True)
    if name == "linear":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class TinyStudentBehaviorDetector(nn.Module):
    """Lightweight detector matching the paper output tensor: B x 75 x 40 x 40."""

    def __init__(self, num_classes: int = 20, anchors: int = 3, dropout: float = 0.10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.output_channels = anchors * (4 + 1 + num_classes)

        self.stem = ConvBNAct(3, 16, kernel=3, stride=2, act="relu6")      # 160 x 160 x 16
        self.block1 = DepthwiseSeparable(16, 16, stride=1, act="relu6")     # 160 x 160 x 16
        self.block2 = InvertedResidual(16, 24, stride=2, act="relu6")       # 80 x 80 x 24
        self.block3 = InvertedResidual(24, 40, stride=2, act="relu6")       # 40 x 40 x 40
        self.block4 = InvertedResidual(40, 80, stride=2, act="hard_swish")  # 20 x 20 x 80
        self.block5 = nn.Sequential(
            InvertedResidual(80, 96, stride=1, act="hard_swish"),
            nn.Dropout2d(dropout),
        )                                                                   # 20 x 20 x 96

        self.proj1 = ConvBNAct(40, 64, kernel=1, stride=1, act="relu6")
        self.proj2 = ConvBNAct(96, 64, kernel=1, stride=1, act="relu6")
        self.upsample = nn.Upsample(size=(40, 40), mode="nearest")
        self.head1 = nn.Sequential(
            DepthwiseSeparable(64, 64, stride=1, act="relu6"),
            nn.Dropout2d(dropout),
        )
        self.pred = nn.Conv2d(64, self.output_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        f40 = self.block3(x)
        x = self.block4(f40)
        f20 = self.block5(x)
        fused = self.proj1(f40) + self.upsample(self.proj2(f20))
        return self.pred(self.head1(fused))

    def prediction_shape(self, grid_size: int = 40) -> Tuple[int, int, int]:
        return grid_size, grid_size, self.output_channels
