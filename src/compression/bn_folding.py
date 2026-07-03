from __future__ import annotations

import torch
from torch import nn


def fold_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    if conv.bias is None:
        bias = torch.zeros(conv.weight.shape[0], device=conv.weight.device)
    else:
        bias = conv.bias.data
    weight = conv.weight.data
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    std = torch.sqrt(var + eps)
    scale = gamma / std
    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = beta + (bias - mean) * scale
    new_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    new_conv.weight.data.copy_(folded_weight)
    new_conv.bias.data.copy_(folded_bias)
    return new_conv
