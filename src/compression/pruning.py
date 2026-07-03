from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


def structured_channel_prune_inplace(model: nn.Module, pruning_ratio: float = 0.40, skip_prediction_layer: bool = True) -> Dict[str, List[int]]:
    """Apply mask-based structured pruning without changing tensor shapes.

    This keeps checkpoints exportable and simple. Channels are zeroed according to L1 importance.
    For actual inference speed-up, rebuild the architecture after pruning or use a deployment backend
    that removes zeroed channels. This implementation is safe for reproducibility and fine-tuning.
    """
    masks: Dict[str, List[int]] = {}
    conv_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for name, conv in conv_layers:
        if skip_prediction_layer and name.endswith("pred"):
            continue
        if conv.out_channels <= 4:
            continue
        weight = conv.weight.data
        scores = weight.abs().sum(dim=(1, 2, 3))
        prune_count = int(round(conv.out_channels * pruning_ratio))
        prune_count = min(max(prune_count, 0), conv.out_channels - 1)
        if prune_count == 0:
            continue
        prune_idx = torch.argsort(scores)[:prune_count]
        conv.weight.data[prune_idx] = 0.0
        if conv.bias is not None:
            conv.bias.data[prune_idx] = 0.0
        masks[name] = prune_idx.cpu().tolist()
    return masks


def count_nonzero_parameters(model: nn.Module) -> int:
    return int(sum(torch.count_nonzero(p.detach()).item() for p in model.parameters()))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
