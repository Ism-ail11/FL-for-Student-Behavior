from __future__ import annotations

from typing import Dict, Iterable

import torch


def state_dict_size_bytes(model: torch.nn.Module) -> int:
    total = 0
    for tensor in model.state_dict().values():
        total += tensor.numel() * tensor.element_size()
    return int(total)


def communication_per_round_mb(model: torch.nn.Module, clients: int, include_download: bool = True) -> float:
    model_mb = state_dict_size_bytes(model) / (1024.0 * 1024.0)
    upload = clients * model_mb
    download = clients * model_mb if include_download else 0.0
    return upload + download


def communication_table(model: torch.nn.Module, client_counts: Iterable[int]) -> Dict[int, float]:
    return {int(k): communication_per_round_mb(model, int(k), include_download=True) for k in client_counts}
