from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, metadata: Dict[str, Any] | None = None) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return payload.get("metadata", {}) if isinstance(payload, dict) else {}
