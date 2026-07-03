from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a YAML dictionary.")
    return cfg


def ensure_output_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    base = Path(cfg["project"]["output_dir"])
    paths = {
        "base": base,
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
        "deployment": base / "deployment",
        "manifests": Path(cfg["dataset"]["manifest_dir"]),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
