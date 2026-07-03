from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.config import load_config
from src.experiments.communication import communication_table, state_dict_size_bytes
from src.models.tiny_detector import TinyStudentBehaviorDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--clients", nargs="+", type=int, default=[5, 10, 20, 30])
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = TinyStudentBehaviorDetector(cfg["dataset"]["num_classes"], cfg["model"]["anchors"], cfg["model"]["dropout"])
    size_mb = state_dict_size_bytes(model) / (1024.0 * 1024.0)
    print(f"model_update_size_mb={size_mb:.4f}")
    for clients, mb in communication_table(model, args.clients).items():
        print(f"clients={clients}, communication_per_round_mb_upload_plus_download={mb:.4f}")


if __name__ == "__main__":
    main()
