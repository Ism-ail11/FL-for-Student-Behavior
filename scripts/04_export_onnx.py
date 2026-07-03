from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from src.config import ensure_output_dirs, load_config
from src.deployment.export import export_onnx
from src.models.tiny_detector import TinyStudentBehaviorDetector
from src.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/pruned_finetuned_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_output_dirs(cfg)
    model = TinyStudentBehaviorDetector(cfg["dataset"]["num_classes"], cfg["model"]["anchors"], cfg["model"]["dropout"])
    load_checkpoint(args.checkpoint, model, map_location="cpu")
    out_path = paths["deployment"] / "student_behavior_detector.onnx"
    export_onnx(model, out_path, cfg["dataset"]["image_size"])
    print(f"ONNX exported to: {out_path}")


if __name__ == "__main__":
    main()
