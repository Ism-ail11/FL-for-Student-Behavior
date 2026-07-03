from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.config import load_config
from src.deployment.benchmark import benchmark_tflite_host


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = benchmark_tflite_host(cfg["deployment"]["tflite_model"], cfg["dataset"]["image_size"], args.runs)
    print(result)


if __name__ == "__main__":
    main()
