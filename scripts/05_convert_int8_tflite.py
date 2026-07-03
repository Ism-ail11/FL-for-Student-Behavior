from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.config import ensure_output_dirs, load_config
from src.deployment.export import convert_onnx_to_tflite_int8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--onnx", default="outputs/deployment/student_behavior_detector.onnx")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_output_dirs(cfg)
    tflite = convert_onnx_to_tflite_int8(args.onnx, paths["deployment"] / "tflite_int8")
    final_path = paths["deployment"] / "student_behavior_detector_int8.tflite"
    final_path.write_bytes(Path(tflite).read_bytes())
    print(f"INT8 TFLite model exported to: {final_path}")


if __name__ == "__main__":
    main()
