from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.config import ensure_output_dirs, load_config
from src.deployment.export import export_tflite_c_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = ensure_output_dirs(cfg)
    tflite_path = Path(cfg["deployment"]["tflite_model"])
    export_tflite_c_array(
        tflite_path,
        paths["deployment"] / "model_data.cc",
        paths["deployment"] / "model_data.h",
        cfg["deployment"]["c_array_name"],
    )
    print("C array exported.")


if __name__ == "__main__":
    main()
