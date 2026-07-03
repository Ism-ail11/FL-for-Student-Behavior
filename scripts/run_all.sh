#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/paper_config.yaml}
python scripts/00_prepare_splits.py --config "$CONFIG" --alpha 0.5
python scripts/01_train_centralized.py --config "$CONFIG"
python scripts/02_train_federated.py --config "$CONFIG" --partition iid
python scripts/02_train_federated.py --config "$CONFIG" --partition noniid --alpha 0.5
python scripts/03_prune_finetune.py --config "$CONFIG"
python scripts/04_export_onnx.py --config "$CONFIG"
python scripts/05_convert_int8_tflite.py --config "$CONFIG"
python scripts/06_export_c_array.py --config "$CONFIG"
