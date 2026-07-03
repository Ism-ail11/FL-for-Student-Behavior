#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/paper_config.yaml}
python scripts/01_train_centralized.py --config "$CONFIG"
python scripts/02_train_federated.py --config "$CONFIG" --partition iid
python scripts/03_prune_finetune.py --config "$CONFIG" --checkpoint outputs/checkpoints/federated_best.pt
python scripts/04_export_onnx.py --config "$CONFIG" --checkpoint outputs/checkpoints/pruned_finetuned_best.pt
python scripts/09_communication_cost.py --config "$CONFIG" --clients 5 10 20 30
