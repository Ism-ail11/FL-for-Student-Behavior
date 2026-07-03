#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/paper_config.yaml}
for ALPHA in 1.0 0.5 0.1; do
  python scripts/00_prepare_splits.py --config "$CONFIG" --alpha "$ALPHA"
  python scripts/02_train_federated.py --config "$CONFIG" --partition noniid --alpha "$ALPHA"
done
