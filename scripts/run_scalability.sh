#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/paper_config.yaml}
for CLIENTS in 5 10 20 30; do
  python scripts/edit_config_value.py --config "$CONFIG" --key federated_learning.clients --value "$CLIENTS" --output "configs/scalability_${CLIENTS}.yaml"
  python scripts/00_prepare_splits.py --config "configs/scalability_${CLIENTS}.yaml" --alpha 0.5
  python scripts/02_train_federated.py --config "configs/scalability_${CLIENTS}.yaml" --partition noniid --alpha 0.5
done
