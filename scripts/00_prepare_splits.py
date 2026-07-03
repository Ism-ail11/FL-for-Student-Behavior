from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.config import ensure_output_dirs, load_config
from src.data.partition import dirichlet_noniid_partition, iid_partition, read_manifest, write_client_manifests
from src.data.splits import make_train_val_test_splits, scan_yolo_dataset, write_manifest
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    paths = ensure_output_dirs(cfg)
    dcfg = cfg["dataset"]
    samples = scan_yolo_dataset(dcfg["root"], dcfg["images_dir"], dcfg["labels_dir"])
    splits = make_train_val_test_splits(samples, dcfg["train_ratio"], dcfg["val_ratio"], cfg["project"]["seed"])
    for split_name, split_samples in splits.items():
        write_manifest(paths["manifests"] / f"{split_name}.csv", split_samples)
        print(f"{split_name}: {len(split_samples)} samples")
    train_samples = splits["train"]
    flcfg = cfg["federated_learning"]
    iid = iid_partition(train_samples, flcfg["clients"], cfg["project"]["seed"])
    write_client_manifests(paths["manifests"] / "clients_iid", iid)
    noniid = dirichlet_noniid_partition(train_samples, flcfg["clients"], args.alpha, dcfg["num_classes"], cfg["project"]["seed"])
    write_client_manifests(paths["manifests"] / f"clients_noniid_alpha_{args.alpha}", noniid)
    print("Client manifests created.")


if __name__ == "__main__":
    main()
