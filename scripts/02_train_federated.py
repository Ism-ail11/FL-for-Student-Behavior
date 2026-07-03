from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ensure_output_dirs, load_config
from src.data.dataset import StudentBehaviorDataset, detection_collate
from src.engine.federated import set_state, train_federated_round
from src.engine.train import EarlyStopping, evaluate
from src.losses.detection_loss import DetectionLoss
from src.models.tiny_detector import TinyStudentBehaviorDetector
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--partition", choices=["iid", "noniid"], default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    paths = ensure_output_dirs(cfg)
    dcfg = cfg["dataset"]
    mcfg = cfg["model"]
    flcfg = cfg["federated_learning"]
    tcfg = cfg["centralized_training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyStudentBehaviorDetector(dcfg["num_classes"], mcfg["anchors"], mcfg["dropout"]).to(device)
    val_set = StudentBehaviorDataset(paths["manifests"] / "val.csv", dcfg["image_size"], augment=False, num_classes=dcfg["num_classes"])
    val_loader = DataLoader(val_set, batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["num_workers"], collate_fn=detection_collate, pin_memory=torch.cuda.is_available())
    criterion = DetectionLoss(dcfg["num_classes"], mcfg["anchors"], cfg["loss"]["lambda_cls"], cfg["loss"]["lambda_obj"], cfg["loss"]["lambda_box"], cfg["loss"]["no_object_weight"])
    if args.partition == "iid":
        client_dir = paths["manifests"] / "clients_iid"
    else:
        client_dir = paths["manifests"] / f"clients_noniid_alpha_{args.alpha}"
    client_manifests = sorted(client_dir.glob("client_*.csv"))
    if len(client_manifests) != flcfg["clients"]:
        raise RuntimeError(f"Expected {flcfg['clients']} client manifests in {client_dir}. Run 00_prepare_splits.py first.")
    stopper = EarlyStopping(flcfg["server_patience"], mode="max")
    for round_id in range(1, flcfg["rounds"] + 1):
        avg_state = train_federated_round(model, client_manifests, cfg, device)
        set_state(model, avg_state)
        val_logs = evaluate(model, val_loader, criterion, device, mcfg["anchors"], dcfg["num_classes"], tcfg["confidence_threshold"], tcfg["iou_threshold"])
        score = val_logs["map50_proxy"]
        print(f"round={round_id} val_f1={val_logs['f1']:.4f} val_map50_proxy={score:.4f}")
        if stopper.step(score):
            save_checkpoint(paths["checkpoints"] / "federated_best.pt", model, {"round": round_id, "val": val_logs, "partition": args.partition, "alpha": args.alpha})
        if stopper.should_stop:
            print("Server-level early stopping.")
            break


if __name__ == "__main__":
    main()
