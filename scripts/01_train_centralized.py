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
from src.engine.train import EarlyStopping, evaluate, train_one_epoch
from src.losses.detection_loss import DetectionLoss
from src.models.tiny_detector import TinyStudentBehaviorDetector
from src.utils.checkpoint import save_checkpoint
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    paths = ensure_output_dirs(cfg)
    dcfg = cfg["dataset"]
    tcfg = cfg["centralized_training"]
    mcfg = cfg["model"]
    lcfg = cfg["loss"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = StudentBehaviorDataset(paths["manifests"] / "train.csv", dcfg["image_size"], augment=True, num_classes=dcfg["num_classes"])
    val_set = StudentBehaviorDataset(paths["manifests"] / "val.csv", dcfg["image_size"], augment=False, num_classes=dcfg["num_classes"])
    train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, num_workers=tcfg["num_workers"], collate_fn=detection_collate, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["num_workers"], collate_fn=detection_collate, pin_memory=torch.cuda.is_available())
    model = TinyStudentBehaviorDetector(dcfg["num_classes"], mcfg["anchors"], mcfg["dropout"]).to(device)
    criterion = DetectionLoss(dcfg["num_classes"], mcfg["anchors"], lcfg["lambda_cls"], lcfg["lambda_obj"], lcfg["lambda_box"], lcfg["no_object_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    stopper = EarlyStopping(tcfg["early_stopping_patience"], mode="max")
    for epoch in range(1, tcfg["epochs"] + 1):
        train_logs = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logs = evaluate(model, val_loader, criterion, device, mcfg["anchors"], dcfg["num_classes"], tcfg["confidence_threshold"], tcfg["iou_threshold"])
        score = val_logs["map50_proxy"]
        print(f"epoch={epoch} train_loss={train_logs['loss']:.4f} val_f1={val_logs['f1']:.4f} val_map50_proxy={score:.4f}")
        if stopper.step(score):
            save_checkpoint(paths["checkpoints"] / "centralized_best.pt", model, {"epoch": epoch, "val": val_logs})
        if stopper.should_stop:
            print("Early stopping.")
            break


if __name__ == "__main__":
    main()
