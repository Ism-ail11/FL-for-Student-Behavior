from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch
from torch.utils.data import DataLoader

from src.compression.pruning import count_nonzero_parameters, count_parameters, structured_channel_prune_inplace
from src.config import ensure_output_dirs, load_config
from src.data.dataset import StudentBehaviorDataset, detection_collate
from src.engine.train import EarlyStopping, evaluate, train_one_epoch
from src.losses.detection_loss import DetectionLoss
from src.models.tiny_detector import TinyStudentBehaviorDetector
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/federated_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    paths = ensure_output_dirs(cfg)
    dcfg = cfg["dataset"]
    mcfg = cfg["model"]
    ccfg = cfg["compression"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyStudentBehaviorDetector(dcfg["num_classes"], mcfg["anchors"], mcfg["dropout"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    masks = structured_channel_prune_inplace(model, pruning_ratio=ccfg["structured_pruning_ratio"])
    print(f"Pruned layers: {len(masks)}")
    print(f"Nonzero/total params: {count_nonzero_parameters(model)}/{count_parameters(model)}")
    train_set = StudentBehaviorDataset(paths["manifests"] / "train.csv", dcfg["image_size"], augment=True, num_classes=dcfg["num_classes"])
    val_set = StudentBehaviorDataset(paths["manifests"] / "val.csv", dcfg["image_size"], augment=False, num_classes=dcfg["num_classes"])
    train_loader = DataLoader(train_set, batch_size=ccfg["fine_tune_batch_size"], shuffle=True, num_workers=cfg["centralized_training"]["num_workers"], collate_fn=detection_collate, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=ccfg["fine_tune_batch_size"], shuffle=False, num_workers=cfg["centralized_training"]["num_workers"], collate_fn=detection_collate, pin_memory=torch.cuda.is_available())
    criterion = DetectionLoss(dcfg["num_classes"], mcfg["anchors"], cfg["loss"]["lambda_cls"], cfg["loss"]["lambda_obj"], cfg["loss"]["lambda_box"], cfg["loss"]["no_object_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=ccfg["fine_tune_learning_rate"], weight_decay=cfg["centralized_training"]["weight_decay"])
    stopper = EarlyStopping(ccfg["fine_tune_patience"], mode="max")
    for epoch in range(1, ccfg["fine_tune_epochs"] + 1):
        train_logs = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_logs = evaluate(model, val_loader, criterion, device, mcfg["anchors"], dcfg["num_classes"], cfg["centralized_training"]["confidence_threshold"], cfg["centralized_training"]["iou_threshold"])
        print(f"finetune_epoch={epoch} loss={train_logs['loss']:.4f} f1={val_logs['f1']:.4f} map50_proxy={val_logs['map50_proxy']:.4f}")
        if stopper.step(val_logs["map50_proxy"]):
            save_checkpoint(paths["checkpoints"] / "pruned_finetuned_best.pt", model, {"epoch": epoch, "val": val_logs, "pruning_masks": masks})
        if stopper.should_stop:
            print("Fine-tuning early stopping.")
            break


if __name__ == "__main__":
    main()
