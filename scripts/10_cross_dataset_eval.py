from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.splits import scan_yolo_dataset, write_manifest
from src.data.dataset import StudentBehaviorDataset, detection_collate
from src.engine.train import evaluate
from src.losses.detection_loss import DetectionLoss
from src.models.tiny_detector import TinyStudentBehaviorDetector
from src.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/pruned_finetuned_best.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyStudentBehaviorDetector(cfg["dataset"]["num_classes"], cfg["model"]["anchors"], cfg["model"]["dropout"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    criterion = DetectionLoss(cfg["dataset"]["num_classes"], cfg["model"]["anchors"], cfg["loss"]["lambda_cls"], cfg["loss"]["lambda_obj"], cfg["loss"]["lambda_box"], cfg["loss"]["no_object_weight"])
    for item in cfg.get("cross_dataset_validation", {}).get("datasets", []):
        root = Path(item["root"])
        samples = scan_yolo_dataset(root, item.get("images_dir", "images"), item.get("labels_dir", "labels"))
        manifest = Path(cfg["dataset"]["manifest_dir"]) / f"cross_{item['name']}.csv"
        write_manifest(manifest, samples)
        dataset = StudentBehaviorDataset(manifest, cfg["dataset"]["image_size"], augment=False, num_classes=cfg["dataset"]["num_classes"])
        loader = DataLoader(dataset, batch_size=cfg["centralized_training"]["batch_size"], shuffle=False, num_workers=cfg["centralized_training"]["num_workers"], collate_fn=detection_collate)
        metrics = evaluate(model, loader, criterion, device, cfg["model"]["anchors"], cfg["dataset"]["num_classes"], cfg["centralized_training"]["confidence_threshold"], cfg["centralized_training"]["iou_threshold"])
        print(item["name"], metrics)


if __name__ == "__main__":
    main()
