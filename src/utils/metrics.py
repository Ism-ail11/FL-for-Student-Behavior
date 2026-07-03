from __future__ import annotations

from typing import Dict, List

import torch

from src.utils.boxes import box_iou, xywh_to_xyxy


def detection_metrics(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    iou_threshold: float = 0.50,
    num_classes: int = 20,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for pred, tgt in zip(predictions, targets):
        tgt = tgt.to(pred.device) if pred.numel() else tgt
        if tgt.numel() == 0:
            fp += int(pred.shape[0])
            continue
        target_boxes = xywh_to_xyxy(tgt[:, 1:5].to(pred.device))
        target_cls = tgt[:, 0].long().to(pred.device)
        matched = torch.zeros((target_boxes.shape[0],), dtype=torch.bool, device=pred.device)
        if pred.numel() == 0:
            fn += int(target_boxes.shape[0])
            continue
        for row in pred:
            box = row[:4].unsqueeze(0)
            cls_id = int(row[5].item())
            cls_mask = target_cls == cls_id
            if not cls_mask.any():
                fp += 1
                continue
            ious = box_iou(box, target_boxes).squeeze(0)
            ious = torch.where(cls_mask, ious, torch.zeros_like(ious))
            best_iou, best_idx = ious.max(dim=0)
            if best_iou >= iou_threshold and not matched[best_idx]:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1
        fn += int((~matched).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50_proxy": precision * recall,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
