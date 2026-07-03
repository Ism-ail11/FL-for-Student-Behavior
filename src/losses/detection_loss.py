from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.decode import reshape_predictions
from src.utils.boxes import box_iou, xywh_to_xyxy


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 20,
        anchors: int = 3,
        lambda_cls: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_box: float = 5.0,
        no_object_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.no_object_weight = no_object_weight

    def forward(self, pred: torch.Tensor, targets: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        p = reshape_predictions(pred, self.anchors, self.num_classes)
        bsz, anchors, grid_h, grid_w, _ = p.shape
        device = pred.device

        obj_target = torch.zeros((bsz, anchors, grid_h, grid_w), device=device)
        cls_target = torch.zeros((bsz, anchors, grid_h, grid_w, self.num_classes), device=device)
        box_target = torch.zeros((bsz, anchors, grid_h, grid_w, 4), device=device)
        pos_mask = torch.zeros((bsz, anchors, grid_h, grid_w), dtype=torch.bool, device=device)

        for bi, target in enumerate(targets):
            if target.numel() == 0:
                continue
            target = target.to(device)
            for row in target:
                cls_id = int(row[0].item())
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue
                x, y, w, h = row[1:].clamp(0.0, 1.0)
                gx = min(int(x.item() * grid_w), grid_w - 1)
                gy = min(int(y.item() * grid_h), grid_h - 1)
                anchor_id = 0
                obj_target[bi, anchor_id, gy, gx] = 1.0
                cls_target[bi, anchor_id, gy, gx, cls_id] = 1.0
                box_target[bi, anchor_id, gy, gx] = torch.tensor([x, y, w, h], device=device)
                pos_mask[bi, anchor_id, gy, gx] = True

        xy = torch.sigmoid(p[..., 0:2])
        wh = torch.sigmoid(p[..., 2:4])
        obj_logits = p[..., 4]
        cls_logits = p[..., 5:]

        grid_y, grid_x = torch.meshgrid(torch.arange(grid_h, device=device), torch.arange(grid_w, device=device), indexing="ij")
        grid_x = grid_x.view(1, 1, grid_h, grid_w).float()
        grid_y = grid_y.view(1, 1, grid_h, grid_w).float()
        pred_box = torch.stack([
            (grid_x + xy[..., 0]) / float(grid_w),
            (grid_y + xy[..., 1]) / float(grid_h),
            wh[..., 0],
            wh[..., 1],
        ], dim=-1)

        obj_weight = torch.ones_like(obj_target)
        obj_weight[obj_target == 0] = self.no_object_weight
        loss_obj = F.binary_cross_entropy_with_logits(obj_logits, obj_target, weight=obj_weight, reduction="mean")

        if pos_mask.any():
            loss_cls = F.binary_cross_entropy_with_logits(cls_logits[pos_mask], cls_target[pos_mask], reduction="mean")
            pred_xyxy = xywh_to_xyxy(pred_box[pos_mask])
            target_xyxy = xywh_to_xyxy(box_target[pos_mask])
            ious = box_iou(pred_xyxy, target_xyxy).diag().clamp(0.0, 1.0)
            loss_iou = (1.0 - ious).mean()
            loss_l1 = F.smooth_l1_loss(pred_box[pos_mask], box_target[pos_mask], reduction="mean")
            loss_box = loss_iou + 0.5 * loss_l1
        else:
            loss_cls = torch.tensor(0.0, device=device)
            loss_box = torch.tensor(0.0, device=device)

        total = self.lambda_cls * loss_cls + self.lambda_obj * loss_obj + self.lambda_box * loss_box
        logs = {
            "loss": float(total.detach().cpu()),
            "loss_cls": float(loss_cls.detach().cpu()),
            "loss_obj": float(loss_obj.detach().cpu()),
            "loss_box": float(loss_box.detach().cpu()),
            "positive_cells": int(pos_mask.sum().detach().cpu()),
        }
        return total, logs
