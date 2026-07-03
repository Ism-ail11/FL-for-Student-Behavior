from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.dataset import StudentBehaviorDataset, detection_collate
from src.engine.train import evaluate, train_one_epoch
from src.losses.detection_loss import DetectionLoss


def get_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def weighted_average(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No client states received for aggregation.")
    total_weight = float(sum(weights))
    normalized = [w / total_weight for w in weights]
    avg: Dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        tensor = torch.zeros_like(states[0][key], dtype=states[0][key].dtype)
        if not torch.is_floating_point(tensor):
            avg[key] = states[0][key]
            continue
        for state, weight in zip(states, normalized):
            tensor += state[key].to(tensor.dtype) * weight
        avg[key] = tensor
    return avg


def train_federated_round(
    global_model: torch.nn.Module,
    client_manifests: List[Path],
    cfg: dict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    fl_cfg = cfg["federated_learning"]
    data_cfg = cfg["dataset"]
    loss_cfg = cfg["loss"]
    client_states: List[Dict[str, torch.Tensor]] = []
    client_sizes: List[int] = []
    global_state = get_state(global_model)

    for client_id, manifest in enumerate(client_manifests):
        local_model = copy.deepcopy(global_model).to(device)
        set_state(local_model, global_state)
        dataset = StudentBehaviorDataset(
            manifest,
            image_size=data_cfg["image_size"],
            augment=True,
            num_classes=data_cfg["num_classes"],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=fl_cfg["local_batch_size"],
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 0),
            collate_fn=detection_collate,
            pin_memory=torch.cuda.is_available(),
        )
        criterion = DetectionLoss(
            num_classes=data_cfg["num_classes"],
            anchors=cfg["model"]["anchors"],
            lambda_cls=loss_cfg["lambda_cls"],
            lambda_obj=loss_cfg["lambda_obj"],
            lambda_box=loss_cfg["lambda_box"],
            no_object_weight=loss_cfg["no_object_weight"],
        )
        optimizer = torch.optim.AdamW(
            local_model.parameters(),
            lr=fl_cfg["local_learning_rate"],
            weight_decay=fl_cfg["weight_decay"],
        )
        reference = {name: p.detach().cpu().clone() for name, p in global_model.named_parameters()}
        for _ in range(fl_cfg["local_epochs"]):
            train_one_epoch(
                local_model,
                dataloader,
                criterion,
                optimizer,
                device,
                proximal_reference=reference,
                proximal_mu=fl_cfg["proximal_mu"],
            )
        client_states.append(get_state(local_model))
        client_sizes.append(len(dataset))
        del local_model
    return weighted_average(client_states, client_sizes)
