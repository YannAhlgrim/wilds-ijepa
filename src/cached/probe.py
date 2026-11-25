"""Training on process-resident cached tensors with index-based mini-batches."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from src.cached.metrics import evaluate_iwildcam


def _features_on_device(features, device):
    return torch.as_tensor(features).to(device=device, dtype=torch.float16)


def _labels_on_device(labels, device):
    return torch.as_tensor(labels).to(device=device, dtype=torch.long).reshape(-1)


def _labels_numpy(labels):
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


def make_cached_head(input_dim, num_classes, head_type):
    if head_type == "linear":
        head = nn.Linear(input_dim, num_classes)
    elif head_type == "bn_linear":
        head = nn.Sequential(
            nn.BatchNorm1d(input_dim, affine=False, eps=1e-6),
            nn.Linear(input_dim, num_classes),
        )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)
    return head


class MaxEarlyStopping:
    """Maximize OOD-val macro-F1 and retain the best head on CPU."""

    def __init__(self, patience=10, min_delta=1e-4, min_epochs=0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.best_metric = -float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.best_state = None

    def step(self, epoch, metric, head):
        if metric > self.best_metric + self.min_delta:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(head.state_dict())
            self.best_state = {key: value.detach().cpu().clone() for key, value in self.best_state.items()}
            return True, False
        self.bad_epochs += 1
        should_stop = epoch >= self.min_epochs and self.bad_epochs >= self.patience
        return False, should_stop

    def restore(self, head):
        if self.best_state is None:
            raise RuntimeError("No best epoch was recorded")
        head.load_state_dict(self.best_state)


def evaluate_head(head, features, labels, metadata, batch_size, device):
    """Evaluate in sequential slices without per-item Python/DataLoader IO."""
    device = torch.device(device)
    features = _features_on_device(features, device)
    labels = _labels_on_device(labels, device)
    criterion = nn.CrossEntropyLoss()
    predictions = []
    loss_sum = torch.zeros((), device=device)
    count = 0
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features), int(batch_size)):
            end = min(start + int(batch_size), len(features))  # drop_last=False
            features_batch = features[start:end].float()
            labels_batch = labels[start:end]
            logits = head(features_batch)
            loss = criterion(logits, labels_batch)
            loss_sum += loss.detach() * (end - start)
            count += end - start
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(predictions)
    metrics, summary = evaluate_iwildcam(predictions, _labels_numpy(labels), metadata)
    metrics["loss"] = float(loss_sum.item()) / max(count, 1)
    return metrics, predictions, summary


def train_cached_head(
    train_features,
    train_labels,
    val_features,
    val_labels,
    val_metadata,
    id_val_features,
    id_val_labels,
    id_val_metadata,
    *,
    num_classes,
    config,
    seed,
    device,
    train_indices=None,
):
    """Train/restore using seeded on-device indices and only OOD-val F1."""
    device = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    batch_size = int(config["batch_size"])
    train_features = _features_on_device(train_features, device)
    train_labels = _labels_on_device(train_labels, device)
    val_features = _features_on_device(val_features, device)
    val_labels = _labels_on_device(val_labels, device)
    id_val_features = _features_on_device(id_val_features, device)
    id_val_labels = _labels_on_device(id_val_labels, device)
    if train_indices is None:
        active_indices = torch.arange(len(train_features), dtype=torch.long, device=device)
    else:
        active_indices = torch.as_tensor(train_indices, dtype=torch.long, device=device).reshape(-1)
    if len(active_indices) == 0:
        raise ValueError("Training subset is empty")
    # A dedicated CPU generator makes run-seed order independent of global RNG.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    head = make_cached_head(int(train_features.shape[1]), num_classes, config["head_type"]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["max_epochs"]), eta_min=0.0
    )
    criterion = nn.CrossEntropyLoss()
    early = MaxEarlyStopping(
        patience=int(config.get("patience", 10)),
        min_delta=float(config.get("min_delta", 1e-4)),
        min_epochs=int(config.get("min_epochs", 0)),
    )
    history = []
    early_stopped = False
    for epoch in range(1, int(config["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        head.train()
        loss_sum = torch.zeros((), device=device)
        count = 0
        order = torch.randperm(len(active_indices), generator=generator).to(device)
        order = active_indices.index_select(0, order)
        optimizer.zero_grad()
        for start in range(0, len(order), batch_size):
            # Preserve existing drop_last=False behavior, including for BN.
            batch_indices = order[start : min(start + batch_size, len(order))]
            features_batch = train_features.index_select(0, batch_indices).float()
            labels_batch = train_labels.index_select(0, batch_indices)
            logits = head(features_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.detach() * int(labels_batch.shape[0])
            count += int(labels_batch.shape[0])
        val_metrics, _, _ = evaluate_head(
            head, val_features, val_labels, val_metadata, batch_size, device
        )
        id_val_metrics, _, _ = evaluate_head(
            head, id_val_features, id_val_labels, id_val_metadata, batch_size, device
        )
        improved, should_stop = early.step(epoch, val_metrics["F1-macro_all"], head)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss_sum.item()) / max(count, 1),
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["F1-macro_all"],
                "id_val_f1": id_val_metrics["F1-macro_all"],
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_f1": early.best_metric,
                "best_epoch": early.best_epoch,
                "improved": int(improved),
                "early_stop": int(should_stop),
                "epoch_time_seconds": float(time.perf_counter() - epoch_started),
            }
        )
        scheduler.step()
        if should_stop:
            early_stopped = True
            break
    early.restore(head)
    return head, history, early, early_stopped
