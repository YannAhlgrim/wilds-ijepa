"""Training primitives for heads on memory-mapped cached features."""

import copy

import numpy as np
import torch
import torch.nn as nn

from src.cached.metrics import evaluate_iwildcam


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, indices=None):
        self.features = features
        self.labels = labels
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.features) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            index = int(self.indices[index])
        # Batch indexing touches only the required mmap pages and returns a
        # writable float32 tensor, avoiding numpy's read-only mmap warning.
        feature = torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True))
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return feature, label


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
    loader = torch.utils.data.DataLoader(
        FeatureDataset(features, labels), batch_size=batch_size, shuffle=False
    )
    criterion = nn.CrossEntropyLoss()
    predictions = []
    loss_sum = 0.0
    count = 0
    head.eval()
    with torch.no_grad():
        for features_batch, labels_batch in loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits = head(features_batch)
            loss = criterion(logits, labels_batch)
            loss_sum += float(loss.item()) * int(labels_batch.shape[0])
            count += int(labels_batch.shape[0])
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(predictions)
    metrics, summary = evaluate_iwildcam(predictions, np.asarray(labels), metadata)
    metrics["loss"] = loss_sum / max(count, 1)
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
    """Train and restore a head using only OOD-val F1 for early stopping."""
    device = torch.device(device)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    batch_size = int(config["batch_size"])
    train_dataset = FeatureDataset(train_features, train_labels, indices=train_indices)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
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
        head.train()
        loss_sum = 0.0
        count = 0
        optimizer.zero_grad()
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            logits = head(features_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += float(loss.item()) * int(labels_batch.shape[0])
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
                "train_loss": loss_sum / max(count, 1),
                "val_loss": val_metrics["loss"],
                "val_f1": val_metrics["F1-macro_all"],
                "id_val_f1": id_val_metrics["F1-macro_all"],
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_f1": early.best_metric,
                "best_epoch": early.best_epoch,
                "improved": int(improved),
                "early_stop": int(should_stop),
            }
        )
        scheduler.step()
        if should_stop:
            early_stopped = True
            break
    early.restore(head)
    return head, history, early, early_stopped
