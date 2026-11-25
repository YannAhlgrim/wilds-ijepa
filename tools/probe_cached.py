#!/usr/bin/env python3
"""Train linear/BN-linear heads on cached features and expand probe grids."""

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cached.probe import evaluate_head, train_cached_head
from src.datasets.wilds import _stratified_subset_indices


GRID_KEYS = ("lr", "weight_decay", "batch_size", "head_type", "representation_type")
ALIASES = {
    "wd": "weight_decay",
    "batch": "batch_size",
    "head": "head_type",
    "rep": "representation_type",
}


def load_grid(source):
    """Read a YAML path or inline YAML and return (defaults, configurations)."""
    text = open(source, "r").read() if os.path.isfile(source) else source
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Grid must be a YAML mapping")
    defaults = value.get("defaults", {})
    raw_grid = value.get("grid", value)
    normalized = {}
    for key, values in raw_grid.items():
        key = ALIASES.get(key, key)
        if key in ("defaults",):
            continue
        if not isinstance(values, list):
            values = [values]
        normalized[key] = values
    missing = [key for key in GRID_KEYS if key not in normalized]
    if missing:
        raise ValueError(f"Grid is missing keys: {missing}")
    configurations = []
    for combination in itertools.product(*(normalized[key] for key in GRID_KEYS)):
        config = dict(zip(GRID_KEYS, combination))
        config.update(
            {
                "max_epochs": int(defaults.get("max_epochs", 100)),
                "patience": int(defaults.get("patience", 10)),
                "min_delta": float(defaults.get("min_delta", 1e-4)),
                "min_epochs": int(defaults.get("min_epochs", 0)),
            }
        )
        configurations.append(config)
    return defaults, configurations


def config_slug(config):
    return (
        f"lr{float(config['lr']):g}-wd{float(config['weight_decay']):g}"
        f"-bs{int(config['batch_size'])}-{config['head_type']}-{config['representation_type']}"
    )


def _load_split(features_dir, split, representation):
    prefix = os.path.join(features_dir, split)
    features = np.load(f"{prefix}_{representation}.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(f"{prefix}_y.npy", mmap_mode="r", allow_pickle=False)
    metadata = np.load(f"{prefix}_metadata.npy", mmap_mode="r", allow_pickle=False)
    if len(features) != len(labels) or len(labels) != len(metadata):
        raise ValueError(f"Cached split {split} has inconsistent array lengths")
    return features, labels, metadata


def _atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run_one_configuration(
    features_dir,
    manifest,
    config,
    seed,
    out_dir,
    *,
    stage,
    label_fraction,
    device,
    overwrite=False,
):
    encoder_key = manifest["encoder_key"]
    run_dir = os.path.join(
        out_dir, encoder_key, stage, f"seed{int(seed)}", config_slug(config)
    )
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.isfile(summary_path) and not overwrite:
        print(f"Skipping completed run: {run_dir}")
        with open(summary_path, "r") as handle:
            return json.load(handle)
    os.makedirs(run_dir, exist_ok=True)
    started = time.perf_counter()
    representation = config["representation_type"]
    train_features, train_labels, train_metadata = _load_split(features_dir, "train", representation)
    val_features, val_labels, val_metadata = _load_split(features_dir, "val", representation)
    id_val_features, id_val_labels, id_val_metadata = _load_split(features_dir, "id_val", representation)
    indices = None
    if label_fraction < 1.0:
        indices = _stratified_subset_indices(train_labels, label_fraction, seed)
    head, history, early, early_stopped = train_cached_head(
        train_features,
        train_labels,
        val_features,
        val_labels,
        val_metadata,
        id_val_features,
        id_val_labels,
        id_val_metadata,
        num_classes=int(manifest["num_classes"]),
        config=config,
        seed=seed,
        device=device,
        train_indices=indices,
    )
    train_seconds = time.perf_counter() - started

    epoch_path = os.path.join(run_dir, "epochs.csv")
    with open(epoch_path, "w", newline="") as handle:
        fields = list(history[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    metrics = {}
    for split in ("train", "val", "id_val", "id_test", "test"):
        if split == "train":
            features, labels, metadata = train_features, train_labels, train_metadata
        else:
            features, labels, metadata = _load_split(features_dir, split, representation)
        split_metrics, predictions, _ = evaluate_head(
            head, features, labels, metadata, int(config["batch_size"]), device
        )
        metrics[split] = split_metrics
        if split in ("id_test", "test"):
            with open(os.path.join(run_dir, f"{split}_predictions.csv"), "w") as handle:
                for prediction in predictions:
                    handle.write(f"{int(prediction)}\n")
    summary = {
        "format_version": 1,
        "encoder_key": encoder_key,
        "feature_manifest": os.path.abspath(os.path.join(features_dir, "manifest.json")),
        "config": config,
        "config_slug": config_slug(config),
        "seed": int(seed),
        "stage": stage,
        "label_fraction": float(label_fraction),
        "num_train_images": len(train_features) if indices is None else len(indices),
        "best_epoch": early.best_epoch,
        "best_val_macro_f1": early.best_metric,
        "epochs_run": len(history),
        "early_stopped": early_stopped,
        "train_time_seconds": float(train_seconds),
        "wall_time_seconds": float(time.perf_counter() - started),
        "metrics": metrics,
    }
    _atomic_json(summary_path, summary)
    print(
        f"{encoder_key} {stage} seed={seed} {config_slug(config)}: "
        f"best epoch={early.best_epoch}, OOD-val F1={early.best_metric:.6f}"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--grid", required=True, help="YAML path or inline YAML grid")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--stage", default="full")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-delta", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.label_fraction <= 1:
        raise ValueError("label-fraction must be in (0, 1]")
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    with open(os.path.join(args.features, "manifest.json"), "r") as handle:
        manifest = json.load(handle)
    _, configurations = load_grid(args.grid)
    for config in configurations:
        if args.max_epochs is not None:
            config["max_epochs"] = args.max_epochs
        if args.patience is not None:
            config["patience"] = args.patience
        if args.min_delta is not None:
            config["min_delta"] = args.min_delta
        if config["representation_type"] == "cls" and not os.path.isfile(
            os.path.join(args.features, "train_cls.npy")
        ):
            raise ValueError("CLS grid requested but this encoder has no cached CLS features")
        for seed in args.seeds:
            run_one_configuration(
                args.features,
                manifest,
                config,
                seed,
                args.out,
                stage=args.stage,
                label_fraction=args.label_fraction,
                device=args.device,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
