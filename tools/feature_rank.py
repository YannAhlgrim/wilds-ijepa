#!/usr/bin/env python3
"""Effective-rank and PCA-variance diagnostics for frozen pooled features.

Compute directly from a saved ``.npy`` matrix (rows are images), or reconstruct
an I-JEPA/external encoder and extract features from a WILDS split. Extraction
runs on CPU by default and never downloads a checkpoint implicitly; WILDS data
must already exist unless ``--download`` is explicitly supplied.

Examples:
  python tools/feature_rank.py --features features.npy --output rank.json
  python tools/feature_rank.py --checkpoint best_model.pth.tar \
    --config configs/supervised_dinov2_vitl14.yaml --root-path ./wilds_data \
    --split val --num-images 5000 --output rank.json
"""

import argparse
import copy
import json
import os
import sys

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def compute_feature_statistics(features, top_k=(1, 5, 10, 50, 100)):
    """Return effective rank and centered top-k principal-component variance.

    Effective rank is exp(H(s_i / sum(s))) over nonzero singular values s_i,
    following the requested definition. A numerical tolerance removes roundoff
    null directions so exactly rank-one synthetic matrices report rank ~1.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a [N, D] feature matrix, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Feature matrix must contain at least one row and one column")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or infinity")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    if singular_values.size and singular_values[0] > 0:
        tolerance = (
            np.finfo(np.float64).eps
            * max(centered.shape)
            * float(singular_values[0])
        )
        nonzero = singular_values[singular_values > tolerance]
    else:
        nonzero = singular_values[:0]

    if nonzero.size:
        probabilities = nonzero / nonzero.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = 0.0

    variances = singular_values ** 2
    total_variance = float(variances.sum())
    variance_fractions = []
    cumulative = np.cumsum(variances)
    for requested_k in top_k:
        k = int(requested_k)
        if k < 1:
            raise ValueError("top-k values must be positive integers")
        if not variances.size or total_variance == 0.0:
            fraction = 0.0
        else:
            k_eff = min(k, len(variances))
            fraction = float(cumulative[k_eff - 1] / total_variance)
        variance_fractions.append({"k": k, "variance_fraction": fraction})

    return {
        "num_samples": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "numerical_rank": int(nonzero.size),
        "effective_rank": effective_rank,
        "top_k_variance": variance_fractions,
    }


def _encoder_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    if isinstance(checkpoint.get("encoder"), dict):
        return checkpoint["encoder"]
    if isinstance(checkpoint.get("model"), dict):
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value
            for key, value in checkpoint["model"].items()
        }
        encoder = {
            key[len("encoder.") :]: value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        if encoder:
            return encoder
    raise KeyError("Checkpoint has neither an 'encoder' state nor 'model' encoder.* weights")


def load_encoder_from_checkpoint(checkpoint_path, config_path=None, device="cpu"):
    """Reconstruct a frozen encoder without loading public pretrained weights."""
    import torch
    import yaml

    from src.models.factory import init_probe_encoder

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path:
        with open(config_path, "r") as handle:
            args = yaml.load(handle, Loader=yaml.FullLoader)
    else:
        args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError("Provide --config when the checkpoint does not embed its args")

    args = copy.deepcopy(args)
    meta = args.setdefault("meta", {})
    data = args.setdefault("data", {})
    mask = args.setdefault("mask", {})
    model_name = meta["model_name"]
    crop_size = int(data.get("crop_size", meta.get("crop_size", 224)))
    patch_size = int(mask.get("patch_size", meta.get("patch_size", 16)))
    encoder = init_probe_encoder(
        device=torch.device(device),
        model_name=model_name,
        patch_size=patch_size,
        crop_size=crop_size,
        use_gradient_checkpointing=False,
        pretrained=False,
        meta_args=meta,
        data_args=data,
        mask_args=mask,
    )
    state = _encoder_state(checkpoint)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()
    return encoder, args


def extract_pooled_features(
    encoder,
    loader,
    representation_type="last_avgpool",
    device="cpu",
    max_images=5000,
):
    """Extract the same pooled representation used by ViTClassifier."""
    import torch

    encoder.eval()
    chunks = []
    total = 0
    with torch.no_grad():
        for batch in loader:
            if not batch:
                continue
            images = batch[0].to(device)
            remaining = int(max_images) - total
            if remaining <= 0:
                break
            if images.shape[0] > remaining:
                images = images[:remaining]
            if representation_type == "last_avgpool":
                features = encoder(images).mean(dim=1)
            elif representation_type == "last4_avgpool_concat":
                _, layers = encoder(
                    images, return_layer_outputs=True, num_last_layers=4
                )
                features = torch.cat([layer.mean(dim=1) for layer in layers], dim=-1)
            elif representation_type == "cls":
                if not hasattr(encoder, "forward_cls"):
                    raise ValueError("CLS representation requires an external encoder")
                features = encoder.forward_cls(images)
            else:
                raise ValueError(f"Unknown representation_type: {representation_type}")
            chunks.append(features.detach().float().cpu().numpy())
            total += int(features.shape[0])
            if total >= max_images:
                break
    if not chunks:
        raise ValueError("No features were extracted")
    return np.concatenate(chunks, axis=0)


def _wilds_loader(args, transform):
    import torch
    from wilds import get_dataset

    dataset = get_dataset(
        dataset="iwildcam", download=args.download, root_dir=args.root_path
    )
    subset = dataset.get_subset(args.split, transform=transform)
    n = min(int(args.num_images), len(subset))
    if n == 0:
        raise ValueError(f"WILDS split {args.split!r} is empty")
    subset = torch.utils.data.Subset(subset, range(n))
    return torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", help="saved [N, D] .npy feature matrix")
    source.add_argument("--checkpoint", help="I-JEPA or supervised checkpoint")
    parser.add_argument("--config", help="YAML config; optional if checkpoint embeds args")
    parser.add_argument("--root-path", default="./wilds_data")
    parser.add_argument("--split", default="val")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--representation", choices=("last_avgpool", "last4_avgpool_concat", "cls"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    parser.add_argument("--save-features", help="optional .npy output for extracted features")
    parser.add_argument("--output", default="feature_rank.json")
    args = parser.parse_args(argv)
    if args.num_images < 1 or args.batch_size < 1:
        raise ValueError("num-images and batch-size must be positive")

    metadata = {}
    if args.features:
        features = np.load(args.features, allow_pickle=False)
        metadata["features_path"] = args.features
    else:
        from src.transforms import make_transform_eval, resolve_normalization

        encoder, config = load_encoder_from_checkpoint(
            args.checkpoint, config_path=args.config, device=args.device
        )
        data = config.get("data", {})
        meta = config.get("meta", {})
        crop_size = int(data.get("crop_size", meta.get("crop_size", 224)))
        transform = make_transform_eval(
            crop_size=crop_size,
            normalization=resolve_normalization(data.get("normalization", "imagenet")),
        )
        loader = _wilds_loader(args, transform)
        representation = args.representation or meta.get("representation_type", "last_avgpool")
        features = extract_pooled_features(
            encoder,
            loader,
            representation_type=representation,
            device=args.device,
            max_images=args.num_images,
        )
        metadata.update(
            {
                "checkpoint": args.checkpoint,
                "model_name": meta.get("model_name"),
                "split": args.split,
                "representation_type": representation,
            }
        )
        if args.save_features:
            np.save(args.save_features, features)
            metadata["saved_features"] = args.save_features

    statistics = compute_feature_statistics(features, top_k=args.top_k)
    statistics.update(metadata)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(statistics, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return statistics


if __name__ == "__main__":
    main()
