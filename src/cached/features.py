"""Encoder construction and one-pass representations for feature caching."""

import hashlib
import logging
import os
import random

import numpy as np
import torch
import yaml

from src.models.factory import init_probe_encoder


LOGGER = logging.getLogger(__name__)
REPRESENTATIONS = ("last_avgpool", "last4_avgpool_concat", "cls")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(state_dict):
    """Fingerprint loaded weights without serializing a second full checkpoint."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def resolve_checkpoint_path(meta_args):
    filename = meta_args.get("read_checkpoint")
    folder = meta_args.get("checkpoint_folder")
    if filename is None:
        return meta_args.get("checkpoint_path")
    if os.path.isabs(filename) or folder is None:
        return filename
    return os.path.join(folder, filename)


def _strip_module_prefix(state):
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_encoder_config(config_path, device):
    """Reproduce the frozen probe's initialization/checkpoint path exactly."""
    with open(config_path, "r") as handle:
        args = yaml.load(handle, Loader=yaml.FullLoader)
    meta = args["meta"]
    data = args["data"]
    mask = args["mask"]
    seed = int(meta.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    encoder = init_probe_encoder(
        device=device,
        patch_size=int(mask["patch_size"]),
        crop_size=int(data["crop_size"]),
        model_name=meta["model_name"],
        use_gradient_checkpointing=bool(meta.get("use_gradient_checkpointing", True)),
        pretrained=bool(meta.get("pretrained", True)),
        meta_args=meta,
        data_args=data,
        mask_args=mask,
    )
    checkpoint_path = None
    checkpoint_sha256 = None
    if meta.get("load_checkpoint", False):
        checkpoint_path = resolve_checkpoint_path(meta)
        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("encoder") if isinstance(checkpoint, dict) else None
        if state is None and isinstance(checkpoint, dict) and "model" in checkpoint:
            model_state = _strip_module_prefix(checkpoint["model"])
            state = {
                key[len("encoder.") :]: value
                for key, value in model_state.items()
                if key.startswith("encoder.")
            }
        if state is None:
            raise KeyError(f"No encoder weights in checkpoint: {checkpoint_path}")
        state = _strip_module_prefix(state)
        incompatible = encoder.load_state_dict(state, strict=False)
        LOGGER.info("Loaded %s with %s", checkpoint_path, incompatible)
        checkpoint_sha256 = sha256_file(checkpoint_path)

    checkpoint_sha256_kind = "file" if checkpoint_sha256 is not None else None
    from src.models.external_encoders import get_external_encoder_spec, is_external_encoder

    if checkpoint_path is None and is_external_encoder(meta["model_name"]):
        spec = get_external_encoder_spec(meta["model_name"])
        checkpoint_path = f"timm://{spec.timm_name}"
        checkpoint_sha256 = sha256_state_dict(encoder.state_dict())
        checkpoint_sha256_kind = "loaded_encoder_state"

    encoder.requires_grad_(False)
    encoder.eval()
    info = {
        "args": args,
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_sha256_kind": checkpoint_sha256_kind,
        "config_path": os.path.abspath(config_path),
        "config_sha256": sha256_file(config_path),
    }
    return encoder, info


def representations_for_batch(encoder, images):
    """Compute all cached representations in one frozen encoder forward pass."""
    if hasattr(encoder, "forward_for_cache"):
        final, layers, cls = encoder.forward_for_cache(images)
    else:
        final, layers = encoder(images, return_layer_outputs=True, num_last_layers=4)
        cls = None
    result = {
        "last_avgpool": final.mean(dim=1),
        "last4_avgpool_concat": torch.cat(
            [layer.mean(dim=1) for layer in layers[-4:]], dim=-1
        ),
    }
    if cls is not None:
        result["cls"] = cls
    return result


def encoder_key(config_path, args):
    submission = args.get("meta", {}).get("submission_model_name")
    if submission:
        return str(submission)
    filename = os.path.splitext(os.path.basename(config_path))[0]
    return filename[len("supervised_") :] if filename.startswith("supervised_") else filename
