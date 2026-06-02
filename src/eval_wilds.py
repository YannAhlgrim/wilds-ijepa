import os
import sys
import json
import yaml
import logging

import numpy as np

import torch
import torch.distributed as dist

from src.datasets.wilds import make_iwildcam_eval
from src.helper import init_model
from src.models.head import ViTClassifier
from src.transforms import make_transform_eval
from src.utils.distributed import init_distributed
from src.utils.logging import resolve_log_dir


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _get_seed(args):
    return int(args.get("meta", {}).get("seed", _GLOBAL_SEED))


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_checkpoint_path(meta_args):
    if meta_args.get("checkpoint_path"):
        return meta_args["checkpoint_path"]
    folder = meta_args.get("checkpoint_folder")
    fname = meta_args.get("read_checkpoint")
    if folder is None or fname is None:
        return None
    return fname if os.path.isabs(fname) else os.path.join(folder, fname)


def _load_model_state(model, checkpoint_path, device):
    if checkpoint_path is None:
        raise ValueError("checkpoint_path or checkpoint_folder/read_checkpoint required")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"No model weights found in checkpoint: {checkpoint_path}")
    state = strip_module_prefix(checkpoint["model"])
    msg = model.load_state_dict(state, strict=True)
    model.to(device)
    logger.info(f"Loaded model from {checkpoint_path} with msg: {msg}")


def main(args):
    force_single = bool(args.get("meta", {}).get("force_single_process", False))
    if force_single:
        world_size, rank = 1, 0
    else:
        world_size, rank = init_distributed()

    if (
        not force_single
        and dist.is_available()
        and dist.is_initialized()
        and world_size > 1
        and rank != 0
    ):
        dist.barrier()
        dist.destroy_process_group()
        return

    seed = _get_seed(args)
    _set_seed(seed)

    meta_args = args.get("meta", {})
    data_args = args.get("data", {})
    log_args = args.get("logging", {})

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    folder = resolve_log_dir(args, stage="eval")
    tag = log_args.get("write_tag", "wilds_eval")

    params_path = os.path.join(folder, "params-eval.yaml")
    with open(params_path, "w") as f:
        yaml.dump(args, f)
    with open(os.path.join(folder, "params.yaml"), "w") as f:
        yaml.dump(args, f)

    model_name = meta_args["model_name"]
    embed_dim = meta_args["embed_dim"]
    num_classes = meta_args["num_classes"]
    patch_size = meta_args.get("patch_size", 16)
    crop_size = meta_args.get("crop_size", 224)
    use_bfloat16 = bool(meta_args.get("use_bfloat16", True))
    use_autocast = use_bfloat16 and torch.cuda.is_available()

    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        model_name=model_name,
    )
    model = ViTClassifier(
        encoder,
        num_classes,
        embed_dim,
        representation_type=meta_args.get("representation_type", "last_avgpool"),
        head_type=meta_args.get("head_type", "linear"),
    ).to(device)

    checkpoint_path = _resolve_checkpoint_path(meta_args)
    _load_model_state(model, checkpoint_path, device)
    model.eval()

    eval_transform = make_transform_eval(crop_size=crop_size)
    split = data_args.get("split", "test")
    root_path = data_args.get("root_path", "./wilds_data")

    full_dataset, eval_dataset, eval_loader = make_iwildcam_eval(
        transform=eval_transform,
        split=split,
        batch_size=data_args.get("batch_size", 128),
        root_path=root_path,
        num_workers=data_args.get("num_workers", 8),
        pin_mem=bool(data_args.get("pin_mem", True)),
        download=bool(data_args.get("download", True)),
    )

    all_y_pred = []
    all_y_true = []
    all_metadata = []

    with torch.no_grad():
        for batch in eval_loader:
            if len(batch) == 3:
                imgs, y_true, metadata = batch
            else:
                raise ValueError("Expected eval loader to return (x, y, metadata)")
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.bfloat16):
                logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()
            all_y_pred.append(preds)
            all_y_true.append(y_true.cpu())
            all_metadata.append(metadata.cpu())

    if all_y_pred:
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y_true = torch.cat(all_y_true, dim=0)
    all_metadata = torch.cat(all_metadata, dim=0) if all_metadata else torch.empty(0)

    if int(all_metadata.shape[0]) != int(all_y_pred.shape[0]):
        raise ValueError(
            "Metadata length mismatch with predictions: "
            f"{int(all_metadata.shape[0])} vs {int(all_y_pred.shape[0])}"
        )

    metrics = full_dataset.eval(all_y_pred, all_y_true, all_metadata)

    metrics_path = os.path.join(folder, f"{tag}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(f"Eval metrics saved to {metrics_path}")
    logger.info(f"Eval metrics: {metrics}")

    if (
        not force_single
        and dist.is_available()
        and dist.is_initialized()
        and world_size > 1
    ):
        dist.barrier()
        dist.destroy_process_group()

    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "folder": folder,
    }


if __name__ == "__main__":
    raise RuntimeError(
        "Use main_eval_wilds.py to launch this script with a config file."
    )
