#!/usr/bin/env python3
"""Extract resumable frozen-encoder features for the cached probe protocol."""

import argparse
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cached.features import (
    encoder_key,
    load_encoder_config,
    representations_for_batch,
    resolve_checkpoint_path,
    sha256_file,
)
from src.transforms import make_transform_eval, resolve_normalization


SPLITS = ("train", "val", "id_val", "id_test", "test")
DTYPES = {"float16": np.float16, "float32": np.float32}


def _atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, "r") as handle:
        return json.load(handle)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _config_without_loading_encoder(config_path):
    with open(config_path, "r") as handle:
        args = yaml.load(handle, Loader=yaml.FullLoader)
    checkpoint = resolve_checkpoint_path(args["meta"]) if args["meta"].get("load_checkpoint") else None
    return args, {
        "args": args,
        "seed": int(args["meta"].get("seed", 0)),
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint and os.path.isfile(checkpoint) else None,
        "config_path": os.path.abspath(config_path),
        "config_sha256": sha256_file(config_path),
    }


class WildsSplitFactory:
    """Load the WILDS metadata once and apply eval transforms per split."""

    def __init__(self, args):
        self.args = args
        self.dataset = None

    def __call__(self, split, transform):
        if self.dataset is None:
            from wilds import get_dataset

            data = self.args["data"]
            self.dataset = get_dataset(
                dataset="iwildcam",
                download=bool(data.get("download", True)),
                root_dir=data.get("root_path", "./wilds_data"),
            )
        return self.dataset.get_subset(split, transform=transform)


def _split_files(out_dir, split, representations):
    files = {rep: f"{split}_{rep}.npy" for rep in representations}
    files["y"] = f"{split}_y.npy"
    files["metadata"] = f"{split}_metadata.npy"
    return files


def _completed_split(manifest, out_dir, split):
    info = manifest.get("splits", {}).get(split)
    if not info or not info.get("complete"):
        return False
    return all(os.path.isfile(os.path.join(out_dir, filename)) for filename in info["files"].values())


def _disk_bytes(out_dir):
    total = 0
    for filename in os.listdir(out_dir):
        if filename.endswith(".npy"):
            total += os.path.getsize(os.path.join(out_dir, filename))
    return total


def extract_features(
    config_path,
    out_dir,
    splits=SPLITS,
    batch_size=16,
    num_workers=4,
    dtype="float16",
    limit=None,
    device="auto",
    encoder=None,
    dataset_factory=None,
    batch_observer=None,
):
    """Extract selected splits; internal factories enable tiny CPU fixtures."""
    start_time = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    if encoder is None:
        encoder, info = load_encoder_config(config_path, device)
        args = info["args"]
    else:
        args, info = _config_without_loading_encoder(config_path)
        encoder = encoder.to(device).eval()
    key = encoder_key(config_path, args)
    signature = {
        "encoder_key": key,
        "config_sha256": info["config_sha256"],
        "checkpoint_sha256": info["checkpoint_sha256"],
        "dtype": dtype,
        "limit": limit,
    }
    manifest = _read_json(os.path.join(out_dir, "manifest.json"), default={})
    if manifest and manifest.get("signature") != signature:
        raise ValueError("Existing manifest belongs to a different extraction configuration")
    requested = [split for split in splits if split not in manifest.get("splits", {}) or not _completed_split(manifest, out_dir, split)]
    skipped = [split for split in splits if split not in requested]
    if skipped:
        print("Skipping completed splits:", ", ".join(skipped))
    if not requested:
        print(f"Feature cache is complete: {_disk_bytes(out_dir) / 1e9:.3f} GB in {out_dir}")
        return manifest

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data_args = args["data"]
    transform = make_transform_eval(
        crop_size=int(data_args["crop_size"]),
        normalization=resolve_normalization(data_args.get("normalization", "imagenet")),
    )
    dataset_factory = dataset_factory or WildsSplitFactory(args)
    manifest.update(
        {
            "format_version": 1,
            "signature": signature,
            "encoder_key": key,
            "model_name": args["meta"]["model_name"],
            "num_classes": int(args["meta"]["num_classes"]),
            "embed_dim": int(args["meta"]["embed_dim"]),
            "crop_size": int(data_args["crop_size"]),
            "patch_size": int(args["mask"]["patch_size"]),
            "normalization": data_args.get("normalization", "imagenet"),
            "config_path": info["config_path"],
            "config_sha256": info["config_sha256"],
            "checkpoint_path": info["checkpoint_path"],
            "checkpoint_sha256": info["checkpoint_sha256"],
            "checkpoint_sha256_kind": info.get("checkpoint_sha256_kind"),
            "pretrained": bool(args["meta"].get("pretrained", True)),
            "random_init_seed": info["seed"] if not args["meta"].get("load_checkpoint") else None,
            "torch_version": torch.__version__,
            "timm_version": _package_version("timm"),
            "git_commit": _git_commit(),
            "splits": manifest.get("splits", {}),
        }
    )

    for split in requested:
        subset = dataset_factory(split, transform)
        available = len(subset)
        count = min(available, limit) if limit is not None else available
        if count < 1:
            raise ValueError(f"Split {split} has no images to extract")
        split_dir = out_dir
        progress_path = os.path.join(split_dir, f".{split}_progress.json")
        progress = _read_json(progress_path, default={})
        if progress and progress.get("signature") != signature:
            raise ValueError(f"Partial split {split} has a different extraction signature")
        next_index = int(progress.get("next_index", 0))
        if next_index > count:
            raise ValueError(f"Partial split {split} has more rows than requested")
        arrays = None
        partial_paths = {}
        final_files = None

        remaining = torch.utils.data.Subset(subset, range(next_index, count))
        loader = torch.utils.data.DataLoader(
            remaining,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        with torch.no_grad():
            for images, labels, metadata in loader:
                images = images.to(device, non_blocking=True)
                use_bf16 = bool(args["meta"].get("use_bfloat16", True)) and device.type == "cuda"
                with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                    outputs = representations_for_batch(encoder, images)
                current_batch = int(labels.shape[0])
                if arrays is None:
                    representations = tuple(outputs)
                    final_files = _split_files(split_dir, split, representations)
                    if progress:
                        if tuple(progress.get("representations", ())) != representations:
                            raise ValueError(f"Partial split {split} representation mismatch")
                    # Initialize explicitly after learning all representation keys.
                    arrays = {}
                    for rep, tensor in outputs.items():
                        partial = os.path.join(split_dir, f".{split}_{rep}.partial.npy")
                        partial_paths[rep] = (partial, os.path.join(split_dir, final_files[rep]))
                        mode = "r+" if os.path.isfile(partial) else "w+"
                        arrays[rep] = np.lib.format.open_memmap(
                            partial, mode=mode, dtype=DTYPES[dtype], shape=(count, int(tensor.shape[-1]))
                        )
                    for name in ("y", "metadata"):
                        partial_paths[name] = (
                            os.path.join(split_dir, f".{split}_{name}.partial.npy"),
                            os.path.join(split_dir, final_files[name]),
                        )
                    arrays["y"] = np.lib.format.open_memmap(
                        partial_paths["y"][0],
                        mode="r+" if os.path.isfile(partial_paths["y"][0]) else "w+",
                        dtype=np.int64,
                        shape=(count,),
                    )
                    arrays["metadata"] = np.lib.format.open_memmap(
                        partial_paths["metadata"][0],
                        mode="r+" if os.path.isfile(partial_paths["metadata"][0]) else "w+",
                        dtype=np.int64,
                        shape=(count, int(metadata.shape[1])),
                    )

                end = next_index + current_batch
                for rep, tensor in outputs.items():
                    values = tensor.detach().float().cpu().numpy()
                    arrays[rep][next_index:end] = values.astype(DTYPES[dtype], copy=False)
                    arrays[rep].flush()
                arrays["y"][next_index:end] = labels.numpy().astype(np.int64, copy=False)
                arrays["metadata"][next_index:end] = metadata.numpy().astype(np.int64, copy=False)
                arrays["y"].flush()
                arrays["metadata"].flush()
                next_index = end
                progress = {
                    "signature": signature,
                    "split": split,
                    "next_index": next_index,
                    "count": count,
                    "representations": tuple(outputs),
                }
                if next_index < count:
                    _atomic_json(progress_path, progress)
                if batch_observer is not None:
                    batch_observer(next_index)
                if next_index > count:
                    raise RuntimeError("Wrote beyond the declared split size")

        if next_index != count or arrays is None:
            raise RuntimeError(f"Split {split} ended at {next_index}/{count} rows")
        for array in arrays.values():
            array.flush()
        for partial, final in partial_paths.values():
            os.replace(partial, final)
        disk_bytes = sum(
            os.path.getsize(os.path.join(split_dir, filename))
            for filename in final_files.values()
        )
        manifest["splits"][split] = {
            "complete": True,
            "num_images": count,
            "available_images": available,
            "representations": {rep: int(arrays[rep].shape[-1]) for rep in arrays if rep not in ("y", "metadata")},
            "files": final_files,
            "disk_bytes": disk_bytes,
        }
        manifest["wall_time_seconds"] = float(time.perf_counter() - start_time)
        manifest["disk_bytes"] = _disk_bytes(out_dir)
        if device.type == "cuda":
            manifest["peak_gpu_alloc_gb"] = float(torch.cuda.max_memory_allocated(device) / 1e9)
            manifest["peak_gpu_reserved_gb"] = float(torch.cuda.max_memory_reserved(device) / 1e9)
        else:
            manifest["peak_gpu_alloc_gb"] = None
            manifest["peak_gpu_reserved_gb"] = None
        _atomic_json(os.path.join(out_dir, "manifest.json"), manifest)
        if os.path.exists(progress_path):
            os.remove(progress_path)
        print(f"Completed {split}: {count} images, {disk_bytes / 1e9:.3f} GB")

    print(f"Feature cache: {_disk_bytes(out_dir) / 1e9:.3f} GB in {out_dir}")
    return manifest


def _direct_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    return extract_features(
        config_path=args.config,
        out_dir=args.out,
        splits=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dtype=args.dtype,
        limit=args.limit,
        device=args.device,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--submitit", action="store_true")
    parser.add_argument("--partition")
    parser.add_argument("--time", type=int, default=120)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--submitit-folder", default="submitit_logs")
    submit_args, remaining = parser.parse_known_args(argv)
    if not submit_args.submitit:
        return _direct_main(remaining)
    if submit_args.gpus != 1 or not submit_args.partition:
        raise ValueError("--submitit requires --partition and exactly --gpus 1")
    import submitit

    executor = submitit.SlurmExecutor(
        folder=os.path.join(submit_args.submitit_folder, "job_%j"), max_num_timeout=20
    )
    executor.update_parameters(
        partition=submit_args.partition,
        time=submit_args.time,
        nodes=1,
        ntasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=10,
        mem_per_gpu="180G",
    )
    job = executor.submit(_direct_main, copy.deepcopy(remaining))
    print(job.job_id)
    return job


if __name__ == "__main__":
    main()
