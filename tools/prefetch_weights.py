#!/usr/bin/env python3
"""Cache public timm weights for external encoders, without GPU or WILDS data.

Downloads use the normal Hugging Face and torch caches, so HF_HOME and TORCH_HOME
are honored. Run on a login node only when policy permits; no model is built.
"""

import argparse
import json
import os
import sys
from urllib.parse import urlparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.external_encoders import EXTERNAL_ENCODERS, get_external_encoder_spec


def _split_hf_id(value):
    parts = value.split("@")
    if len(parts) > 2:
        raise ValueError(f"Invalid Hugging Face model id: {value}")
    return parts[0], parts[1] if len(parts) == 2 else None


def prefetch_key(key, downloader=None):
    """Download one registered external model using timm's resolution order."""
    import timm

    spec = get_external_encoder_spec(key)
    cfg = timm.get_pretrained_cfg(spec.timm_name)
    if not cfg.hf_hub_id:
        raise ValueError(f"External encoder {key} has no verified public timm source")

    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    repo_id, revision = _split_hf_id(cfg.hf_hub_id)
    candidates = [cfg.hf_hub_filename] if cfg.hf_hub_filename else ["model.safetensors", "pytorch_model.bin"]
    errors = []
    for filename in candidates:
        try:
            path = downloader(repo_id=repo_id, filename=filename, revision=revision)
            return {
                "key": key,
                "timm_name": spec.timm_name,
                "source": cfg.hf_hub_id,
                "filename": filename,
                "path": os.path.abspath(path),
                "bytes": os.path.getsize(path),
            }
        except EntryNotFoundError as exc:
            errors.append(str(exc))

    # Compatibility fallback for timm configurations without HF weight files.
    if cfg.url:
        from timm.models._hub import download_cached_file

        path = download_cached_file(cfg.url, progress=True)
        return {
            "key": key,
            "timm_name": spec.timm_name,
            "source": cfg.url,
            "filename": os.path.basename(urlparse(cfg.url).path),
            "path": os.path.abspath(path),
            "bytes": os.path.getsize(path),
        }
    raise RuntimeError(f"No public weight file found for {key}: {errors}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keys", nargs="+", choices=sorted(EXTERNAL_ENCODERS), default=sorted(EXTERNAL_ENCODERS)
    )
    args = parser.parse_args(argv)
    print("HF_HOME:", os.environ.get("HF_HOME", "<default>"))
    print("TORCH_HOME:", os.environ.get("TORCH_HOME", "<default>"))
    results = [prefetch_key(key) for key in args.keys]
    for result in results:
        print(
            f"{result['key']}: {result['path']} "
            f"({result['bytes'] / 1e9:.3f} GB; {result['bytes']} bytes)"
        )
    print(json.dumps({"files": results, "total_bytes": sum(row["bytes"] for row in results)}, indent=2))
    return results


if __name__ == "__main__":
    main()
