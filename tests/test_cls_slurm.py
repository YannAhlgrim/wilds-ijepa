import json
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_probe_sbatch_rejects_cls_grid_without_cached_cls(tmp_path):
    feature_root = tmp_path / "features"
    feature_dir = feature_root / "enc"
    feature_dir.mkdir(parents=True)
    base_files = {
        "last_avgpool": "last_avgpool.npy",
        "last4_avgpool_concat": "last4.npy",
        "y": "y.npy",
        "metadata": "metadata.npy",
    }
    manifest = {"splits": {}}
    # The checker only verifies existence; no arrays are loaded before refusal.
    for split in ("train", "val", "id_val", "id_test", "test"):
        files = {key: f"{split}_{name}" for key, name in base_files.items()}
        manifest["splits"][split] = {"complete": True, "files": files}
        for filename in files.values():
            (feature_dir / filename).write_bytes(b"")
    (feature_dir / "manifest.json").write_text(json.dumps(manifest))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\necho 'Fake GPU'\n")
    nvidia.chmod(0o755)
    env = dict(os.environ)
    env.update(
        SLURM_SUBMIT_DIR=ROOT, KEY="enc", FEATURES_ROOT=str(feature_root),
        PYTHON=sys.executable, GRID="with_cls", PATH=str(fake_bin) + os.pathsep + env["PATH"],
    )
    env.pop("VENV", None)
    result = subprocess.run(
        ["bash", os.path.join(ROOT, "tools/slurm/probe_cached.sbatch")], env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert result.returncode != 0
    assert "GRID requires CLS" in result.stderr
