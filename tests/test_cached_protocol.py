import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

from tools.probe_cached import load_grid


def _write_synthetic_cache(root):
    root.mkdir()
    rng = np.random.default_rng(12)
    counts = {"train": 24, "val": 12, "id_val": 12, "id_test": 10, "test": 11}
    for split, count in counts.items():
        labels = np.arange(count) % 3
        features = rng.normal(size=(count, 4)).astype(np.float16)
        np.save(root / f"{split}_last_avgpool.npy", features)
        np.save(root / f"{split}_last4_avgpool_concat.npy", rng.normal(size=(count, 16)).astype(np.float16))
        np.save(root / f"{split}_y.npy", labels.astype(np.int64))
        metadata = np.zeros((count, 9), dtype=np.int64)
        metadata[:, -1] = labels
        np.save(root / f"{split}_metadata.npy", metadata)
    manifest = {
        "encoder_key": "synthetic", "num_classes": 3, "embed_dim": 4,
        "splits": {split: {"complete": True, "num_images": count} for split, count in counts.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_default_cached_grid_has_160_configs():
    _, configs = load_grid("configs/grids/cached/default.yaml")
    assert len(configs) == 160
    assert {config["batch_size"] for config in configs} == {256, 1024}


def test_cached_protocol_end_to_end_cpu_smoke(tmp_path):
    features = tmp_path / "features"
    out = tmp_path / "runs"
    _write_synthetic_cache(features)
    grid = tmp_path / "tiny.yaml"
    grid.write_text(
        yaml.safe_dump(
            {
                "defaults": {"max_epochs": 1, "patience": 1, "min_delta": 0.0001},
                "grid": {
                    "lr": [0.001], "weight_decay": [0.0], "batch_size": [8],
                    "head_type": ["linear"], "representation_type": ["last_avgpool"],
                },
            }
        )
    )
    env = dict(os.environ)
    env.update(
        PYTHON=sys.executable,
        DEVICE="cpu",
        GRID=str(grid),
        MAX_EPOCHS="1",
        SEEDS="0 1",
    )
    started = time.monotonic()
    completed = subprocess.run(
        ["bash", "tools/run_cached_protocol.sh", "synthetic", str(features), str(out)],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=110,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert time.monotonic() - started < 120
    summaries = list(out.rglob("summary.json"))
    # 1 grid + 2 full-seed + (3 fractions x 2 seeds) = 9 runs.
    assert len(summaries) == 9
    for path in summaries:
        summary = json.loads(path.read_text())
        assert summary["best_epoch"] == 1
        assert summary["metrics"]["val"]["F1-macro_all"] >= 0
        assert (path.parent / "epochs.csv").is_file()
    selection = json.loads((out / "synthetic/selection/cached_selection.json").read_text())
    assert selection["models"]["synthetic"]["full_seed_results"]["num_seeds"] == 2
    assert set(selection["models"]["synthetic"]["label_efficiency"]) == {"0.01", "0.1", "0.5"}
    assert (out / "synthetic/selection/cached_selection.tex").is_file()
