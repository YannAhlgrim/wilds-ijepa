import json

import pytest

from tools import probe_cached, select_cached


def _summary(config, val_f1):
    return {
        "format_version": 1,
        "encoder_key": "synthetic",
        "config": config,
        "seed": 0,
        "stage": "grid",
        "label_fraction": 1.0,
        "metrics": {
            split: {"F1-macro_all": val_f1, "acc_avg": 0.5}
            for split in ("val", "id_val", "id_test", "test")
        },
    }


def test_cls_grid_reuses_160_default_runs_and_selects_over_240(tmp_path, monkeypatch):
    _, defaults = probe_cached.load_grid("configs/grids/cached/default.yaml")
    _, extended = probe_cached.load_grid("configs/grids/cached/with_cls.yaml")
    assert len(defaults) == 160
    assert len(extended) == 240
    default_identities = {
        probe_cached.config_slug(config) for config in defaults
    }
    assert all(probe_cached.config_slug(config) in default_identities for config in defaults)

    out = tmp_path / "runs"
    features = tmp_path / "features"
    features.mkdir()
    manifest = {
        "encoder_key": "synthetic",
        "num_classes": 3,
        "splits": {
            split: {"complete": True, "files": {"cls": f"{split}_cls.npy"}}
            for split in ("train", "val", "id_val", "id_test", "test")
        },
    }
    (features / "manifest.json").write_text(json.dumps(manifest))
    (features / "train_cls.npy").write_bytes(b"presence checked before mocked execution")

    for config in defaults:
        run_dir = (
            out / "synthetic" / "grid" / "seed0" / probe_cached.config_slug(config)
        )
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps(_summary(config, 0.6)))

    executed_cls = []
    original = probe_cached.run_one_configuration

    def execute_or_skip(*args, **kwargs):
        config = args[2]
        if config["representation_type"] != "cls":
            return original(*args, **kwargs)
        executed_cls.append(config)
        run_dir = (
            out / "synthetic" / "grid" / "seed0" / probe_cached.config_slug(config)
        )
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps(_summary(config, 0.5)))
        return _summary(config, 0.5)

    monkeypatch.setattr(probe_cached, "run_one_configuration", execute_or_skip)
    probe_cached.main([
        "--features", str(features), "--grid", "configs/grids/cached/with_cls.yaml",
        "--seeds", "0", "--stage", "grid", "--out", str(out), "--max-epochs", "1",
    ])
    assert len(executed_cls) == 80
    assert all(config["representation_type"] == "cls" for config in executed_cls)

    result = select_cached.main([
        "--runs", str(out / "synthetic"), "--out", str(tmp_path / "selection")
    ])
    assert len(result["configurations"]) == 240
    assert len(result["models"]["synthetic"]["selection_rows"]) == 240
    # Existing non-CLS configurations remain eligible and win this fixture.
    assert result["models"]["synthetic"]["selected_config"]["representation_type"] != "cls"


def test_probe_rejects_cls_grid_without_manifest_cls(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    (features / "manifest.json").write_text(json.dumps({
        "encoder_key": "synthetic", "num_classes": 3,
        "splits": {split: {"files": {}} for split in ("train", "val", "id_val", "id_test", "test")},
    }))
    with pytest.raises(ValueError, match="manifest lacks CLS"):
        probe_cached.main([
            "--features", str(features), "--grid", "configs/grids/cached/with_cls.yaml",
            "--seeds", "0", "--stage", "grid", "--out", str(tmp_path / "runs"),
        ])
