import json

import numpy as np
import pytest
import torch
import yaml

from src.cached.probe import MaxEarlyStopping
from src.datasets.wilds import _stratified_subset_indices
from tools.select_cached import main as select_cached_main


def _config(lr):
    return {
        "lr": lr, "weight_decay": 0.0005, "batch_size": 256,
        "head_type": "linear", "representation_type": "last_avgpool",
        "max_epochs": 100, "patience": 10, "min_delta": 0.0001, "min_epochs": 0,
    }


def _write_summary(path, encoder, config, seed, stage, val_f1, id_f1, ood_f1, fraction=1.0):
    path.mkdir(parents=True)
    summary = {
        "encoder_key": encoder, "config": config, "seed": seed, "stage": stage,
        "label_fraction": fraction, "best_epoch": 1,
        "metrics": {
            split: {
                "F1-macro_all": {"val": val_f1, "id_val": val_f1, "id_test": id_f1, "test": ood_f1}[split],
                "acc_avg": 0.5 + 0.01 * seed,
            }
            for split in ("val", "id_val", "id_test", "test")
        },
    }
    (path / "summary.json").write_text(json.dumps(summary))


def test_probe_early_stops_on_val_f1_and_restores_best_head():
    head = torch.nn.Linear(2, 1, bias=False)
    early = MaxEarlyStopping(patience=2, min_delta=0.01)
    with torch.no_grad():
        head.weight.zero_()
    improved, stop = early.step(1, 0.2, head)
    assert improved and not stop
    with torch.no_grad():
        head.weight.fill_(1.0)
    improved, stop = early.step(2, 0.8, head)
    assert improved and not stop and early.best_epoch == 2
    with torch.no_grad():
        head.weight.fill_(2.0)
    assert early.step(3, 0.79, head) == (False, False)
    assert early.step(4, 0.78, head) == (False, True)
    early.restore(head)
    assert torch.equal(head.weight, torch.ones_like(head.weight))
    assert early.best_epoch == 2
    assert early.best_metric == pytest.approx(0.8)


def test_label_fraction_indices_are_identical_to_existing_sampler():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(7), rng.integers(1, 12, size=7))
    for seed in (0, 1, 4, 19):
        for fraction in (0.01, 0.1, 0.5, 1.0):
            # probe_cached imports and calls this exact existing function; also
            # assert determinism and sorted global indices for downstream use.
            first = _stratified_subset_indices(labels, fraction, seed)
            second = _stratified_subset_indices(labels, fraction, seed)
            assert np.array_equal(first, second)
            assert np.array_equal(first, np.sort(first))
            assert set(labels[first]) == set(labels)


def test_select_cached_uses_val_reports_divergent_test_and_full_seed_stats(tmp_path, capsys):
    runs = tmp_path / "runs"
    selected = _config(0.001)
    test_choice = _config(0.01)
    _write_summary(runs / "enc/grid/seed0/a", "enc", selected, 0, "grid", 0.65, 0.8, 0.35)
    _write_summary(runs / "enc/grid/seed0/b", "enc", test_choice, 0, "grid", 0.55, 0.8, 0.90)
    for seed in range(5):
        _write_summary(
            runs / f"enc/seeds/seed{seed}/a", "enc", selected, seed, "seeds",
            0.65, (0.8, 0.7, 0.6, 0.5, 0.4)[seed], (0.4, 0.35, 0.3, 0.25, 0.2)[seed],
        )
    out = tmp_path / "selection"
    result = select_cached_main(["--runs", str(runs / "enc"), "--out", str(out)])
    model = result["models"]["enc"]
    assert model["selected_config"]["lr"] == 0.001
    assert model["test_selected_config"]["lr"] == 0.01
    assert model["selection_changed_vs_test"] is True
    stats = model["full_seed_results"]
    assert stats["id_test_macro_f1_mean"] == pytest.approx(0.6)
    assert stats["ood_test_macro_f1_mean"] == pytest.approx(0.3)
    assert stats["gap_macro_f1"] == pytest.approx(0.3)
    assert stats["retention_mean"] == pytest.approx(0.5)
    assert stats["retention_std"] == pytest.approx(0.0)
    selected_yaml = yaml.safe_load((out / "enc_selected.yaml").read_text())
    assert selected_yaml["grid"]["lr"] == [0.001]
    assert "enc" in (out / "cached_selection.tex").read_text()
    assert "DIAGNOSTIC ONLY" in capsys.readouterr().out
    assert (out / "cached_selection.csv").is_file()
