import csv
import json
import os
import subprocess
import sys

import pytest
import yaml

from tools import summarize_grid
from tools.select_hparams import (
    ID_FILE,
    OOD_FILE,
    VAL_FILE,
    best_validation_from_csv,
    main as select_main,
    normalize_config_name,
)


def _write_metrics(run, split, filename, f1, acc, seed):
    (run / filename).write_text(
        json.dumps(
            {
                "split": split,
                "F1-macro_all": f1,
                "acc_avg": acc,
                "run_info": {"seed": seed},
            }
        )
    )


def _make_run(
    root,
    name,
    seed,
    *,
    val_f1=None,
    id_f1=0.8,
    ood_f1=0.4,
    val_acc=0.7,
    id_acc=0.75,
    ood_acc=0.6,
    model="fixture",
):
    run = root / name
    run.mkdir(parents=True)
    (run / "params.yaml").write_text(
        yaml.safe_dump({"meta": {"model_name": "vit_huge", "submission_model_name": model}})
    )
    if val_f1 is not None:
        _write_metrics(run, "val", VAL_FILE, val_f1, val_acc, seed)
    _write_metrics(run, "id_test", ID_FILE, id_f1, id_acc, seed)
    _write_metrics(run, "test", OOD_FILE, ood_f1, ood_acc, seed)
    return run


def _write_training_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_loss", "val_loss", "val_acc"))
        writer.writerows(rows)


def test_validation_selection_and_test_diagnostic_disagree(tmp_path, capsys):
    root = tmp_path / "experiment_logs" / "eval-wilds"
    for seed in (0, 1):
        # Validation prefers lr0.001 (mean .65 vs .525); test prefers lr0.01.
        _make_run(
            root,
            f"fixture-lr0.001-seed{seed}",
            seed,
            val_f1=(0.60, 0.70)[seed],
            id_f1=(0.8, 0.6)[seed],
            ood_f1=(0.4, 0.3)[seed],
            id_acc=(0.7, 0.8)[seed],
            ood_acc=(0.5, 0.4)[seed],
        )
        _make_run(
            root,
            f"fixture-lr0.01-seed{seed}",
            seed,
            val_f1=(0.50, 0.55)[seed],
            ood_f1=(0.90, 0.80)[seed],
        )
    out = tmp_path / "selection"
    result = select_main(
        ["--root", str(root), "--model-prefix", "fixture", "--out", str(out)]
    )
    selection = result["models"]["fixture"]
    assert selection == {
        "selected_config": "fixture-lr0.001",
        "selection_criterion": "val_macro_f1",
        "test_selected_config": "fixture-lr0.01",
        "selection_changed_vs_test": True,
    }
    selected = next(row for row in result["configurations"] if row["selected_by_val"])
    assert selected["val_criterion_mean"] == pytest.approx(0.65)
    assert selected["id_test_macro_f1_mean"] == pytest.approx(0.70)
    assert selected["ood_test_macro_f1_mean"] == pytest.approx(0.35)
    assert selected["id_test_avg_acc_mean"] == pytest.approx(0.75)
    assert selected["ood_test_avg_acc_mean"] == pytest.approx(0.45)
    assert selected["retention_mean"] == pytest.approx(0.5)
    assert selected["retention_std"] == pytest.approx(0.0)
    test_selected = next(row for row in result["configurations"] if row["selected_by_test"])
    assert test_selected["config"] == "fixture-lr0.01"
    console = capsys.readouterr().out
    assert "Test-based selection is not valid for reporting" in console
    assert "Retention R: 0.5000 ± 0.0000" in console
    written = json.loads((out / "hparam_selection.json").read_text())
    assert written["models"] == result["models"]
    with open(out / "hparam_selection.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["criterion"] for row in rows} == {"val_macro_f1"}
    assert sum(row["selected_by_val"] == "True" for row in rows) == 1


def test_csv_loss_fallback_searches_old_and_new_log_locations_and_uses_acc_tiebreak(tmp_path):
    root = tmp_path / "experiment_logs" / "eval-wilds"
    train_root = tmp_path / "experiment_logs"
    for config in ("fallback-lr0.001", "fallback-lr0.01"):
        for seed in (0, 1):
            _make_run(root, f"{config}-seed{seed}", seed, model="fallback")
    # Both configurations have mean minimum loss .3. The second has higher
    # mean accuracy at its best epochs (.85 vs .60), so it must win the tie.
    _write_training_csv(
        train_root / "fallback-lr0.001-seed0" / "linear_probe_r0.csv",
        [(1, 1.0, 0.5, 0.1), (2, 0.9, 0.2, 0.5)],
    )
    _write_training_csv(
        train_root / "fallback-lr0.001-seed1" / "linear_probe_r0.csv",
        [(1, 1.0, float("nan"), 0.2), (2, 0.9, 0.4, 0.7)],
    )
    _write_training_csv(
        train_root / "fallback-lr0.01-seed0" / "linear_probe_r0.csv",
        [(1, 1.0, 0.3, 0.8)],
    )
    # New baseline-style retained log lives beside the eval metrics.
    _write_training_csv(
        root / "fallback-lr0.01-seed1" / "linear_probe_r0.csv",
        [(1, 1.0, 0.4, 0.7), (2, 0.9, 0.3, 0.9)],
    )
    loss, acc, epoch = best_validation_from_csv(
        train_root / "fallback-lr0.001-seed0" / "linear_probe_r0.csv"
    )
    assert (loss, acc, epoch) == (0.2, 0.5, 2)

    result = select_main(
        [
            "--root", str(root), "--train-root", str(train_root),
            "--model-prefix", "fallback", "--out", str(tmp_path / "out"),
        ]
    )
    selection = result["models"]["fallback"]
    assert selection["selection_criterion"] == "best_val_loss"
    assert selection["selected_config"] == "fallback-lr0.01"
    rows = {row["config"]: row for row in result["configurations"]}
    assert rows["fallback-lr0.001"]["val_criterion_mean"] == pytest.approx(0.3)
    assert rows["fallback-lr0.01"]["val_criterion_mean"] == pytest.approx(0.3)
    assert rows["fallback-lr0.01"]["val_acc_mean"] == pytest.approx(0.85)
    assert rows["fallback-lr0.01"]["val_rank"] == 1


def test_config_normalization_removes_seed_and_fraction_tags():
    assert normalize_config_name("model-lf0.01-eval1-seed3") == "model-eval1"
    assert normalize_config_name("model-frac0.10-seed2") == "model"
    assert normalize_config_name("model-label_efficiency-lf1-seed0") == "model"


def test_summarize_grid_defaults_to_val_and_test_warns(tmp_path):
    root = tmp_path / "eval-wilds"
    _make_run(root, "fixture-a-seed0", 0, val_f1=0.4, ood_f1=0.9)
    rows = summarize_grid._collect_rows(str(root), "F1-macro_all", [])
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.4)
    command = [
        sys.executable,
        "tools/summarize_grid.py",
        "--root", str(root), "--metric", "F1-macro_all", "--split", "test",
    ]
    completed = subprocess.run(
        command,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "not valid for reporting" in completed.stderr
    assert "0.9" in completed.stdout
