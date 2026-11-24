import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.train_supervised import macro_f1_from_confusion
from tools.feature_rank import compute_feature_statistics
from tools.per_class_f1 import analyze_split, latex_table, main as per_class_main, read_predictions
from tools.plot_val_curve import extract_val_curves, write_val_curves
from tools.retention import relative_retention, summarize_retention


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_retention_is_mean_of_per_seed_ratios():
    result = summarize_retention([0.8, 0.6], [0.4, 0.24])
    assert result["per_seed"] == pytest.approx([0.5, 0.4])
    assert result["mean"] == pytest.approx(0.45)
    assert result["std"] == pytest.approx(math.sqrt(0.005))
    assert result["n"] == 2
    assert relative_retention(0.0, 0.2) is None
    assert relative_retention(None, 0.2) is None


def test_aggregate_seeds_emits_retention_next_to_gap(tmp_path):
    eval_root = tmp_path / "eval-wilds"
    for seed, id_f1, ood_f1 in ((0, 0.8, 0.4), (1, 0.6, 0.24)):
        run = eval_root / f"fixture-seed{seed}"
        run.mkdir(parents=True)
        for split, value in (("id_test", id_f1), ("test", ood_f1)):
            metrics = {
                "F1-macro_all": value,
                "acc_avg": 0.5,
                "run_info": {"seed": seed, "train_time_seconds": 10.0, "epochs_run": 3},
                "split": split,
            }
            filename = (
                "iwildcam_id_test_metrics.json"
                if split == "id_test"
                else "iwildcam_test_metrics.json"
            )
            (run / filename).write_text(json.dumps(metrics))
    output = tmp_path / "seed-runs"
    result = subprocess.run(
        [
            sys.executable,
            "tools/aggregate_seeds.py",
            "--root",
            str(eval_root),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "fixture" / "summary.json").read_text())
    assert summary["leaderboard"]["retention_macro_f1"] == {
        "mean": pytest.approx(0.45),
        "std": pytest.approx(math.sqrt(0.005)),
        "n": 2,
    }
    assert [row["retention_macro_f1"] for row in summary["per_seed"]] == pytest.approx(
        [0.5, 0.4]
    )
    with open(output / "summary_all.csv", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["retention_macro_f1_mean"]) == pytest.approx(0.45)
    assert float(row["gen_gap_macro_f1_mean"]) == pytest.approx(0.38)


def _synthetic_labels():
    train = [0]
    for cls in range(1, 7):
        train.extend([cls] * cls)
    y_true = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6])
    y_pred = np.array([0, 1, 1, 1, 2, 4, 4, 5, 0])
    return np.array(train), y_true, y_pred


def test_per_class_frequency_bins_and_empty_match_hand_computation():
    train, y_true, y_pred = _synthetic_labels()
    rows, summary = analyze_split(y_true, y_pred, train, split="test")
    by_class = {row["class_id"]: row for row in rows}
    expected_f1 = {
        0: 0.5,  # TP=1, FP=1 (class 6), FN=1 (second empty)
        1: 0.8,  # TP=2, FP=1 (an empty), FN=0
        2: 1.0,
        3: 0.0,
        4: 2.0 / 3.0,  # TP=1, FP=1 (class 3), FN=0
        5: 1.0,
        6: 0.0,
    }
    for cls, expected in expected_f1.items():
        assert by_class[cls]["f1"] == pytest.approx(expected)
    # Counts 1..6 give terciles {1,2}, {3,4}, {5,6}.
    macros = {row["frequency_bin"]: row["macro_f1"] for row in summary}
    assert macros["tail"] == pytest.approx(0.9)
    assert macros["mid"] == pytest.approx(1.0 / 3.0)
    assert macros["head"] == pytest.approx(0.5)
    assert macros["empty"] == pytest.approx(0.5)


def test_per_class_cli_reads_saved_predictions_and_writes_csv_and_latex(tmp_path):
    train, ood_true, ood_pred = _synthetic_labels()
    id_true = np.arange(7)
    id_pred = np.arange(7)
    np.savez(tmp_path / "labels.npz", train=train, test=ood_true, id_test=id_true)
    run = tmp_path / "run"
    run.mkdir()
    (run / "iwildcam_test_predictions.csv").write_text(
        "\n".join(map(str, ood_pred.tolist())) + "\n"
    )
    (run / "iwildcam_id_test_predictions.csv").write_text(
        "prediction\n" + "\n".join(map(str, id_pred.tolist())) + "\n"
    )
    assert np.array_equal(read_predictions(run / "iwildcam_id_test_predictions.csv"), id_pred)
    prefix = str(tmp_path / "table")
    per_class_main(
        [
            "--run-dir",
            str(run),
            "--labels-npz",
            str(tmp_path / "labels.npz"),
            "--output-prefix",
            prefix,
        ]
    )
    with open(prefix + "_summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["split"], row["frequency_bin"]): float(row["macro_f1"]) for row in rows}
    assert lookup[("test", "tail")] == pytest.approx(0.9)
    assert lookup[("id_test", "head")] == pytest.approx(1.0)
    latex = (tmp_path / "table.tex").read_text()
    assert r"\toprule" in latex
    assert "Tail & 1.0000 & 0.9000" in latex
    assert "Empty & 1.0000 & 0.5000" in latex
    assert os.path.getsize(prefix + "_per_class.csv") > 0


def test_feature_effective_rank_isotropic_vs_rank_one():
    rng = np.random.default_rng(7)
    dim = 32
    isotropic = rng.normal(size=(2048, dim))
    isotropic_stats = compute_feature_statistics(isotropic, top_k=(1, dim))
    assert isotropic_stats["effective_rank"] == pytest.approx(dim, rel=0.05)
    assert isotropic_stats["top_k_variance"][0]["variance_fraction"] < 0.1
    assert isotropic_stats["top_k_variance"][-1]["variance_fraction"] == pytest.approx(1.0)

    rank_one = np.linspace(-2.0, 2.0, 2048)[:, None] * rng.normal(size=(1, dim))
    rank_one_stats = compute_feature_statistics(rank_one, top_k=(1, dim))
    assert rank_one_stats["effective_rank"] == pytest.approx(1.0, abs=1e-8)
    assert rank_one_stats["numerical_rank"] == 1
    assert rank_one_stats["top_k_variance"][0]["variance_fraction"] == pytest.approx(1.0)


def test_wilds_compatible_macro_f1_from_confusion():
    _, y_true, y_pred = _synthetic_labels()
    num_classes = 7
    confusion = torch.bincount(
        torch.tensor(y_true) * num_classes + torch.tensor(y_pred),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    expected = np.mean([0.5, 0.8, 1.0, 0.0, 2.0 / 3.0, 1.0, 0.0])
    assert macro_f1_from_confusion(confusion) == pytest.approx(expected)


def test_plot_val_curve_handles_resumed_headers_and_nan(tmp_path):
    log = tmp_path / "linear_probe_r0.csv"
    log.write_text(
        "epoch,val_loss,val_acc,val_f1,id_val_f1\n"
        "1,1.0,0.2,0.25,0.50\n"
        "epoch,val_loss,val_acc,val_f1,id_val_f1\n"
        "2,0.8,0.3,0.50,0.75\n"
        "3,0.7,0.4,nan,0.80\n"
    )
    rows = extract_val_curves(log)
    assert rows == [
        {"epoch": 1, "val_f1": 0.25, "id_val_f1": 0.5},
        {"epoch": 2, "val_f1": 0.5, "id_val_f1": 0.75},
        {"epoch": 3, "val_f1": None, "id_val_f1": 0.8},
    ]
    output = tmp_path / "curve.csv"
    write_val_curves(rows, output)
    assert output.read_text().splitlines()[-1] == "3,,0.8"
