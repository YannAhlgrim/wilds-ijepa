#!/usr/bin/env python3
"""
Aggregate multi-seed WILDS-iWildCam supervised runs into mean +/- std.

For each model configuration (grouped across seeds), this reads the per-seed
evaluation metrics for both splits:

  - ID  (in-distribution):  iwildcam_id_test_metrics.json   (split "id_test")
  - OOD (out-of-distribution): iwildcam_test_metrics.json    (split "test")

and computes, across seeds, the mean and std of every WILDS metric plus the
training time and epochs recorded during training. It also reports the
generalization gap (ID - OOD) on the headline metric.

Leaderboard columns reported:
  Test ID Macro F1 | Test ID Avg Acc | Test OOD Macro F1 | Test OOD Avg Acc

Outputs:
  experiment_logs/seed-runs/<model>/summary.json   (per-seed rows + mean/std)
  experiment_logs/seed-runs/summary_all.csv        (one row per model)

Usage:
  python3 tools/aggregate_seeds.py --root experiment_logs/eval-wilds
  python3 tools/aggregate_seeds.py --primary F1-macro_all --acc acc_avg
"""
import argparse
import csv
import json
import math
import os
import re


ID_METRICS_FILE = "iwildcam_id_test_metrics.json"
OOD_METRICS_FILE = "iwildcam_test_metrics.json"

# Trailing "-seedN" (and any leftover separators) so runs of the same config
# collapse into one group.
_SEED_SUFFIX_RE = re.compile(r"-seed\d+$")


def _find_metric(metrics, key):
    """Recursively search a nested dict/list for `key`."""
    if isinstance(metrics, dict):
        if key in metrics:
            return metrics[key]
        for value in metrics.values():
            found = _find_metric(value, key)
            if found is not None:
                return found
    elif isinstance(metrics, list):
        for item in metrics:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _strip_seed(run_name):
    return _SEED_SUFFIX_RE.sub("", run_name)


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _mean_std(values):
    vals = [v for v in values if v is not None and not _is_nan(v)]
    if not vals:
        return None, None, 0
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)  # sample std
        std = math.sqrt(var)
    else:
        std = 0.0
    return mean, std, n


def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _collect_metric_keys(metrics_obj):
    """All scalar metric keys in a WILDS metrics dict (excludes our extras)."""
    keys = set()
    if isinstance(metrics_obj, dict):
        for k, v in metrics_obj.items():
            if k in ("run_info", "split"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    return keys


def _fmt(mean, std):
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--root",
        default="experiment_logs/eval-wilds",
        help="root folder holding per-run eval subfolders (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="experiment_logs/seed-runs",
        help="output folder for aggregated summaries (default: %(default)s)",
    )
    parser.add_argument(
        "--primary",
        default="F1-macro_all",
        help="headline metric key (default: %(default)s)",
    )
    parser.add_argument(
        "--acc",
        default="acc_avg",
        help="average-accuracy metric key (default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Root folder not found: {args.root}")
        return

    # group_name -> list of per-seed records
    groups = {}
    for entry in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, entry)
        if not os.path.isdir(run_dir):
            continue
        id_metrics = _load_json(os.path.join(run_dir, ID_METRICS_FILE))
        ood_metrics = _load_json(os.path.join(run_dir, OOD_METRICS_FILE))
        if id_metrics is None and ood_metrics is None:
            continue

        run_info = {}
        for m in (ood_metrics, id_metrics):
            if isinstance(m, dict) and isinstance(m.get("run_info"), dict):
                run_info = m["run_info"]
                break

        group = _strip_seed(entry)
        groups.setdefault(group, []).append(
            {
                "run_name": entry,
                "seed": run_info.get("seed"),
                "id_metrics": id_metrics,
                "ood_metrics": ood_metrics,
                "run_info": run_info,
            }
        )

    if not groups:
        print(f"No metrics found under {args.root}")
        return

    os.makedirs(args.out, exist_ok=True)

    csv_rows = []
    csv_fields = [
        "model",
        "num_seeds",
        "seeds",
        "id_macro_f1_mean",
        "id_macro_f1_std",
        "id_avg_acc_mean",
        "id_avg_acc_std",
        "ood_macro_f1_mean",
        "ood_macro_f1_std",
        "ood_avg_acc_mean",
        "ood_avg_acc_std",
        "gen_gap_macro_f1_mean",  # ID - OOD on primary metric
        "train_time_seconds_mean",
        "train_time_seconds_std",
        "epochs_run_mean",
        "epochs_run_std",
    ]

    for group_name in sorted(groups.keys()):
        records = sorted(groups[group_name], key=lambda r: (r["seed"] is None, r["seed"]))
        seeds = [r["seed"] for r in records]

        # Determine the union of metric keys present on either split.
        all_metric_keys = set()
        for r in records:
            all_metric_keys |= _collect_metric_keys(r["id_metrics"])
            all_metric_keys |= _collect_metric_keys(r["ood_metrics"])

        def metric_values(split_key, mkey):
            return [_find_metric(r[split_key], mkey) for r in records]

        # Per-seed rows for the JSON summary.
        per_seed = []
        for r in records:
            per_seed.append(
                {
                    "seed": r["seed"],
                    "run_name": r["run_name"],
                    "id": {
                        "macro_f1": _find_metric(r["id_metrics"], args.primary),
                        "avg_acc": _find_metric(r["id_metrics"], args.acc),
                    },
                    "ood": {
                        "macro_f1": _find_metric(r["ood_metrics"], args.primary),
                        "avg_acc": _find_metric(r["ood_metrics"], args.acc),
                    },
                    "train_time_seconds": r["run_info"].get("train_time_seconds"),
                    "train_time_hms": r["run_info"].get("train_time_hms"),
                    "epochs_run": r["run_info"].get("epochs_run"),
                    "configured_epochs": r["run_info"].get("configured_epochs"),
                    "best_epoch": r["run_info"].get("best_epoch"),
                    "early_stopped": r["run_info"].get("early_stopped"),
                }
            )

        # Aggregate every metric for both splits.
        def agg_all(split_key):
            out = {}
            for mkey in sorted(all_metric_keys):
                mean, std, n = _mean_std(metric_values(split_key, mkey))
                if n > 0:
                    out[mkey] = {"mean": mean, "std": std, "n": n}
            return out

        id_agg = agg_all("id_metrics")
        ood_agg = agg_all("ood_metrics")

        time_vals = [r["run_info"].get("train_time_seconds") for r in records]
        epoch_vals = [r["run_info"].get("epochs_run") for r in records]
        time_mean, time_std, _ = _mean_std(time_vals)
        epoch_mean, epoch_std, _ = _mean_std(epoch_vals)

        # Headline (leaderboard) numbers.
        id_f1_mean, id_f1_std, _ = _mean_std(metric_values("id_metrics", args.primary))
        id_acc_mean, id_acc_std, _ = _mean_std(metric_values("id_metrics", args.acc))
        ood_f1_mean, ood_f1_std, _ = _mean_std(metric_values("ood_metrics", args.primary))
        ood_acc_mean, ood_acc_std, _ = _mean_std(metric_values("ood_metrics", args.acc))

        gen_gap = None
        if id_f1_mean is not None and ood_f1_mean is not None:
            gen_gap = id_f1_mean - ood_f1_mean

        summary = {
            "model": group_name,
            "primary_metric": args.primary,
            "acc_metric": args.acc,
            "num_seeds": len(records),
            "seeds": seeds,
            "leaderboard": {
                "test_id_macro_f1": {"mean": id_f1_mean, "std": id_f1_std},
                "test_id_avg_acc": {"mean": id_acc_mean, "std": id_acc_std},
                "test_ood_macro_f1": {"mean": ood_f1_mean, "std": ood_f1_std},
                "test_ood_avg_acc": {"mean": ood_acc_mean, "std": ood_acc_std},
                "generalization_gap_macro_f1": gen_gap,
            },
            "training": {
                "train_time_seconds": {"mean": time_mean, "std": time_std},
                "epochs_run": {"mean": epoch_mean, "std": epoch_std},
            },
            "id_metrics_aggregated": id_agg,
            "ood_metrics_aggregated": ood_agg,
            "per_seed": per_seed,
        }

        model_out_dir = os.path.join(args.out, group_name)
        os.makedirs(model_out_dir, exist_ok=True)
        with open(os.path.join(model_out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        csv_rows.append(
            {
                "model": group_name,
                "num_seeds": len(records),
                "seeds": " ".join(str(s) for s in seeds),
                "id_macro_f1_mean": id_f1_mean,
                "id_macro_f1_std": id_f1_std,
                "id_avg_acc_mean": id_acc_mean,
                "id_avg_acc_std": id_acc_std,
                "ood_macro_f1_mean": ood_f1_mean,
                "ood_macro_f1_std": ood_f1_std,
                "ood_avg_acc_mean": ood_acc_mean,
                "ood_avg_acc_std": ood_acc_std,
                "gen_gap_macro_f1_mean": gen_gap,
                "train_time_seconds_mean": time_mean,
                "train_time_seconds_std": time_std,
                "epochs_run_mean": epoch_mean,
                "epochs_run_std": epoch_std,
            }
        )

    csv_path = os.path.join(args.out, "summary_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # Terminal table.
    print(f"\nAggregated {len(csv_rows)} model(s). Primary metric: {args.primary}\n")
    header = (
        f"{'model':<40} {'seeds':>5}  "
        f"{'ID MacroF1':>18}  {'ID AvgAcc':>18}  "
        f"{'OOD MacroF1':>18}  {'OOD AvgAcc':>18}  {'gap':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in csv_rows:
        gap = row["gen_gap_macro_f1_mean"]
        gap_str = "" if gap is None else f"{gap:.4f}"
        model_str = row["model"][:40]
        print(
            f"{model_str:<40} {row['num_seeds']:>5}  "
            f"{_fmt(row['id_macro_f1_mean'], row['id_macro_f1_std']):>18}  "
            f"{_fmt(row['id_avg_acc_mean'], row['id_avg_acc_std']):>18}  "
            f"{_fmt(row['ood_macro_f1_mean'], row['ood_macro_f1_std']):>18}  "
            f"{_fmt(row['ood_avg_acc_mean'], row['ood_avg_acc_std']):>18}  "
            f"{gap_str:>8}"
        )

    print(f"\nWrote per-model summaries to: {args.out}/<model>/summary.json")
    print(f"Wrote combined CSV to:        {csv_path}")


if __name__ == "__main__":
    main()
