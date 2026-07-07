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
  python3 tools/aggregate_seeds.py --root experiment_logs/eval-wilds --min-seeds 5
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


def _extract_run_info(*metrics_objs):
    """Find a run_info dict inside any metrics object (dict- or list-form).

    Eval-only runs write metrics as a list ([{metrics}, "summary"]) with no
    run_info; training runs write a dict with a top-level run_info. Search both.
    """

    def _search(obj):
        if isinstance(obj, dict):
            ri = obj.get("run_info")
            if isinstance(ri, dict):
                return ri
            for v in obj.values():
                found = _search(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _search(item)
                if found is not None:
                    return found
        return None

    for m in metrics_objs:
        found = _search(m)
        if found is not None:
            return found
    return {}


def _seed_from_name(run_name):
    """Parse a trailing -seedN from a run folder name; None if absent."""
    m = re.search(r"-seed(\d+)$", run_name)
    return int(m.group(1)) if m else None


def _seed_from_params(run_dir):
    """Read `seed:` from params-eval.yaml without requiring PyYAML."""
    path = os.path.join(run_dir, "params-eval.yaml")
    try:
        with open(path, "r") as f:
            for line in f:
                m = re.match(r"\s*seed\s*:\s*(\d+)\s*$", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def _resolve_seed(run_info, run_name, run_dir):
    """Seed detection chain: run_info -> folder name -> params-eval.yaml."""
    seed = run_info.get("seed")
    if seed is not None:
        return seed
    seed = _seed_from_name(run_name)
    if seed is not None:
        return seed
    return _seed_from_params(run_dir)


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
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="only report models with at least this many seeds (default: %(default)s)",
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

        run_info = _extract_run_info(ood_metrics, id_metrics)

        group = _strip_seed(entry)
        groups.setdefault(group, []).append(
            {
                "run_name": entry,
                "seed": _resolve_seed(run_info, entry, run_dir),
                "id_metrics": id_metrics,
                "ood_metrics": ood_metrics,
                "run_info": run_info,
            }
        )

    if not groups:
        print(f"No metrics found under {args.root}")
        return

    if args.min_seeds > 1:
        dropped = {
            name: len(recs)
            for name, recs in groups.items()
            if len(recs) < args.min_seeds
        }
        groups = {
            name: recs for name, recs in groups.items() if len(recs) >= args.min_seeds
        }
        if dropped:
            print(
                f"Skipping {len(dropped)} model(s) with fewer than "
                f"{args.min_seeds} seed(s):"
            )
            for name in sorted(dropped):
                print(f"  {name} ({dropped[name]} seed(s))")
        if not groups:
            print(f"No models have >= {args.min_seeds} seeds under {args.root}")
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
        "peak_host_ram_gb_mean",
        "peak_host_ram_gb_std",
        "peak_gpu_alloc_gb_mean",
        "peak_gpu_alloc_gb_std",
        "peak_gpu_reserved_gb_mean",
        "peak_gpu_reserved_gb_std",
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
                    "peak_host_ram_gb": r["run_info"].get("peak_host_ram_gb"),
                    "peak_gpu_alloc_gb": r["run_info"].get("peak_gpu_alloc_gb"),
                    "peak_gpu_reserved_gb": r["run_info"].get("peak_gpu_reserved_gb"),
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
        host_ram_vals = [r["run_info"].get("peak_host_ram_gb") for r in records]
        gpu_alloc_vals = [r["run_info"].get("peak_gpu_alloc_gb") for r in records]
        gpu_reserved_vals = [r["run_info"].get("peak_gpu_reserved_gb") for r in records]
        time_mean, time_std, _ = _mean_std(time_vals)
        epoch_mean, epoch_std, _ = _mean_std(epoch_vals)
        host_ram_mean, host_ram_std, host_ram_n = _mean_std(host_ram_vals)
        gpu_alloc_mean, gpu_alloc_std, _ = _mean_std(gpu_alloc_vals)
        gpu_reserved_mean, gpu_reserved_std, _ = _mean_std(gpu_reserved_vals)

        if host_ram_n == 0:
            # No usable peak_host_ram_gb in any seed's run_info. This usually means
            # the metrics JSONs were (re)generated by a standalone eval run that did
            # not fold in train_supervised.py's run_info, or run_info is absent.
            missing_run_info = sum(1 for r in records if not r["run_info"])
            print(
                f"[warn] {group_name}: RAM was not recorded for these eval runs "
                f"(no peak_host_ram_gb across {len(records)} seed(s); "
                f"{missing_run_info} missing run_info entirely). RAM column stays blank."
            )

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
            "resources": {
                "peak_host_ram_gb": {"mean": host_ram_mean, "std": host_ram_std},
                "peak_gpu_alloc_gb": {"mean": gpu_alloc_mean, "std": gpu_alloc_std},
                "peak_gpu_reserved_gb": {
                    "mean": gpu_reserved_mean,
                    "std": gpu_reserved_std,
                },
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
                "peak_host_ram_gb_mean": host_ram_mean,
                "peak_host_ram_gb_std": host_ram_std,
                "peak_gpu_alloc_gb_mean": gpu_alloc_mean,
                "peak_gpu_alloc_gb_std": gpu_alloc_std,
                "peak_gpu_reserved_gb_mean": gpu_reserved_mean,
                "peak_gpu_reserved_gb_std": gpu_reserved_std,
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
        f"{'OOD MacroF1':>18}  {'OOD AvgAcc':>18}  {'gap':>8}  "
        f"{'peakRAM_GB':>14}  {'peakVRAM_GB':>14}"
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
            f"{gap_str:>8}  "
            f"{_fmt(row['peak_host_ram_gb_mean'], row['peak_host_ram_gb_std']):>14}  "
            f"{_fmt(row['peak_gpu_alloc_gb_mean'], row['peak_gpu_alloc_gb_std']):>14}"
        )

    print(f"\nWrote per-model summaries to: {args.out}/<model>/summary.json")
    print(f"Wrote combined CSV to:        {csv_path}")


if __name__ == "__main__":
    main()
