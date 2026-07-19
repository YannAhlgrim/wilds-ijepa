#!/usr/bin/env python3
"""
Aggregate label-efficiency supervised runs into a paper-style Table 4 CSV.

For each model, this collects the OOD F1-Macro across seeds and label
fractions (1%, 10%, 50%, 100%) and outputs mean +/- std in a wide CSV table.

It expects the run_info written by train_supervised.py to contain
`seed` and `data.label_fraction` (or the eval folder name to contain
`-lf<FRACTION>-seed<SEED>`). The 100% runs are the full-data runs already
configured for the paper.

Outputs:
  experiment_logs/label-efficiency/summary.csv
  experiment_logs/label-efficiency/<model>/summary.json

Usage:
  python3 tools/aggregate_label_efficiency.py --root experiment_logs/eval-wilds
  python3 tools/aggregate_label_efficiency.py --root experiment_logs/eval-wilds --metric F1-macro_all
"""
import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

import yaml

OOD_METRICS_FILE = "iwildcam_test_metrics.json"

# Match trailing -lf0.01-seed0 or -seed0.
_LF_SEED_SUFFIX_RE = re.compile(r"-lf(?P<frac>0\.\d+|[1-9]\d*\.?\d*)-seed(?P<seed>\d+)$")
_SEED_SUFFIX_RE = re.compile(r"-seed(?P<seed>\d+)$")


FRACTIONS = [0.01, 0.10, 0.50, 1.00]


def _find_metric(metrics, key):
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


def _extract_run_info(metrics_obj):
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

    if not metrics_obj:
        return {}
    return _search(metrics_obj) or {}


def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _load_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except (OSError, yaml.YAMLError):
        return None


def _parse_fraction_and_seed(run_name, run_info, run_dir):
    """Return (fraction, seed) from run_info, folder name, or params.yaml."""
    fraction = run_info.get("label_fraction")
    seed = run_info.get("seed")

    # Try folder name first for both values.
    m = _LF_SEED_SUFFIX_RE.search(run_name)
    if m:
        if fraction is None:
            try:
                fraction = float(m.group("frac"))
            except ValueError:
                pass
        if seed is None:
            try:
                seed = int(m.group("seed"))
            except ValueError:
                pass
    else:
        m = _SEED_SUFFIX_RE.search(run_name)
        if m and seed is None:
            try:
                seed = int(m.group("seed"))
            except ValueError:
                pass

    # Fallback to params.yaml.
    if fraction is None or seed is None:
        params = _load_yaml(os.path.join(run_dir, "params.yaml"))
        if params:
            if fraction is None:
                fraction = _find_metric(params, "label_fraction")
            if seed is None:
                seed = _find_metric(params, "seed")

    return fraction, seed


def _model_key(run_name):
    """Strip the label-fraction and seed suffix to obtain a model group key."""
    key = _LF_SEED_SUFFIX_RE.sub("", run_name)
    key = _SEED_SUFFIX_RE.sub("", key)
    return key


def _mean_std(values):
    vals = [v for v in values if v is not None and not _is_nan(v)]
    if not vals:
        return None, None, 0
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    else:
        std = 0.0
    return mean, std, n


def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _fmt(mean, std):
    if mean is None:
        return ""
    if std is None or std == 0.0:
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
        default="experiment_logs/label-efficiency",
        help="output folder for summaries (default: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        default="F1-macro_all",
        help="OOD metric to aggregate (default: %(default)s)",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="only report fractions with at least this many seeds (default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Root folder not found: {args.root}")
        return

    # model_key -> {fraction: [records]}
    groups = defaultdict(lambda: defaultdict(list))
    skipped = []

    for entry in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, entry)
        if not os.path.isdir(run_dir):
            continue

        ood_metrics = _load_json(os.path.join(run_dir, OOD_METRICS_FILE))
        if ood_metrics is None:
            continue

        run_info = _extract_run_info(ood_metrics) or {}
        fraction, seed = _parse_fraction_and_seed(entry, run_info, run_dir)
        value = _find_metric(ood_metrics, args.metric)

        if fraction is None:
            skipped.append((entry, "no label_fraction"))
            continue
        if value is None:
            skipped.append((entry, f"metric {args.metric} missing"))
            continue

        key = _model_key(entry)
        groups[key][fraction].append(
            {
                "run_name": entry,
                "seed": seed,
                "value": float(value),
                "run_info": run_info,
            }
        )

    if not groups:
        print(f"No usable metrics found under {args.root}")
        return

    os.makedirs(args.out, exist_ok=True)

    # CSV fieldnames.
    fieldnames = ["model"]
    for frac in FRACTIONS:
        frac_label = f"frac{frac:.2f}"
        fieldnames.extend(
            [
                f"{frac_label}_mean",
                f"{frac_label}_std",
                f"{frac_label}_n",
            ]
        )
    fieldnames.append("seeds")

    csv_rows = []

    for model_key in sorted(groups.keys()):
        fractions = groups[model_key]
        row = {"model": model_key}
        per_frac = {}
        all_seeds = set()

        for frac in FRACTIONS:
            records = fractions.get(frac, [])
            values = [r["value"] for r in records]
            mean, std, n = _mean_std(values)
            frac_label = f"frac{frac:.2f}"
            row[f"{frac_label}_mean"] = mean
            row[f"{frac_label}_std"] = std
            row[f"{frac_label}_n"] = n
            per_frac[frac] = {
                "mean": mean,
                "std": std,
                "n": n,
                "seeds": [r["seed"] for r in records],
                "values": values,
                "run_names": [r["run_name"] for r in records],
            }
            for r in records:
                if r["seed"] is not None:
                    all_seeds.add(r["seed"])

        row["seeds"] = " ".join(str(s) for s in sorted(all_seeds))
        csv_rows.append(row)

        # Write per-model JSON summary.
        model_dir = os.path.join(args.out, model_key)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "model": model_key,
                    "metric": args.metric,
                    "fractions": {f"{k:.2f}": v for k, v in per_frac.items()},
                },
                f,
                indent=2,
                sort_keys=True,
            )

    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    # Terminal table.
    print(f"\nAggregated {len(csv_rows)} model(s). Metric: {args.metric}\n")
    header = f"{'model':<50}"
    for frac in FRACTIONS:
        header += f"  {f'{int(frac*100)}%':>18}"
    print(header)
    print("-" * len(header))
    for row in csv_rows:
        line = f"{row['model'][:50]:<50}"
        for frac in FRACTIONS:
            frac_label = f"frac{frac:.2f}"
            mean = row[f"{frac_label}_mean"]
            std = row[f"{frac_label}_std"]
            line += f"  {_fmt(mean, std):>18}"
        print(line)

    if skipped:
        print(f"\nSkipped {len(skipped)} run(s):")
        for run_name, reason in skipped:
            print(f"  {run_name}: {reason}")

    print(f"\nWrote per-model summaries to: {args.out}/<model>/summary.json")
    print(f"Wrote combined CSV to:        {csv_path}")


if __name__ == "__main__":
    main()
