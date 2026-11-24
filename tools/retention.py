#!/usr/bin/env python3
"""Relative OOD retention R = F1_OOD / F1_ID, aggregated per seed.

``aggregate_seeds.py`` uses these helpers and includes retention in its normal
summary files. This module also provides a small CLI for saved per-seed values:

  python tools/retention.py --id-f1 0.8 0.6 --ood-f1 0.4 0.24

Zero or missing ID F1 makes the ratio undefined; that seed is retained as null
and excluded from the aggregate rather than silently producing infinity.
"""

import argparse
import json
import math


def relative_retention(id_f1, ood_f1):
    """Return one seed's relative retention, or None when it is undefined."""
    if id_f1 is None or ood_f1 is None:
        return None
    id_f1 = float(id_f1)
    ood_f1 = float(ood_f1)
    if not math.isfinite(id_f1) or not math.isfinite(ood_f1) or id_f1 == 0.0:
        return None
    return ood_f1 / id_f1


def summarize_retention(id_f1_values, ood_f1_values):
    """Compute mean/std of ratios (not the ratio of aggregate F1 means)."""
    if len(id_f1_values) != len(ood_f1_values):
        raise ValueError("ID and OOD F1 sequences must have the same number of seeds")
    per_seed = [
        relative_retention(id_f1, ood_f1)
        for id_f1, ood_f1 in zip(id_f1_values, ood_f1_values)
    ]
    values = [value for value in per_seed if value is not None]
    n = len(values)
    if not values:
        mean = None
        std = None
    else:
        mean = sum(values) / n
        std = (
            math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
            if n > 1
            else 0.0
        )
    return {"per_seed": per_seed, "mean": mean, "std": std, "n": n}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-f1", type=float, nargs="+", required=True)
    parser.add_argument("--ood-f1", type=float, nargs="+", required=True)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args(argv)
    result = summarize_retention(args.id_f1, args.ood_f1)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return result


if __name__ == "__main__":
    main()
