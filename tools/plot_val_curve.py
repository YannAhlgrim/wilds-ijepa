#!/usr/bin/env python3
"""Extract per-epoch OOD-val and ID-val F1 into a pgfplots-ready CSV.

The baseline configs set ``validation.log_per_epoch_f1: true``, which appends
``val_f1`` (OOD ``val``) and ``id_val_f1`` to the normal training CSV. Repeated
headers from submitit resumes are tolerated; a later duplicate epoch supersedes
an earlier partial attempt.

Example:
  python tools/plot_val_curve.py --log RUN_DIR/linear_probe_r0.csv \
    --output RUN_DIR/val_f1_curve.csv

pgfplots can then use ``table[col sep=comma, x=epoch, y=val_f1]`` (or
``y=id_val_f1``). Missing/non-finite metric values are emitted as empty cells.
"""

import argparse
import csv
import math
import os


def extract_val_curves(log_path):
    rows_by_epoch = {}
    with open(log_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"Training log {log_path} has no epoch column")
        if "val_f1" not in reader.fieldnames and "ood_val_f1" not in reader.fieldnames:
            raise ValueError(
                f"Training log {log_path} has no val_f1 column; "
                "set validation.log_per_epoch_f1: true"
            )
        for row in reader:
            try:
                epoch = int(row["epoch"])
            except (TypeError, ValueError):
                # CSVLogger writes a fresh header after a job resume.
                continue
            ood = row.get("val_f1", row.get("ood_val_f1"))
            rows_by_epoch[epoch] = {
                "epoch": epoch,
                "val_f1": _finite_float_or_none(ood),
                "id_val_f1": _finite_float_or_none(row.get("id_val_f1")),
            }
    if not rows_by_epoch:
        raise ValueError(f"No metric rows found in {log_path}")
    return [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]


def _finite_float_or_none(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def write_val_curves(rows, output_path):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "val_f1", "id_val_f1"))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if key != "epoch" and row[key] is None else row[key]
                    for key in ("epoch", "val_f1", "id_val_f1")
                }
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="linear_probe_r0.csv training log")
    parser.add_argument("--output", required=True, help="output CSV for pgfplots")
    args = parser.parse_args(argv)
    rows = extract_val_curves(args.log)
    write_val_curves(rows, args.output)
    print(f"Wrote {len(rows)} epochs to {args.output}")
    return rows


if __name__ == "__main__":
    main()
