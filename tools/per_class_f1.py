#!/usr/bin/env python3
"""Per-class and frequency-binned F1 for saved iWildCam predictions.

By default labels are read through the WILDS package from ``--root-path`` (no
images are loaded). ``--labels-npz`` is an offline alternative for tests or
previously exported labels; it must contain ``train``, ``test`` and ``id_test``
integer arrays.

Outputs are ``PREFIX_per_class.csv``, ``PREFIX_summary.csv`` and ``PREFIX.tex``.
Frequency bins are deterministic count-ordered terciles over non-empty classes:
tail (lowest Source counts), mid, then head (highest). Bin macro-F1 follows the
WILDS convention of averaging classes present in that split's ground truth.
"""

import argparse
import csv
import os

import numpy as np


BIN_ORDER = ("tail", "mid", "head")


def per_class_counts(y_true, y_pred, num_classes=None):
    """Return TP/FP/FN/support/predicted counts for every class."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction/label length mismatch: {len(y_pred)} vs {len(y_true)}")
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Class labels and predictions must be non-negative integers")
    if num_classes is None:
        num_classes = int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1)
    indices = y_true * int(num_classes) + y_pred
    confusion = np.bincount(indices, minlength=int(num_classes) ** 2).reshape(
        int(num_classes), int(num_classes)
    )
    tp = np.diag(confusion).astype(np.int64)
    support = confusion.sum(axis=1).astype(np.int64)
    predicted = confusion.sum(axis=0).astype(np.int64)
    return {
        "tp": tp,
        "fp": predicted - tp,
        "fn": support - tp,
        "support": support,
        "predicted": predicted,
    }


def f1_from_counts(tp, fp, fn):
    """Binary one-vs-rest F1; a class with no true or predicted examples is 0."""
    tp = np.asarray(tp, dtype=np.float64)
    denominator = 2.0 * tp + np.asarray(fp, dtype=np.float64) + np.asarray(fn, dtype=np.float64)
    return np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def source_frequency_bins(train_labels, num_classes, empty_class=0):
    """Assign non-empty classes to equal-sized, count-sorted Source terciles."""
    counts = np.bincount(
        np.asarray(train_labels, dtype=np.int64).reshape(-1), minlength=int(num_classes)
    )[: int(num_classes)]
    species = np.array(
        sorted((cls for cls in range(num_classes) if cls != empty_class)),
        dtype=np.int64,
    )
    species = species[np.lexsort((species, counts[species]))]
    assignments = {int(empty_class): "empty"}
    for bin_name, members in zip(BIN_ORDER, np.array_split(species, 3)):
        for cls in members:
            assignments[int(cls)] = bin_name
    return counts.astype(np.int64), assignments


def analyze_split(y_true, y_pred, train_labels, empty_class=0, split="test", num_classes=None):
    """Compute class rows and tail/mid/head/empty summaries for one split."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    if num_classes is None:
        num_classes = int(
            max(y_true.max(initial=-1), y_pred.max(initial=-1), train_labels.max(initial=-1)) + 1
        )
    counts = per_class_counts(y_true, y_pred, num_classes=num_classes)
    f1 = f1_from_counts(counts["tp"], counts["fp"], counts["fn"])
    train_counts, assignments = source_frequency_bins(
        train_labels, num_classes=num_classes, empty_class=empty_class
    )

    class_rows = []
    for cls in range(num_classes):
        class_rows.append(
            {
                "split": split,
                "class_id": cls,
                "frequency_bin": assignments.get(cls, "unknown"),
                "train_count": int(train_counts[cls]),
                "support": int(counts["support"][cls]),
                "predicted_count": int(counts["predicted"][cls]),
                "tp": int(counts["tp"][cls]),
                "fp": int(counts["fp"][cls]),
                "fn": int(counts["fn"][cls]),
                "f1": float(f1[cls]),
            }
        )

    summary_rows = []
    for bin_name in BIN_ORDER:
        members = [cls for cls in range(num_classes) if assignments.get(cls) == bin_name]
        active = [cls for cls in members if counts["support"][cls] > 0]
        macro = float(np.mean(f1[active])) if active else None
        summary_rows.append(
            {
                "split": split,
                "frequency_bin": bin_name,
                "n_classes": len(active),
                "n_bin_classes": len(members),
                "macro_f1": macro,
            }
        )
    empty_f1 = float(f1[empty_class]) if 0 <= empty_class < num_classes else None
    summary_rows.append(
        {
            "split": split,
            "frequency_bin": "empty",
            "n_classes": int(0 <= empty_class < num_classes and counts["support"][empty_class] > 0),
            "n_bin_classes": 1,
            "macro_f1": empty_f1,
        }
    )
    return class_rows, summary_rows


def read_predictions(path):
    """Read the one-integer-per-line files emitted by eval_wilds.py."""
    values = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            if line_number == 1 and text.lower() in ("prediction", "y_pred", "pred"):
                continue
            values.append(int(text))
    return np.asarray(values, dtype=np.int64)


def _load_labels(root_path, download=False):
    from wilds import get_dataset

    dataset = get_dataset(dataset="iwildcam", download=download, root_dir=root_path)
    return {
        split: np.asarray(dataset.get_subset(split).y_array, dtype=np.int64)
        for split in ("train", "test", "id_test")
    }


def _write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_table(summary_rows):
    """Return a compact booktabs table with ID and OOD columns."""
    lookup = {(row["split"], row["frequency_bin"]): row["macro_f1"] for row in summary_rows}
    labels = {"tail": "Tail", "mid": "Mid", "head": "Head", "empty": "Empty"}

    def format_value(split, bin_name):
        value = lookup.get((split, bin_name))
        return "--" if value is None else f"{value:.4f}"

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Frequency bin & ID test macro-F1 & OOD test macro-F1 \\",
        r"\midrule",
    ]
    for bin_name in (*BIN_ORDER, "empty"):
        lines.append(
            f"{labels[bin_name]} & {format_value('id_test', bin_name)} & "
            f"{format_value('test', bin_name)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_outputs(output_prefix, class_rows, summary_rows):
    output_dir = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(output_dir, exist_ok=True)
    class_path = output_prefix + "_per_class.csv"
    summary_path = output_prefix + "_summary.csv"
    latex_path = output_prefix + ".tex"
    _write_csv(class_path, class_rows)
    _write_csv(summary_path, summary_rows)
    latex = latex_table(summary_rows)
    with open(latex_path, "w") as handle:
        handle.write(latex)
    return class_path, summary_path, latex_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="one eval_wilds.py run directory")
    parser.add_argument("--root-path", default="./wilds_data", help="WILDS root directory")
    parser.add_argument("--download", action="store_true", help="allow WILDS metadata download")
    parser.add_argument("--labels-npz", help="offline train/test/id_test labels; bypasses WILDS")
    parser.add_argument("--empty-class", type=int, default=0)
    parser.add_argument("--output-prefix", help="default: RUN_DIR/per_class_f1")
    args = parser.parse_args(argv)

    if args.labels_npz:
        with np.load(args.labels_npz, allow_pickle=False) as data:
            labels = {split: data[split].astype(np.int64) for split in ("train", "test", "id_test")}
    else:
        labels = _load_labels(args.root_path, download=args.download)

    all_class_rows = []
    all_summary_rows = []
    for split, filename in (
        ("test", "iwildcam_test_predictions.csv"),
        ("id_test", "iwildcam_id_test_predictions.csv"),
    ):
        predictions = read_predictions(os.path.join(args.run_dir, filename))
        class_rows, summary_rows = analyze_split(
            labels[split],
            predictions,
            labels["train"],
            empty_class=args.empty_class,
            split=split,
        )
        all_class_rows.extend(class_rows)
        all_summary_rows.extend(summary_rows)

    prefix = args.output_prefix or os.path.join(args.run_dir, "per_class_f1")
    paths = write_outputs(prefix, all_class_rows, all_summary_rows)
    for path in paths:
        print(f"Wrote {path}")
    print(latex_table(all_summary_rows))
    return all_class_rows, all_summary_rows


if __name__ == "__main__":
    main()
