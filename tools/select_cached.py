#!/usr/bin/env python3
"""Select cached-probe configurations on OOD validation and format paper rows."""

import argparse
import csv
import json
import math
import os

import yaml

try:
    from tools.retention import summarize_retention
except ImportError:
    from retention import summarize_retention


CONFIG_FIELDS = ("lr", "weight_decay", "batch_size", "head_type", "representation_type")


def _identity(config):
    return (
        float(config["lr"]),
        float(config["weight_decay"]),
        int(config["batch_size"]),
        str(config["head_type"]),
        str(config["representation_type"]),
    )


def _mean_std(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, std


def _read_runs(runs_dir):
    records = []
    for dirpath, dirnames, filenames in os.walk(runs_dir):
        dirnames.sort()
        if "summary.json" not in filenames:
            continue
        path = os.path.join(dirpath, "summary.json")
        try:
            with open(path, "r") as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary.get("config"), dict) or "metrics" not in summary:
            continue
        summary["_path"] = path
        records.append(summary)
    if not records:
        raise ValueError(f"No cached-probe summary.json files found under {runs_dir}")
    return records


def _metric(record, split, key):
    return record.get("metrics", {}).get(split, {}).get(key)


def _config_row(records):
    first = records[0]
    identity = _identity(first["config"])
    val_mean, val_std = _mean_std([_metric(record, "val", "F1-macro_all") for record in records])
    test_mean, test_std = _mean_std([_metric(record, "test", "F1-macro_all") for record in records])
    return {
        "identity": identity,
        "config": {key: first["config"][key] for key in CONFIG_FIELDS},
        "selection_seeds": sorted(int(record["seed"]) for record in records),
        "val_macro_f1_mean": val_mean,
        "val_macro_f1_std": val_std,
        "test_macro_f1_mean": test_mean,
        "test_macro_f1_std": test_std,
        "selected_by_val": False,
        "selected_by_test": False,
    }


def _full_stats(records):
    if not records:
        return None
    records = sorted(records, key=lambda record: int(record["seed"]))
    result = {"num_seeds": len(records), "seeds": [int(record["seed"]) for record in records]}
    for split, prefix in (("id_test", "id_test"), ("test", "ood_test")):
        for metric, suffix in (("F1-macro_all", "macro_f1"), ("acc_avg", "avg_acc")):
            mean, std = _mean_std([_metric(record, split, metric) for record in records])
            result[f"{prefix}_{suffix}_mean"] = mean
            result[f"{prefix}_{suffix}_std"] = std
    retention = summarize_retention(
        [_metric(record, "id_test", "F1-macro_all") for record in records],
        [_metric(record, "test", "F1-macro_all") for record in records],
    )
    result["retention_mean"] = retention["mean"]
    result["retention_std"] = retention["std"]
    if result["id_test_macro_f1_mean"] is not None and result["ood_test_macro_f1_mean"] is not None:
        result["gap_macro_f1"] = result["id_test_macro_f1_mean"] - result["ood_test_macro_f1_mean"]
    else:
        result["gap_macro_f1"] = None
    return result


def select_cached(records):
    by_encoder = {}
    for record in records:
        by_encoder.setdefault(record["encoder_key"], []).append(record)
    output = {"models": {}, "configurations": []}
    for encoder, encoder_records in sorted(by_encoder.items()):
        grid_records = [
            record for record in encoder_records
            if record.get("stage") == "grid" and int(record["seed"]) == 0
        ]
        if not grid_records:
            grid_records = [
                record for record in encoder_records
                if float(record.get("label_fraction", 1.0)) == 1.0
                and record.get("stage") != "seeds"
                and int(record["seed"]) == 0
            ]
        grouped = {}
        for record in grid_records:
            grouped.setdefault(_identity(record["config"]), []).append(record)
        if not grouped:
            raise ValueError(f"No seed-0 selection runs found for {encoder}")
        rows = [_config_row(group) for group in grouped.values()]
        rows.sort(key=lambda row: (-row["val_macro_f1_mean"], row["identity"]))
        selected = rows[0]
        selected["selected_by_val"] = True
        test_selected = max(rows, key=lambda row: (row["test_macro_f1_mean"], row["identity"]))
        test_selected["selected_by_test"] = True
        full_records = [
            record for record in encoder_records
            if float(record.get("label_fraction", 1.0)) == 1.0
            and record.get("stage") in ("seeds", "full")
            and _identity(record["config"]) == selected["identity"]
        ]
        stats = _full_stats(full_records)
        label_efficiency = {}
        fractions = sorted({
            float(record.get("label_fraction", 1.0)) for record in encoder_records
            if float(record.get("label_fraction", 1.0)) < 1.0
            and _identity(record["config"]) == selected["identity"]
        })
        for fraction in fractions:
            subset = [
                record for record in encoder_records
                if float(record.get("label_fraction", 1.0)) == fraction
                and _identity(record["config"]) == selected["identity"]
            ]
            mean, std = _mean_std([_metric(record, "test", "F1-macro_all") for record in subset])
            label_efficiency[str(fraction)] = {"ood_macro_f1_mean": mean, "ood_macro_f1_std": std, "num_seeds": len(subset)}
        serializable_rows = []
        for row in rows:
            converted = dict(row)
            converted["identity"] = list(row["identity"])
            serializable_rows.append(converted)
            output["configurations"].append({"encoder_key": encoder, **converted})
        output["models"][encoder] = {
            "selected_config": selected["config"],
            "selection_criterion": "seed0_ood_val_macro_f1",
            "test_selected_config": test_selected["config"],
            "selection_changed_vs_test": selected["identity"] != test_selected["identity"],
            "full_seed_results": stats,
            "label_efficiency": label_efficiency,
            "selection_rows": serializable_rows,
        }
    return output


def _fmt(mean, std):
    if mean is None:
        return "NA"
    return f"{mean:.4f} ± {std:.4f}"


def _fmt_tex(mean, std):
    if mean is None:
        return "NA"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def _write_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cached_selection.json")
    csv_path = os.path.join(out_dir, "cached_selection.csv")
    tex_path = os.path.join(out_dir, "cached_selection.tex")
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fields = [
        "encoder_key", "config", "selection_seeds", "val_macro_f1_mean", "val_macro_f1_std",
        "test_macro_f1_mean", "test_macro_f1_std", "selected_by_val", "selected_by_test",
        "id_test_macro_f1_mean", "id_test_macro_f1_std", "id_test_avg_acc_mean", "id_test_avg_acc_std",
        "ood_test_macro_f1_mean", "ood_test_macro_f1_std", "ood_test_avg_acc_mean", "ood_test_avg_acc_std",
        "gap_macro_f1", "retention_mean", "retention_std",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["configurations"]:
            output = dict(row)
            output["config"] = json.dumps(output["config"], sort_keys=True)
            output["selection_seeds"] = " ".join(map(str, output["selection_seeds"]))
            if output["selected_by_val"]:
                output.update(result["models"][output["encoder_key"]]["full_seed_results"] or {})
            writer.writerow(output)

    latex_rows = []
    for encoder, model in result["models"].items():
        selected_yaml = os.path.join(out_dir, f"{encoder}_selected.yaml")
        with open(selected_yaml, "w") as handle:
            yaml.safe_dump(
                {
                    "grid": {key: [model["selected_config"][key]] for key in CONFIG_FIELDS},
                    "defaults": {"max_epochs": 100, "patience": 10, "min_delta": 1e-4, "min_epochs": 0},
                },
                handle,
                sort_keys=False,
            )
        stats = model["full_seed_results"]
        if stats:
            escaped_encoder = encoder.replace("_", "\\_")
            latex_rows.append(
                f"{escaped_encoder} & "
                f"{_fmt_tex(stats['id_test_macro_f1_mean'], stats['id_test_macro_f1_std'])} & "
                f"{_fmt_tex(stats['id_test_avg_acc_mean'], stats['id_test_avg_acc_std'])} & "
                f"{_fmt_tex(stats['ood_test_macro_f1_mean'], stats['ood_test_macro_f1_std'])} & "
                f"{_fmt_tex(stats['ood_test_avg_acc_mean'], stats['ood_test_avg_acc_std'])} & "
                f"{stats['gap_macro_f1']:.4f} & {_fmt_tex(stats['retention_mean'], stats['retention_std'])} \\\\"
            )
        print(f"\n{encoder}: selected {model['selected_config']}")
        if stats:
            print(
                f"  ID F1 {_fmt(stats['id_test_macro_f1_mean'], stats['id_test_macro_f1_std'])}; "
                f"OOD F1 {_fmt(stats['ood_test_macro_f1_mean'], stats['ood_test_macro_f1_std'])}; "
                f"R {_fmt(stats['retention_mean'], stats['retention_std'])}"
            )
        print(
            f"  DIAGNOSTIC ONLY: OOD test would select {model['test_selected_config']}; "
            f"changed={model['selection_changed_vs_test']}"
        )
    with open(tex_path, "w") as handle:
        handle.write("% Encoder & ID F1 & ID Acc & OOD F1 & OOD Acc & Gap & Retention\n")
        handle.write("\n".join(latex_rows) + ("\n" if latex_rows else ""))
    print(f"\nWrote {json_path}\nWrote {csv_path}\nWrote {tex_path}")
    return json_path, csv_path, tex_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--out", help="default: RUNS/cached-selection")
    args = parser.parse_args(argv)
    result = select_cached(_read_runs(args.runs))
    _write_outputs(result, args.out or os.path.join(args.runs, "cached-selection"))
    return result


if __name__ == "__main__":
    main()
