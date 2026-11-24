#!/usr/bin/env python3
"""Select existing probe hyperparameters using OOD validation, never OOD test.

Expected run directories are immediate children of ``--root`` (normally
``experiment_logs/eval-wilds``). Each may contain iwildcam_{val,id_test,test}
metrics and a retained ``*_r0.csv``. For old I-JEPA runs whose training directory
was not retained, pass its parent as ``--train-root``; ``<train-root>/<run-name>``
and matching nested run directories are searched. Existing completed runs are
sufficient: no model or WILDS data is loaded.

Validation criterion priority is (a) mean ``F1-macro_all`` from
``iwildcam_val_metrics.json`` when available for every seed, then (b) each seed's
minimum ``val_loss`` from its training CSV, using OOD ``val_acc`` as a tiebreak.
F1 and loss candidates are never ranked against each other. ``--out`` is a
directory receiving ``hparam_selection.csv`` and ``hparam_selection.json``.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass

import yaml

try:
    from tools.retention import summarize_retention
except ImportError:
    from retention import summarize_retention


VAL_FILE = "iwildcam_val_metrics.json"
ID_FILE = "iwildcam_id_test_metrics.json"
OOD_FILE = "iwildcam_test_metrics.json"
_SEED_RE = re.compile(r"-seed(\d+)")
_FRACTION_RE = re.compile(
    r"-(?:lf|frac|label[_-]frac(?:tion)?)(?:=)?"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
    re.IGNORECASE,
)
_LABEL_EFFICIENCY_RE = re.compile(r"-label[_-]efficiency", re.IGNORECASE)


def _load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _find_metric(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            value = obj[key]
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
        for value in obj.values():
            found = _find_metric(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_metric(value, key)
            if found is not None:
                return found
    return None


def _find_run_info(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("run_info"), dict):
            return obj["run_info"]
        for value in obj.values():
            found = _find_run_info(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_run_info(value)
            if found is not None:
                return found
    return None


def normalize_config_name(run_name):
    """Remove seed and label-efficiency/fraction components from a run name."""
    config = _SEED_RE.sub("", run_name)
    config = _FRACTION_RE.sub("", config)
    config = _LABEL_EFFICIENCY_RE.sub("", config)
    return re.sub(r"-+", "-", config).strip("-")


def _is_label_efficiency_run(run_name):
    return bool(_FRACTION_RE.search(run_name) or _LABEL_EFFICIENCY_RE.search(run_name))


def _load_params(run_dir):
    for filename in ("params.yaml", "params-eval.yaml", "params-supervised.yaml"):
        path = os.path.join(run_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as handle:
                value = yaml.load(handle, Loader=yaml.FullLoader)
            if isinstance(value, dict):
                return value
        except (OSError, yaml.YAMLError):
            pass
    return {}


def _model_key(run_name, params):
    meta = params.get("meta", {}) if isinstance(params, dict) else {}
    data = params.get("data", {}) if isinstance(params, dict) else {}
    mask = params.get("mask", {}) if isinstance(params, dict) else {}
    submission = meta.get("submission_model_name")
    if submission:
        return str(submission)
    name = meta.get("model_name")
    if name:
        parts = [str(name)]
        if meta.get("tag"):
            parts.append(str(meta["tag"]))
        patch = mask.get("patch_size", meta.get("patch_size"))
        crop = data.get("crop_size", meta.get("crop_size"))
        if patch is not None:
            parts.append(f"p{patch}")
        if crop is not None:
            parts.append(f"c{crop}")
        return "-".join(parts)
    # Run names produced by build_run_name commonly begin with a submission
    # prefix followed by one of these architecture markers.
    match = re.search(r"-(?=vit_|dinov2_|mae_|sup_|clip_)", run_name)
    if match:
        return run_name[: match.start()]
    return "all"


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def best_validation_from_csv(path):
    """Return (minimum OOD val loss, val acc at that epoch, best epoch)."""
    rows_by_epoch = {}
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"epoch", "val_loss"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} is not a supervised validation log")
        for row in reader:
            try:
                epoch = int(row["epoch"])
            except (TypeError, ValueError):
                continue  # Repeated header after submitit resume.
            loss = _finite_float(row.get("val_loss"))
            if loss is not None:
                rows_by_epoch[epoch] = (loss, _finite_float(row.get("val_acc")))
    if not rows_by_epoch:
        raise ValueError(f"No finite val_loss rows in {path}")
    epoch, (loss, acc) = min(
        rows_by_epoch.items(),
        key=lambda item: (
            item[1][0],
            -(item[1][1] if item[1][1] is not None else -math.inf),
            item[0],
        ),
    )
    return loss, acc, epoch


def _candidate_training_dirs(run_name, run_dir, train_roots):
    yield run_dir
    seen = {os.path.abspath(run_dir)}
    for root in train_roots:
        direct = os.path.join(root, run_name)
        candidates = [direct]
        if not os.path.isdir(direct) and os.path.isdir(root):
            for dirpath, dirnames, _ in os.walk(root):
                dirnames.sort()
                if os.path.basename(dirpath) == run_name:
                    candidates.append(dirpath)
        for candidate in candidates:
            absolute = os.path.abspath(candidate)
            if absolute not in seen and os.path.isdir(candidate):
                seen.add(absolute)
                yield candidate


def _validation_from_training_logs(run_name, run_dir, train_roots):
    for directory in _candidate_training_dirs(run_name, run_dir, train_roots):
        preferred = os.path.join(directory, "linear_probe_r0.csv")
        paths = [preferred] if os.path.isfile(preferred) else []
        paths += sorted(
            path for path in glob.glob(os.path.join(directory, "*_r*.csv")) if path != preferred
        )
        for path in paths:
            try:
                return (*best_validation_from_csv(path), path)
            except (OSError, ValueError, csv.Error):
                continue
    return None, None, None, None


@dataclass
class RunRecord:
    run_name: str
    run_dir: str
    seed: int
    model: str
    config: str
    val_f1: float = None
    val_acc: float = None
    val_loss: float = None
    best_epoch: int = None
    training_log: str = None
    id_f1: float = None
    id_acc: float = None
    ood_f1: float = None
    ood_acc: float = None


def collect_runs(root, train_roots=(), model_prefix=None):
    """Read existing run directories without importing torch or WILDS."""
    root = os.path.abspath(root)
    train_roots = [os.path.abspath(path) for path in train_roots]
    grouped = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Evaluation root not found: {root}")
    for run_name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, run_name)
        if not os.path.isdir(run_dir):
            continue
        metrics = {name: _load_json(os.path.join(run_dir, name)) for name in (VAL_FILE, ID_FILE, OOD_FILE)}
        if not any(value is not None for value in metrics.values()):
            continue
        params = _load_params(run_dir)
        config = normalize_config_name(run_name)
        model = _model_key(run_name, params)
        if model_prefix and not any(
            value.startswith(model_prefix) for value in (run_name, config, model)
        ):
            continue
        run_info = _find_run_info(metrics[OOD_FILE]) or _find_run_info(metrics[ID_FILE]) or {}
        seed_value = run_info.get("seed")
        match = _SEED_RE.search(run_name)
        seed = int(seed_value) if seed_value is not None else (int(match.group(1)) if match else 0)
        val_loss, csv_acc, best_epoch, training_log = _validation_from_training_logs(
            run_name, run_dir, train_roots
        )
        record = RunRecord(
            run_name=run_name,
            run_dir=run_dir,
            seed=seed,
            model=model,
            config=config,
            val_f1=_find_metric(metrics[VAL_FILE], "F1-macro_all"),
            val_acc=_find_metric(metrics[VAL_FILE], "acc_avg"),
            val_loss=val_loss,
            best_epoch=best_epoch,
            training_log=training_log,
            id_f1=_find_metric(metrics[ID_FILE], "F1-macro_all"),
            id_acc=_find_metric(metrics[ID_FILE], "acc_avg"),
            ood_f1=_find_metric(metrics[OOD_FILE], "F1-macro_all"),
            ood_acc=_find_metric(metrics[OOD_FILE], "acc_avg"),
        )
        # Prefer metrics-file val accuracy; CSV accuracy supplies the loss tiebreak.
        if record.val_acc is None:
            record.val_acc = csv_acc
        key = (model, config, seed)
        if key in grouped:
            previous = grouped[key]
            previous_fraction = _is_label_efficiency_run(previous.run_name)
            current_fraction = _is_label_efficiency_run(run_name)
            if previous_fraction and not current_fraction:
                grouped[key] = record  # Full-label runs take precedence for HP selection.
                continue
            if current_fraction and not previous_fraction:
                continue
            raise ValueError(
                f"Duplicate run for model/config/seed {key}: {previous.run_dir} and {run_dir}. "
                "Separate label-fraction results before HP selection."
            )
        grouped[key] = record
    return grouped


def _mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, std


def _config_row(model, config, records):
    records = sorted(records, key=lambda record: record.seed)
    seeds = [record.seed for record in records]
    val_f1_complete = all(record.val_f1 is not None for record in records)
    val_loss_complete = all(record.val_loss is not None for record in records)
    if val_f1_complete:
        criterion = "val_macro_f1"
        priority = 2
        criterion_values = [record.val_f1 for record in records]
    elif val_loss_complete:
        criterion = "best_val_loss"
        priority = 1
        criterion_values = [record.val_loss for record in records]
    else:
        criterion = "incomplete"
        priority = 0
        criterion_values = []
    criterion_mean, criterion_std = _mean_std(criterion_values)
    val_acc_mean, val_acc_std = _mean_std([record.val_acc for record in records])
    id_f1_mean, id_f1_std = _mean_std([record.id_f1 for record in records])
    id_acc_mean, id_acc_std = _mean_std([record.id_acc for record in records])
    ood_f1_mean, ood_f1_std = _mean_std([record.ood_f1 for record in records])
    ood_acc_mean, ood_acc_std = _mean_std([record.ood_acc for record in records])
    retention = summarize_retention(
        [record.id_f1 for record in records], [record.ood_f1 for record in records]
    )
    return {
        "model": model,
        "config": config,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "criterion": criterion,
        "criterion_priority": priority,
        "val_criterion_mean": criterion_mean,
        "val_criterion_std": criterion_std,
        "val_acc_mean": val_acc_mean,
        "val_acc_std": val_acc_std,
        "id_test_macro_f1_mean": id_f1_mean,
        "id_test_macro_f1_std": id_f1_std,
        "id_test_avg_acc_mean": id_acc_mean,
        "id_test_avg_acc_std": id_acc_std,
        "ood_test_macro_f1_mean": ood_f1_mean,
        "ood_test_macro_f1_std": ood_f1_std,
        "ood_test_avg_acc_mean": ood_acc_mean,
        "ood_test_avg_acc_std": ood_acc_std,
        "retention_mean": retention["mean"],
        "retention_std": retention["std"],
        "test_complete": all(record.ood_f1 is not None for record in records),
        "run_names": [record.run_name for record in records],
        "training_logs": [record.training_log for record in records],
        "val_rank": None,
        "selected_by_val": False,
        "selected_by_test": False,
    }


def _selection_sort(row):
    val_acc = row["val_acc_mean"] if row["val_acc_mean"] is not None else -math.inf
    if row["criterion"] == "val_macro_f1":
        return (-row["val_criterion_mean"], -val_acc, row["config"])
    return (row["val_criterion_mean"], -val_acc, row["config"])


def select_configurations(grouped_runs):
    """Return per-model selections and the flat configuration table."""
    by_config = {}
    for (model, config, _seed), record in grouped_runs.items():
        by_config.setdefault((model, config), []).append(record)
    rows_by_model = {}
    for (model, config), records in sorted(by_config.items()):
        rows_by_model.setdefault(model, []).append(_config_row(model, config, records))

    result = {"models": {}, "configurations": []}
    for model, rows in sorted(rows_by_model.items()):
        highest_priority = max((row["criterion_priority"] for row in rows), default=0)
        candidates = [row for row in rows if row["criterion_priority"] == highest_priority and highest_priority]
        candidates.sort(key=_selection_sort)
        for rank, row in enumerate(candidates, 1):
            row["val_rank"] = rank
        selected = candidates[0] if candidates else None
        if selected:
            selected["selected_by_val"] = True
        test_candidates = [row for row in rows if row["test_complete"]]
        test_selected = max(
            test_candidates,
            key=lambda row: (row["ood_test_macro_f1_mean"], row["config"]),
        ) if test_candidates else None
        if test_selected:
            test_selected["selected_by_test"] = True
        result["models"][model] = {
            "selected_config": selected["config"] if selected else None,
            "selection_criterion": selected["criterion"] if selected else None,
            "test_selected_config": test_selected["config"] if test_selected else None,
            "selection_changed_vs_test": bool(
                selected and test_selected and selected["config"] != test_selected["config"]
            ),
        }
        result["configurations"].extend(rows)
    return result


def _fmt(mean, std=None, digits=4):
    if mean is None:
        return "NA"
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _print_results(result):
    for model, selection in result["models"].items():
        rows = [row for row in result["configurations"] if row["model"] == model]
        print(f"\nModel: {model}")
        print("rank  criterion       seeds  val criterion        OOD-test F1       config")
        print("-" * 110)
        for row in sorted(rows, key=lambda row: (row["val_rank"] is None, row["val_rank"] or 0, row["config"])):
            rank = row["val_rank"] if row["val_rank"] is not None else "-"
            print(
                f"{str(rank):>4}  {row['criterion']:<15} {str(row['seeds']):<6} "
                f"{_fmt(row['val_criterion_mean'], row['val_criterion_std']):<20} "
                f"{_fmt(row['ood_test_macro_f1_mean'], row['ood_test_macro_f1_std']):<17} {row['config']}"
            )
        selected_name = selection["selected_config"]
        selected = next((row for row in rows if row["config"] == selected_name), None)
        print(f"\nOOD-validation selected: {selected_name} ({selection['selection_criterion']})")
        if selected:
            for label, prefix in (("ID test", "id_test"), ("OOD test", "ood_test")):
                print(
                    f"  {label}: macro-F1 {_fmt(selected[prefix + '_macro_f1_mean'], selected[prefix + '_macro_f1_std'])}; "
                    f"avg acc {_fmt(selected[prefix + '_avg_acc_mean'], selected[prefix + '_avg_acc_std'])}"
                )
            print(f"  Retention R: {_fmt(selected['retention_mean'], selected['retention_std'])}")
        print(
            "DIAGNOSTIC ONLY — OOD test would select: "
            f"{selection['test_selected_config']}. Test-based selection is not valid for reporting."
        )
        print(f"Selection differs between validation and test: {selection['selection_changed_vs_test']}")


def write_results(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "hparam_selection.json")
    csv_path = os.path.join(out_dir, "hparam_selection.csv")
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fields = [
        "model", "config", "num_seeds", "seeds", "criterion", "val_rank",
        "selected_by_val", "selected_by_test", "val_criterion_mean", "val_criterion_std",
        "val_acc_mean", "val_acc_std", "id_test_macro_f1_mean", "id_test_macro_f1_std",
        "id_test_avg_acc_mean", "id_test_avg_acc_std", "ood_test_macro_f1_mean",
        "ood_test_macro_f1_std", "ood_test_avg_acc_mean", "ood_test_avg_acc_std",
        "retention_mean", "retention_std",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result["configurations"]:
            output = dict(row)
            output["seeds"] = " ".join(map(str, row["seeds"]))
            writer.writerow(output)
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiment_logs/eval-wilds")
    parser.add_argument(
        "--train-root", action="append", default=[],
        help="parent of retained training run directories; repeat for custom locations",
    )
    parser.add_argument("--model-prefix", help="only run/config/model names with this prefix")
    parser.add_argument("--out", default="experiment_logs/hparam-selection", help="output directory")
    args = parser.parse_args(argv)
    grouped = collect_runs(args.root, train_roots=args.train_root, model_prefix=args.model_prefix)
    if not grouped:
        raise ValueError("No matching evaluation runs with val/ID-test/OOD-test metrics found")
    result = select_configurations(grouped)
    _print_results(result)
    csv_path, json_path = write_results(result, args.out)
    print(f"\nWrote {csv_path}\nWrote {json_path}")
    return result


if __name__ == "__main__":
    main()
