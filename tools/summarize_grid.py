import argparse
import json
import os


def _find_metric(metrics, key):
    if isinstance(metrics, dict):
        if key in metrics:
            return metrics[key]
        for value in metrics.values():
            found = _find_metric(value, key)
            if found is not None:
                return found
    return None


def _collect_metrics(root_dir, key):
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith("_metrics.json"):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "r") as f:
                    metrics = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            value = _find_metric(metrics, key)
            if value is None:
                continue
            run_name = os.path.basename(os.path.dirname(path))
            rows.append((float(value), run_name, path))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiment_logs/eval-wilds",
        help="root folder for eval logs",
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        help="metric key to rank by",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of runs to display",
    )
    args = parser.parse_args()

    rows = _collect_metrics(args.root, args.metric)
    rows.sort(key=lambda r: r[0], reverse=True)

    if not rows:
        print("No metrics found.")
        return

    print(f"Ranking by '{args.metric}' (top {min(args.top, len(rows))})")
    for idx, (value, run_name, path) in enumerate(rows[: args.top], start=1):
        print(f"{idx:>3} | {value:.6f} | {run_name} | {path}")


if __name__ == "__main__":
    main()
