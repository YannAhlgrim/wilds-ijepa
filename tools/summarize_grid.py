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
    if isinstance(metrics, list):
        for item in metrics:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _parse_yaml_value(value):
    value = value.strip()
    if not value or value in ("null", "~"):
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _load_params_simple(dirpath):
    path = os.path.join(dirpath, "params.yaml")
    alt_path = os.path.join(dirpath, "params-eval.yaml")
    for p in [path, alt_path]:
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    return _parse_simple_yaml(f.read())
            except (OSError, ValueError):
                continue
    return None


def _parse_simple_yaml(text):
    lines = text.split("\n")

    def _parse_block(start, indent):
        result = {}
        list_items = []
        is_list = False
        i = start

        while i < len(lines):
            raw = lines[i].rstrip()
            if not raw.strip() or raw.strip().startswith("#"):
                i += 1
                continue

            stripped = raw.lstrip(" ")
            cur_indent = len(raw) - len(stripped)

            if cur_indent < indent:
                break
            if cur_indent > indent:
                i += 1
                continue

            if " #" in stripped:
                effective = stripped[: stripped.index(" #")].rstrip()
            else:
                effective = stripped

            if not effective:
                i += 1
                continue

            if effective.startswith("- "):
                is_list = True
                list_items.append(_parse_yaml_value(effective[2:]))
                i += 1
            elif effective.endswith(":"):
                key = effective[:-1].strip()
                sub_val, i = _parse_block(i + 1, indent + 2)
                result[key] = sub_val
            elif ": " in effective:
                key, _, val_str = effective.partition(": ")
                result[key.strip()] = _parse_yaml_value(val_str)
                i += 1
            else:
                i += 1

        if is_list:
            return list_items, i
        return result, i

    return _parse_block(0, 0)[0]


def _get_in_params(params, path):
    if params is None:
        return None
    keys = path.split(".")
    current = params
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


_MODEL_PREFIXES = {
    "vith": "vit_huge",
    "vitb": "vit_base",
    "vitl": "vit_large",
    "vitt": "vit_tiny",
    "vits": "vit_small",
    "vitg": "vit_giant",
}


def _infer_model_from_run_name(run_name):
    for prefix, model in _MODEL_PREFIXES.items():
        if run_name.lower().startswith(prefix):
            return model
    return None


def _get_model_name(dirpath, run_name=None):
    params = _load_params_simple(dirpath)
    name = None
    if params is not None:
        name = _get_in_params(params, "meta.model_name")
    if name is None and run_name:
        name = _infer_model_from_run_name(run_name)
    return name or "unknown"


def _collect_rows(root_dir, metric_key, col_paths):
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
            value = _find_metric(metrics, metric_key)
            if value is None:
                continue
            run_name = os.path.basename(os.path.dirname(path))
            model_name = _get_model_name(dirpath, run_name)
            params = _load_params_simple(dirpath)
            col_values = [
                _get_in_params(params, cp) for cp in col_paths
            ]
            rows.append(
                (float(value), model_name, col_values, run_name, path)
            )
    return rows


def _format_value(val, width=None):
    if val is None:
        s = "\u2014"
    elif isinstance(val, bool):
        s = str(val)
    elif isinstance(val, int):
        s = str(val)
    elif isinstance(val, float):
        if abs(val) < 0.001 or abs(val) >= 10000:
            s = f"{val:.2e}"
        elif val == int(val):
            s = f"{val:.1f}"
        else:
            s = f"{val:.6f}".rstrip("0").rstrip(".")
    elif isinstance(val, list):
        s = str(val)
    else:
        s = str(val)
    if width is not None and len(s) > width:
        s = s[: width - 1] + "\u2026"
    return s


def _col_width(header, values):
    w = len(header)
    for val in values:
        s = _format_value(val)
        w = max(w, len(s))
    return w


_BLOCK = "\u2588"
_LIGHT_H = "\u2500"
_LIGHT_V = "\u2502"
_LIGHT_D = "\u253c"
_HEAVY_H = "\u2501"
_HEAVY_V = "\u2503"
_HEAVY_D = "\u254b"


def _print_separator(widths, heavy=False):
    h = _HEAVY_H if heavy else _LIGHT_H
    d = _HEAVY_D if heavy else _LIGHT_D
    parts = []
    for i, w in enumerate(widths):
        if i > 0:
            parts.append(d)
        segments = max(w, 1) - 1
        if segments <= 0:
            parts.append(h)
        else:
            parts.append(h * segments)
    line = h.join(parts) if len(parts) > 0 else ""
    print(f"  {line}")


def _print_row(values, widths, align_right=None):
    if align_right is None:
        align_right = [True] * len(values)
    parts = []
    for i, (val, w) in enumerate(zip(values, widths)):
        s = _format_value(val, w)
        if align_right[i]:
            s = s.rjust(w)
        else:
            s = s.ljust(w)
        if i > 0:
            parts.append(f" {_LIGHT_V} ")
        parts.append(s)
    print("  " + "".join(parts))


def _print_header(full_cols, widths):
    align = [False] + [True] * (len(full_cols) - 2) + [False]
    _print_row(full_cols, widths, align)
    _print_separator(widths)


def _print_results(model_type, rows, metric_key, col_headers, top):
    if not rows:
        return

    display_rows = rows[:top]

    n = len(display_rows)
    rank_width = max(3, len(str(n)))
    metric_header = metric_key
    metric_vals = [r[0] for r in display_rows]
    metric_width = max(
        len(metric_header), max(len(_format_value(v)) for v in metric_vals)
    )

    col_widths = []
    for i, ch in enumerate(col_headers):
        col_vals = [r[2][i] for r in display_rows]
        cw = _col_width(ch, col_vals)
        col_widths.append(cw)

    run_name_vals = [r[3] for r in display_rows]
    run_name_width = max(
        len("run_name"), max(len(v) for v in run_name_vals)
    )

    widths = [rank_width, metric_width] + col_widths + [run_name_width]

    full_cols = (
        ["#", metric_header] + col_headers + ["run_name"]
    )

    title = f"{model_type} \u2014 Top {top} by {metric_key}"
    sep_len = sum(w + 3 for w in widths) + 2
    print(_HEAVY_H * sep_len)
    print(f"  {title}")
    print(_HEAVY_H * sep_len)
    _print_header(full_cols, widths)

    for idx, row in enumerate(display_rows, start=1):
        metric_str = _format_value(row[0])
        col_strs = [_format_value(row[2][i]) for i in range(len(col_headers))]
        vals = [str(idx), metric_str] + col_strs + [row[3]]
        align = [False] + [True] * (len(full_cols) - 2) + [False]
        _print_row(vals, widths, align)

    print()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Summarize grid evaluation results grouped by model type.

Examples:
  %(prog)s --top 10
  %(prog)s --metric macro_f1 --cols data.batch_size optimization.lr --top 5
""",
    )
    parser.add_argument(
        "--root",
        default="experiment_logs/eval-wilds",
        help="root folder for eval logs (default: %(default)s)",
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        help="metric key to rank by (default: %(default)s)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="top N results per model type (default: %(default)s)",
    )
    parser.add_argument(
        "--cols",
        nargs="*",
        default=[],
        help="extra columns from params.yaml, e.g. data.batch_size optimization.lr",
    )
    args = parser.parse_args()

    rows = _collect_rows(args.root, args.metric, args.cols)

    if not rows:
        print("No metrics found.")
        return

    col_headers = [c.split(".")[-1] for c in args.cols]

    groups = {}
    for row in rows:
        model = row[1]
        groups.setdefault(model, []).append(row)

    for model_type in sorted(groups.keys()):
        group = groups[model_type]
        group.sort(key=lambda r: r[0], reverse=True)
        _print_results(model_type, group, args.metric, col_headers, args.top)


if __name__ == "__main__":
    main()
