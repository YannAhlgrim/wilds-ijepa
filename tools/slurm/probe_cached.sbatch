#!/usr/bin/env bash
#SBATCH --job-name=probe-cached
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
# Submit with: sbatch --partition=P --export=ALL,KEY=vith14_224_in22k tools/slurm/probe_cached.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
: "${KEY:?Set KEY with --export=ALL,KEY=<encoder-key>}"
FEATURES_ROOT="${FEATURES_ROOT:-experiment_logs/features}"
RUNS_ROOT="${RUNS_ROOT:-experiment_logs/cached-runs}"
DEVICE="${DEVICE:-cuda}"
GRID="${GRID:-configs/grids/cached/default.yaml}"
case "${GRID}" in
  default|with_cls) GRID="configs/grids/cached/${GRID}.yaml" ;;
esac
export GRID
if [[ -n "${VENV:-}" ]]; then
  if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "VENV activation script not found: ${VENV}/bin/activate" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
fi
PYTHON="${PYTHON:-python3}"

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}; key: ${KEY}"
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="probe-${KEY}"
fi
nvidia-smi -L

FEATURE_DIR="${FEATURES_ROOT}/${KEY}"
"${PYTHON}" - "${FEATURE_DIR}" "${GRID}" <<'PY'
import os
import sys
import json
import yaml

feature_dir = sys.argv[1]
grid_path = sys.argv[2]
manifest_path = os.path.join(feature_dir, "manifest.json")
if not os.path.isfile(manifest_path):
    raise SystemExit(f"Incomplete extraction: missing {manifest_path}")
with open(manifest_path) as handle:
    manifest = json.load(handle)
required = ("train", "val", "id_val", "id_test", "test")
with open(grid_path) as handle:
    grid = yaml.safe_load(handle)
representations = grid.get("grid", grid).get("representation_type", [])
if not isinstance(representations, list):
    representations = [representations]
requires_cls = "cls" in representations
problems = []
for split in required:
    info = manifest.get("splits", {}).get(split)
    if not info or not info.get("complete"):
        problems.append(f"{split}: not marked complete")
        continue
    files = info.get("files", {})
    for required_file in ("last_avgpool", "last4_avgpool_concat", "y", "metadata"):
        filename = files.get(required_file)
        if not filename:
            problems.append(f"{split}: missing manifest entry for {required_file}")
        elif not os.path.isfile(os.path.join(feature_dir, filename)):
            problems.append(f"{split}: missing {filename}")
    if requires_cls:
        cls_file = files.get("cls")
        if not cls_file:
            problems.append(f"{split}: GRID requires CLS but manifest has no cls entry")
        elif not os.path.isfile(os.path.join(feature_dir, cls_file)):
            problems.append(f"{split}: missing CLS feature file {cls_file}")
if problems:
    raise SystemExit("Incomplete extraction:\n  " + "\n  ".join(problems))
print(f"Complete extraction manifest verified: {manifest_path}")
PY

DEVICE="${DEVICE}" GRID="${GRID}" bash tools/run_cached_protocol.sh \
  "${KEY}" "${FEATURE_DIR}" "${RUNS_ROOT}"
