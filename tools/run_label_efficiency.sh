#!/usr/bin/env bash
#
# Run label-efficiency supervised experiments for all model grids.
#
# For every grid under configs/grids/label_efficiency/, this submits one
# submitit job per seed (via tools/run_grid.py). Models are launched
# sequentially so you can run them "one by one"; within a grid, the 5 seeds
# are submitted together.
#
# Each run automatically:
#   - uses a stratified subset of the Source split (data.label_fraction)
#   - seeds training from meta.seed
#   - evaluates on id_test (ID) and test (OOD) WILDS splits
#   - records WILDS metrics + training time + epochs_run into the metrics JSON
#
# After all jobs finish, aggregate with:
#   python3 tools/aggregate_label_efficiency.py --root experiment_logs/eval-wilds
#
# Usage:
#   bash tools/run_label_efficiency.sh [--partition P] [--time MIN] [--folder DIR]
#                                    [--models "vith14_224_in22k vitg16_224_in22k"]
#                                    [--fractions "0.01 0.10 0.50"]
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_DIR="${PROJECT_ROOT}/configs/grids/label_efficiency"

PARTITION=""
TIME=""
FOLDER=""
MODELS=""
FRACTIONS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)  PARTITION="$2"; shift 2 ;;
    --time)       TIME="$2"; shift 2 ;;
    --folder)     FOLDER="$2"; shift 2 ;;
    --models)     MODELS="$2"; shift 2 ;;
    --fractions)  FRACTIONS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Resolve the list of grid files to run.
if [[ -n "${MODELS}" ]]; then
  GRIDS=()
  for m in ${MODELS}; do
    if [[ -n "${FRACTIONS}" ]]; then
      for frac in ${FRACTIONS}; do
        g="${GRID_DIR}/${m}_frac${frac}.yaml"
        if [[ ! -f "${g}" ]]; then
          echo "Grid not found for model '${m}' fraction '${frac}': ${g}" >&2
          exit 1
        fi
        GRIDS+=("${g}")
      done
    else
      for g in "${GRID_DIR}/${m}"_frac*.yaml; do
        if [[ -f "${g}" ]]; then
          GRIDS+=("${g}")
        fi
      done
    fi
  done
else
  # All models and fractions, sorted.
  GRIDS=()
  while IFS= read -r g; do GRIDS+=("${g}"); done < <(ls "${GRID_DIR}"/*.yaml | sort)
fi

if [[ ${#GRIDS[@]} -eq 0 ]]; then
  echo "No grid files found in ${GRID_DIR}" >&2
  exit 1
fi

echo "Launching label-efficiency sweeps for ${#GRIDS[@]} grid(s):"
for g in "${GRIDS[@]}"; do echo "  - $(basename "${g}")"; done
echo

for g in "${GRIDS[@]}"; do
  echo "=================================================================="
  echo "Model grid: $(basename "${g}")"
  echo "=================================================================="

  cmd=("${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/tools/run_grid.py" --grid "${g}")
  [[ -n "${PARTITION}" ]] && cmd+=(--partition "${PARTITION}")
  [[ -n "${TIME}" ]]      && cmd+=(--time "${TIME}")
  [[ -n "${FOLDER}" ]]    && cmd+=(--folder "${FOLDER}")

  echo "+ ${cmd[*]}"
  "${cmd[@]}"
  echo
done

echo "All label-efficiency jobs submitted."
echo "When they finish, aggregate results with:"
echo "  ${PROJECT_ROOT}/.venv/bin/python tools/aggregate_label_efficiency.py --root experiment_logs/eval-wilds"
