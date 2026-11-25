#!/usr/bin/env bash
# Run seed-0 HP selection, five-seed confirmation, and label-efficiency probes.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash tools/run_cached_protocol.sh <encoder-key> <features-dir> <out-dir>" >&2
  exit 2
fi
KEY="$1"
FEATURES="$2"
OUT="$3"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cpu}"
GRID="${GRID:-configs/grids/cached/default.yaml}"
case "${GRID}" in
  default|with_cls) GRID="configs/grids/cached/${GRID}.yaml" ;;
esac
MAX_EPOCHS="${MAX_EPOCHS:-100}"
SEEDS="${SEEDS:-0 1 2 3 4}"

# Intended for direct use on a workstation or inside one small allocation, e.g.
# srun --gpus 1 --cpus-per-task 8 bash tools/run_cached_protocol.sh ...
"${PYTHON}" tools/probe_cached.py \
  --features "${FEATURES}" --grid "${GRID}" --seeds 0 \
  --stage grid --device "${DEVICE}" --max-epochs "${MAX_EPOCHS}" --out "${OUT}"

# This first selection writes <key>_selected.yaml even before five-seed test
# metrics exist. It uses only the seed-0 OOD-val grid results.
"${PYTHON}" tools/select_cached.py \
  --runs "${OUT}/${KEY}" --out "${OUT}/${KEY}/selection-stage1"
SELECTED="${OUT}/${KEY}/selection-stage1/${KEY}_selected.yaml"

# Each probe_cached.py invocation handles every config x seed in its stage in a
# single Python process. Device-resident arrays are shared across those runs;
# completed summary.json files are still skipped after a timeout/resubmission.
# shellcheck disable=SC2086
"${PYTHON}" tools/probe_cached.py \
  --features "${FEATURES}" --grid "${SELECTED}" --seeds ${SEEDS} \
  --stage seeds --device "${DEVICE}" --max-epochs "${MAX_EPOCHS}" --out "${OUT}"

for FRACTION in 0.01 0.1 0.5; do
  # shellcheck disable=SC2086
  "${PYTHON}" tools/probe_cached.py \
    --features "${FEATURES}" --grid "${SELECTED}" --seeds ${SEEDS} \
    --label-fraction "${FRACTION}" --stage "label_fraction_${FRACTION}" \
    --device "${DEVICE}" --max-epochs "${MAX_EPOCHS}" --out "${OUT}"
done

"${PYTHON}" tools/select_cached.py \
  --runs "${OUT}/${KEY}" --out "${OUT}/${KEY}/selection"
echo "Cached protocol complete for ${KEY}. See ${OUT}/${KEY}/selection/"
