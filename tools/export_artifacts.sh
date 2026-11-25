#!/usr/bin/env bash
# Package small paper artifacts, never model weights, images, or WILDS data.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="experiment_logs"
OUTPUT_DIR="."

usage() {
  echo "Usage: bash tools/export_artifacts.sh [--root experiment_logs] [--output-dir .]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${ROOT}" = /* ]]; then
  ROOT_ABS="${ROOT}"
else
  ROOT_ABS="${PROJECT_ROOT}/${ROOT}"
fi
if [[ "${OUTPUT_DIR}" = /* ]]; then
  OUTPUT_ABS="${OUTPUT_DIR}"
else
  OUTPUT_ABS="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi
if [[ ! -d "${ROOT_ABS}" ]]; then
  echo "Artifact root not found: ${ROOT_ABS}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ABS}"

DATE="$(date -u +%Y%m%d)"
ARCHIVE="${OUTPUT_ABS}/tmlr_artifacts_${DATE}.tar.gz"
if [[ -e "${ARCHIVE}" ]]; then
  echo "Refusing to overwrite existing archive: ${ARCHIVE}" >&2
  exit 1
fi

ROOT_PARENT="$(dirname "${ROOT_ABS}")"
ROOT_NAME="$(basename "${ROOT_ABS}")"
LIST="$(mktemp)"
trap 'rm -f "${LIST}"' EXIT

# src/train.py writes pretraining loss as <write_tag>_r<rank>.csv (normally
# jepa_r0.csv) in its logging folder. The *_r[0-9]*.csv rule includes both those
# logs and supervised per-epoch linear_probe_r*.csv files. Cached runs add
# epochs.csv, summaries/predictions, selected grids, and cached_selection.*.
# Only features/<KEY>/manifest.json is included; never .npy feature arrays.
(
  cd "${ROOT_PARENT}"
  find "${ROOT_NAME}" \
    -type d -name wilds_data -prune -o \
    -type f \( \
      -name '*_metrics.json' -o \
      -name '*_predictions.csv' -o \
      -name '*_r[0-9]*.csv' -o \
      -name 'epochs.csv' -o \
      -name 'params*.yaml' -o \
      -name '*_selected.yaml' -o \
      -name 'summary*.json' -o \
      -name 'summary*.csv' -o \
      -name 'cached_selection.json' -o \
      -name 'cached_selection.csv' -o \
      -name 'cached_selection.tex' -o \
      -path '*/features/*/manifest.json' \
    \) -print | LC_ALL=C sort
) > "${LIST}"

if [[ ! -s "${LIST}" ]]; then
  echo "No matching small artifacts found under ${ROOT_ABS}" >&2
  exit 1
fi

# Paths in the manifest are relative to ROOT_PARENT, so archive members retain
# the experiment_logs/... layout without absolute paths.
tar -C "${ROOT_PARENT}" -czf "${ARCHIVE}" --no-recursion -T "${LIST}"
COUNT="$(wc -l < "${LIST}" | tr -d ' ')"
BYTES="$(stat -c '%s' "${ARCHIVE}")"
HUMAN="$(du -h "${ARCHIVE}" | cut -f1)"
printf 'Created %s\n' "${ARCHIVE}"
printf 'Files: %s; archive size: %s (%s bytes)\n' "${COUNT}" "${HUMAN}" "${BYTES}"
