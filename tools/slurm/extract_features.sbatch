#!/usr/bin/env bash
#SBATCH --job-name=extract-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x-%j.out
# Submit with: sbatch --partition=P --export=ALL,KEY=vith14_224_in22k tools/slurm/extract_features.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
: "${KEY:?Set KEY with --export=ALL,KEY=<encoder-key>}"
FEATURES_ROOT="${FEATURES_ROOT:-experiment_logs/features}"
CONFIG="configs/supervised_${KEY}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown encoder ${KEY}: ${CONFIG} not found" >&2
  exit 1
fi
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
  scontrol update JobId="${SLURM_JOB_ID}" JobName="extract-${KEY}"
fi
nvidia-smi -L

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  if [[ ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer" >&2
    exit 1
  fi
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

"${PYTHON}" tools/extract_features.py \
  --config "${CONFIG}" \
  --out "${FEATURES_ROOT}/${KEY}" \
  --splits train val id_val id_test test \
  --batch-size 16 --num-workers 8 --dtype float16 \
  --device cuda "${LIMIT_ARGS[@]}"
