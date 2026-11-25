# Cached-feature linear-probe protocol

All 13 encoders use one protocol: deterministic frozen features, seed-0 HP
selection on OOD validation, five-seed confirmation, and label-efficiency runs.
The 13 keys are `vith14_224_in1k`, `vith16_448_in1k`,
`vith14_224_in22k`, `vitg16_224_in22k`, `vith14_224` (iWildCam),
`vitb16_448` (iWildCam), `vith14_224_randinit`, `dinov2_vitg14`,
`dinov2_vitl14`, `mae_vith14`, `sup_vith14_in21k`, `clip_vitl14`, and
`clip_vitl14_336`.

## 1. Extract each encoder once

Run from the repository root with the checkpoints and WILDS data in the paths
specified by the existing YAMLs. The following loop is the complete per-encoder
extraction command list. It uses one GPU, 16-image micro-batches, and all five
splits; no training or augmentation is performed.

```bash
for KEY in vith14_224_in1k vith16_448_in1k vith14_224_in22k vitg16_224_in22k vith14_224 vitb16_448 vith14_224_randinit dinov2_vitg14 dinov2_vitl14 mae_vith14 sup_vith14_in21k clip_vitl14 clip_vitl14_336; do
  python3 tools/extract_features.py \
    --config "configs/supervised_${KEY}.yaml" \
    --out "experiment_logs/features/${KEY}" \
    --splits train val id_val id_test test \
    --batch-size 16 --num-workers 8 --dtype float16
done
```

For a smoke test, append `--limit 128` (the limit applies independently to each
split). Use `--device cpu` only for tiny tests; full extraction is intended for a
GPU. The equivalent optional submitit form is:

```bash
python3 tools/extract_features.py --config configs/supervised_vith14_224_in22k.yaml --out experiment_logs/features/vith14_224_in22k --submitit --partition "$PARTITION" --time 180 --gpus 1
```

Each forward computes `last_avgpool`, `last4_avgpool_concat`, and CLS where the
architecture has one. Arrays are memory-mappable float16 `.npy` files with labels
and WILDS metadata. Partial arrays are flushed per chunk; progress and the
manifest are updated atomically. Rerunning resumes incomplete splits and skips
completed ones. Local checkpoint files are SHA-256 hashed. Public timm weights
use a `timm://<identifier>` source plus a SHA-256 fingerprint of the loaded
encoder state. Random initialization records its fixed seed.

### SLURM pilot and full submission

The scripts in `tools/slurm/` use one node/task/GPU, 8 CPUs, 64 GB RAM, and
conservative 3-hour extraction / 12-hour probing limits. No partition is
hardcoded. Create the log directory once and optionally set `VENV` to an
existing environment. If compute nodes have no internet, prefetch the six
public external models on a login node first; this honors `HF_HOME`/`TORCH_HOME`:

```bash
mkdir -p slurm_logs
python3 tools/prefetch_weights.py --keys dinov2_vitg14 dinov2_vitl14 mae_vith14 sup_vith14_in21k clip_vitl14 clip_vitl14_336
```

Pilot one encoder at 256 images per split, then run its cached probe only after
extraction succeeds. `--parsable` returns the job ID used in the dependency;
the optional job-name arguments make queue output immediately identifiable.
Scripts also rename their own running jobs as a fallback.

```bash
PARTITION=<your-partition>
KEY=vith14_224_in22k
export LIMIT=256
EXTRACT_JOB="$(sbatch --parsable --partition="$PARTITION" --job-name="extract-${KEY}" --export=ALL,KEY="${KEY}" tools/slurm/extract_features.sbatch)"
EXTRACT_JOB="${EXTRACT_JOB%%;*}"
PROBE_JOB="$(sbatch --parsable --partition="$PARTITION" --job-name="probe-${KEY}" --dependency="afterok:${EXTRACT_JOB}" --export=ALL,KEY="${KEY}" tools/slurm/probe_cached.sbatch)"
PROBE_JOB="${PROBE_JOB%%;*}"
echo "Pilot jobs: ${EXTRACT_JOB} -> ${PROBE_JOB}"
squeue -u "$USER"
sacct -j "${EXTRACT_JOB},${PROBE_JOB}" --format=JobID,JobName%40,State,Elapsed,MaxRSS
unset LIMIT
```

After checking the pilot manifest, logs, metrics, and actual elapsed time, submit
all full encoders. Each probe has its own `afterok` dependency on exactly one
extraction. Remove `--job-name` if the cluster disallows duplicate names before
the script's `scontrol` rename; job dependency IDs remain unambiguous.

```bash
for KEY in vith14_224_in1k vith16_448_in1k vith14_224_in22k vitg16_224_in22k vith14_224 vitb16_448 vith14_224_randinit dinov2_vitg14 dinov2_vitl14 mae_vith14 sup_vith14_in21k clip_vitl14 clip_vitl14_336; do
  EXTRACT_JOB="$(sbatch --parsable --partition="$PARTITION" --job-name="extract-${KEY}" --export=ALL,KEY="${KEY}" tools/slurm/extract_features.sbatch)"
  EXTRACT_JOB="${EXTRACT_JOB%%;*}"
  PROBE_JOB="$(sbatch --parsable --partition="$PARTITION" --job-name="probe-${KEY}" --dependency="afterok:${EXTRACT_JOB}" --export=ALL,KEY="${KEY}" tools/slurm/probe_cached.sbatch)"
  PROBE_JOB="${PROBE_JOB%%;*}"
  echo "${KEY}: ${EXTRACT_JOB} -> ${PROBE_JOB}"
done
squeue -u "$USER"
sacct -u "$USER" --starttime "$(date +%Y-%m-%d)" --format=JobID,JobName%40,State,Elapsed,MaxRSS
```

Overrides such as `FEATURES_ROOT`, `RUNS_ROOT`, `DEVICE`, `MAX_EPOCHS`, `GRID`,
`SEEDS`, `PYTHON`, and `VENV` are passed through `--export=ALL`. The probe refuses
to run unless all five manifest splits are complete and their arrays exist.

After every encoder completes, export the small results and feature manifests,
inspect the archive, and only then reclaim feature space. This deletion is
intentional and irreversible; the encoder forward passes would be required to
recreate the arrays. Do not delete features while jobs are pending/running.

```bash
bash tools/export_artifacts.sh --root experiment_logs --output-dir .
ARCHIVE="tmlr_artifacts_$(date -u +%Y%m%d).tar.gz"
tar -tzf "${ARCHIVE}" | less
squeue -u "$USER"
# After verifying cached_selection outputs and the archive:
rm -rf -- experiment_logs/features
```

## 2. Run selection, five seeds, and label fractions

On a workstation with enough RAM, or inside one small GPU allocation, run:

```bash
DEVICE=cuda bash tools/run_cached_protocol.sh \
  vith14_224_in22k \
  experiment_logs/features/vith14_224_in22k \
  experiment_logs/cached-runs
```

Repeat with the same feature/output naming for every other key. Use
`DEVICE=cpu` for CPU probing; one GPU is recommended for the full grid. An
allocation can wrap the command directly, for example
`srun --gpus 1 --cpus-per-task 8 ...`; the driver itself does not submit jobs.
It performs exactly:

1. all **160 configurations at seed 0** from `configs/grids/cached/default.yaml`;
2. the OOD-val winner at seeds **0, 1, 2, 3, 4**;
3. that winner at label fractions **0.01, 0.1, 0.5**, each at seeds 0–4;
4. final validation selection and paper-table aggregation.

Completed probe runs are skipped. `MAX_EPOCHS`, `GRID`, `SEEDS`, `PYTHON`, and
`DEVICE` environment variables override defaults for testing. The production
settings are AdamW, cosine LR to zero, cross-entropy, true mini-batches (no
accumulation), patience 10, min-delta `1e-4`, and at most 100 epochs. Early
stopping and checkpoint restoration use **OOD-val macro-F1 only**.

### Probing I/O and timing

Each `probe_cached.py` stage is one Python process covering all of that stage's
configurations and seeds. On the first access for a split/representation, the
process sequentially reads each `.npy` once and moves features/labels to the
selected device as float16/long tensors. Subsequent runs reuse those resident
tensors; there is no mmap/DataLoader row access on the shared filesystem. Only
selected mini-batches are cast to float32. Training uses run-seeded
`torch.randperm`; evaluation uses sequential slices. Stratified label fractions
are applied to resident indices. `drop_last=False` remains unchanged, so the
final true mini-batch is used even when smaller.

Runs with an existing `summary.json` are skipped after a timeout/resubmission.
Progress lines show `[current/total]`; `epochs.csv` adds `epoch_time_seconds`,
and `summary.json` retains per-run wall/train timings. The sbatch commands and
environment variables are unchanged.

The underlying individual commands are:

```bash
python3 tools/probe_cached.py --features experiment_logs/features/vith14_224_in22k --grid configs/grids/cached/default.yaml --seeds 0 --stage grid --device cuda --out experiment_logs/cached-runs
python3 tools/select_cached.py --runs experiment_logs/cached-runs/vith14_224_in22k --out experiment_logs/cached-runs/vith14_224_in22k/selection
```

## Outputs and paper tables

For encoder `KEY`, run directories have this layout:

```text
experiment_logs/cached-runs/KEY/{grid,seeds,label_fraction_F}/seedS/CONFIG/
  summary.json
  epochs.csv
  test_predictions.csv
  id_test_predictions.csv
experiment_logs/cached-runs/KEY/selection/
  cached_selection.json
  cached_selection.csv
  cached_selection.tex
```

- **Main ID/OOD table:** selected five-seed mean/sample-std macro-F1 and average
  accuracy, gap, and per-seed retention in `cached_selection.*`.
- **Label-efficiency table:** selected-config OOD macro-F1 for 1%, 10%, and 50%
  labels in `cached_selection.json` under `label_efficiency` (100% is the main
  five-seed result).
- **HP audit:** every seed-0 val rank and the divergent OOD-test choice are in
  the JSON/CSV. Test choice is diagnostic only.
- **Per-class/frequency analysis:** `tools/per_class_f1.py --run-dir <run-dir>`
  accepts the cached prediction filenames.
- **Feature collapse:** `python3 tools/feature_rank.py --features
  experiment_logs/features/KEY/train_last_avgpool.npy --output rank.json`.
- **Validation curves:** `python3 tools/plot_val_curve.py --log <run-dir>/epochs.csv
  --output val-f1.csv`.

Each summary contains all available split metrics (including train, OOD val, ID
val, ID test, and OOD test), the best epoch, timings, seed, label fraction, and
complete head configuration. Test metrics are computed only after restoring the
best OOD-val-F1 head.

## Disk estimates

Source-split sizes below are exact for 129,809 images and exclude small NumPy
headers, labels, and metadata. `last4` is four times `avg`; CLS adds the same
size as `avg` for external encoders. For all five splits, use
`sum_split(N_split × bytes_per_image)`; the final column is only a planning
estimate assuming non-train splits add about 50% to Source storage.

| Encoder(s) | Width / cached sets | Source avg | Source last4 | Source all reps | All splits (est.) |
|---|---|---:|---:|---:|---:|
| ViT-H/14 I-JEPA + random | 1280; avg+last4 | 0.332 GB | 1.329 GB | 1.662 GB | ≈2.5 GB |
| ViT-H/16@448 I-JEPA | 1280; avg+last4 | 0.332 GB | 1.329 GB | 1.662 GB | ≈2.5 GB |
| ViT-g/16 I-JEPA | 1408; avg+last4 | 0.366 GB | 1.462 GB | 1.828 GB | ≈2.75 GB |
| ViT-B/16@448 I-JEPA | 768; avg+last4 | 0.199 GB | 0.798 GB | 0.997 GB | ≈1.5 GB |
| DINOv2 ViT-g/14 | 1536; avg+last4+CLS | 0.399 GB | 1.595 GB | 2.393 GB | ≈3.6 GB |
| DINOv2 ViT-L/14 | 1024; avg+last4+CLS | 0.266 GB | 1.063 GB | 1.595 GB | ≈2.4 GB |
| MAE / supervised ViT-H/14 | 1280; avg+last4+CLS | 0.332 GB | 1.329 GB | 1.994 GB | ≈3.0 GB |
| CLIP ViT-L/14 @224/@336 | 1024; avg+last4+CLS | 0.266 GB | 1.063 GB | 1.595 GB | ≈2.4 GB |

## GPU-time estimates

No full extraction was run locally. The Source estimate scales the measured
L40S reference: one ViT-H/14@224 probe epoch over 129,809 Source images took
about 10 minutes, and ViT-H/16@448 took about 60 minutes. `All splits` adds a
rough 50% for validation/test images. These are planning ranges, not
measurements; I/O and GPU model can dominate.

| Encoder(s) | Source extraction | All splits (est.) |
|---|---:|---:|
| ViT-H/14 I-JEPA (IN-1K/22K/iWildCam/random), MAE, supervised ViT-H | ≈10 min | ≈15 min |
| ViT-g/16@224 I-JEPA | ≈12 min | ≈18 min |
| ViT-B/16@448 I-JEPA | ≈20 min | ≈30 min |
| ViT-H/16@448 I-JEPA | ≈60 min | ≈90 min |
| DINOv2 ViT-g/14 | ≈20–30 min | ≈30–45 min |
| DINOv2 ViT-L/14, CLIP ViT-L/14@224 | ≈6–8 min | ≈9–12 min |
| CLIP ViT-L/14@336 | ≈15–20 min | ≈22–30 min |

Feature extraction dominates encoder setup but happens once. The 180 cached
runs per encoder (160 + 5 + 15) then avoid encoder forwards. Expect minutes,
rather than hours, per head on one GPU; exact time depends strongly on feature
width, early stopping, and mmap/I/O. Benchmark a pilot before reserving CPU
time for the full grid. Synthetic CPU protocol tests complete in seconds. As a
specific I/O/head benchmark, one synthetic zero-feature epoch with 130,000 x
5120 float16 inputs, 182 classes, batch 1024, and six CPU threads took 3.05 s
(3.87 s including setup, tiny validation, and restore). This measures the new
batching path, not real-data accuracy or L40S throughput.
