# Run outputs and reproducibility details

## Run directories and metrics

Supervised training starts in `experiment_logs/<run-name>/` unless
`logging.folder` overrides it. Evaluation runs are written to
`experiment_logs/eval-wilds/<run-name>-seed<N>/`. The run name contains model,
resolution, batch size, representation, head, optimizer, LR, weight decay,
schedule, and seed; `data.label_fraction` adds an `-lf...` component.

Each supervised run evaluates `val`, `id_val`, `id_test`, and `test`. It writes
`iwildcam_<split>_metrics.json`, `iwildcam_<split>_predictions.csv`, parameter
YAML files, and `best_model.pth.tar`. New baseline runs also preserve
`linear_probe_r0.csv` beside the evaluation outputs. Older I-JEPA runs may only
have evaluation outputs because the original training directory is removed after
the best checkpoint is copied.

Metrics JSON files include `run_info` with the seed, epochs run, early-stopping
state, and wall-clock training time. Resource fields are:

- `peak_host_ram_gb`: process RSS high-water mark from
  `resource.getrusage(RUSAGE_SELF).ru_maxrss`; with one task per node this covers
  the training worker.
- `peak_gpu_alloc_gb` / `peak_gpu_reserved_gb`: peak allocated/reserved CUDA
  memory from `torch.cuda.max_memory_allocated` and `max_memory_reserved`.

`aggregate_seeds.py` reports per-seed and mean/sample-std metrics, timing, memory,
ID-OOD gap, and relative retention. `aggregate_label_efficiency.py` writes the
label-fraction summary under `experiment_logs/label-efficiency/`.

## Pretraining logs

`src/train.py` writes `<write_tag>_r<rank>.csv` directly under its configured
`logging.folder`. For `configs/wilds_vith14_ep300.yaml`, this is normally
`experiment_logs/vith14.224-bs.128-ep.300/jepa_r0.csv`. Columns include `epoch`,
`itr`, Smooth-L1 `loss`, mask sizes, and iteration time. These logs are included
by `tools/export_artifacts.sh` when present.

## WILDS leaderboard package

After the five-seed runs finish:

```bash
python3 tools/generate_wilds_submission.py \
  --submission-name vith14_224_in22k \
  --eval-root experiment_logs/eval-wilds \
  --out-dir .
```

Artifacts are `vith14_224_in22k_wilds_submission.tar.gz` for upload,
`vith14_224_in22k_submission/f1_macro.txt` for verification, and
`vith14_224_in22k_seed{0..4}.pth.tar` checkpoints. This is a non-standard
leaderboard submission because the encoder was pretrained on ImageNet-22K.

## Small artifact archive

```bash
bash tools/export_artifacts.sh --root experiment_logs --output-dir .
```

The archive contains only `*_metrics.json`, `*_predictions.csv`, supervised and
pretraining `*_r<rank>.csv` logs, `params*.yaml`, and `summary*.json/csv` under
`experiment_logs/`. It excludes all `*.pth*`, images, and `wilds_data`. The
filename is `tmlr_artifacts_<UTC YYYYMMDD>.tar.gz`; an existing same-day archive
is never overwritten.
