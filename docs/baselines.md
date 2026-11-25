# Frozen external-encoder baselines

These runs use the **same frozen-encoder linear-probe pipeline** as I-JEPA. Only
the encoder construction and its explicit normalization preset differ. Public
weights are loaded through timm; all encoder parameters have
`requires_grad=False`, the encoder remains in evaluation mode, and only the
selected `linear` or `bn_linear` head is optimized. The probe uses no data
augmentation, AdamW, cosine LR, early stopping on OOD-val loss, and the existing
metrics/checkpoint/prediction writers.

Patch-token averages exclude CLS and register tokens. `last4_avgpool_concat`
applies the final norm to each of the last four block outputs, then averages and
concatenates patch tokens, matching I-JEPA. `cls` is available as an extra
ablation but is not in the main grid. For both CLIP models this is the
**1024-wide pre-projection image tower**, not its 768-wide output projection.

| Key | timm identifier | Resolution | Probe width | Normalization |
|---|---|---:|---:|---|
| `dinov2_vitg14` | `vit_giant_patch14_dinov2.lvd142m` | 224, patch 14 | 1536 | ImageNet |
| `dinov2_vitl14` | `vit_large_patch14_dinov2.lvd142m` | 224, patch 14 | 1024 | ImageNet |
| `mae_vith14` | `vit_huge_patch14_224.mae` | 224, patch 14 | 1280 | ImageNet |
| `sup_vith14_in21k` | `vit_huge_patch14_224.orig_in21k` | 224, patch 14 | 1280 | Inception (0.5/0.5) |
| `clip_vitl14` | `vit_large_patch14_clip_224.openai` | 224, patch 14 | 1024 | OpenAI CLIP |
| `clip_vitl14_336` | `vit_large_patch14_clip_336.openai` | 336, patch 14 | 1024 | OpenAI CLIP |

Install the pinned project stack plus `timm>=0.9.16,<2`. On the first run timm
downloads public weights into its normal cache. No GitHub hub repository is
executed. Every external model uses its timm-native normalization: the
supervised IN-21k model uses 0.5/0.5 mean/std, DINOv2 and MAE use ImageNet
statistics, and CLIP uses OpenAI's statistics; see `DECISIONS.md`.

## SLURM launch commands

Run from the repository root. Each grid item is a separate one-GPU submitit job.
Set the partition and per-job wall-clock limit for the cluster (the YAML values
are `gpu1` and 2880 minutes):

```bash
export BASELINE_PARTITION=gpu1
export BASELINE_TIME_MIN=2880
```

### Reduced HP search (48 jobs total)

Every model has exactly 8 runs at seed 0: LR `{1e-3, 1e-2}` x head
`{linear, bn_linear}` x representation `{last_avgpool, last4_avgpool_concat}`.
Weight decay is `5e-4`; micro-batch 16 x 32 accumulation steps gives effective
batch 512. After the jobs finish, select using **OOD validation**, never test:

```bash
python3 tools/select_hparams.py --root experiment_logs/eval-wilds --train-root experiment_logs --out experiment_logs/hparam-selection
```

Each baseline run evaluates `val`, so the selector uses complete per-seed
`iwildcam_val_metrics.json` macro-F1 (`val_macro_f1`). If those files are absent,
it can fall back to best-epoch OOD-val loss from retained training CSVs. The
report also shows the test-selected configuration for diagnosis only. The
training CSV contains `val_f1` (OOD val) and `id_val_f1` on each evaluation epoch.

```bash
python3 tools/run_grid.py --grid configs/grids/hp/dinov2_vitg14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/hp/dinov2_vitl14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/hp/mae_vith14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/hp/sup_vith14_in21k.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/hp/clip_vitl14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/hp/clip_vitl14_336.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
```

### Five-seed runs (30 jobs total)

First replace `optimization.lr`, `meta.head_type`, and
`meta.representation_type` in each `configs/grids/seeds/<key>.yaml` with the HP
row selected on OOD validation. Their committed values are merely launchable
placeholders, **not selected optima**. Then run:

```bash
python3 tools/run_grid.py --grid configs/grids/seeds/dinov2_vitg14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/seeds/dinov2_vitl14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/seeds/mae_vith14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/seeds/sup_vith14_in21k.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/seeds/clip_vitl14.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
python3 tools/run_grid.py --grid configs/grids/seeds/clip_vitl14_336.yaml --folder submitit_logs --partition "$BASELINE_PARTITION" --time "$BASELINE_TIME_MIN"
```

Each run automatically evaluates `val`, `id_val`, `id_test`, and `test`.
Expected artifacts under `experiment_logs/eval-wilds/<run-name>-seed<N>/` are:

- `params.yaml`, `params-eval.yaml`, and `best_model.pth.tar`;
- `iwildcam_{val,id_val,id_test,test}_{metrics.json,predictions.csv}`;
- `linear_probe_r0.csv`, including per-epoch `val_f1` and `id_val_f1`.

Aggregate means/sample standard deviations, ID-OOD gap, and per-seed relative
retention `R = F1_OOD / F1_ID` with:

```bash
python3 tools/aggregate_seeds.py --root experiment_logs/eval-wilds --out experiment_logs/seed-runs --min-seeds 5
```

This writes `experiment_logs/seed-runs/<model>/summary.json` and
`experiment_logs/seed-runs/summary_all.csv`. Retention columns are named
`retention_macro_f1_mean` and `retention_macro_f1_std`; each per-seed ratio is
computed before aggregation (it is not the ratio of mean F1 values).

## GPU-hour planning estimates

No baseline training was run on the development machine. These are conservative
**planning estimates**, not measurements: one A100/H100-class GPU per run,
roughly 25 epochs after early stopping, including validation and the four final
evaluation passes. Hardware, image I/O, and stopping time can readily change
them by 0.5–2x. Replace estimates with the actual `train_time_seconds` fields
after a pilot run.

| Encoder | Estimated GPU-hours/run | HP: 8 runs | Seeds: 5 runs |
|---|---:|---:|---:|
| DINOv2 ViT-g/14 | 16 | 128 | 80 |
| DINOv2 ViT-L/14 | 8 | 64 | 40 |
| MAE ViT-H/14 | 10 | 80 | 50 |
| Supervised IN-21k ViT-H/14 | 10 | 80 | 50 |
| OpenAI CLIP ViT-L/14 (224) | 8 | 64 | 40 |
| OpenAI CLIP ViT-L/14 (336) | 14 | 112 | 70 |
| **Total** | | **528** | **330** |

The complete HP + five-seed study therefore has a conservative upper budget of **858
GPU-hours**. Actual aggregate time is available in `summary_all.csv`.

**Measured reference (I-JEPA probes, thesis runs, one NVIDIA L40S, micro-batch 16):** all
runs early-stopped after 14 epochs; ViT-H/14 @224 took 2.35–2.38 h per run, ViT-g/16 @224
2.74 h, ViT-H/16 @448 13.97 h. Scaling by these numbers, a realistic budget is
≈ 2.5–3 h/run for the 224 px baselines and ≈ 6–8 h/run for CLIP ViT-L/14 @336, i.e. roughly
**200–250 GPU-hours** for the full study (13 runs × 6 encoders). Run one pilot per encoder first.

## Analysis commands

Per-class F1 and Source-frequency terciles for one run (WILDS labels are read
without loading images; class 0 is `empty`):

```bash
python3 tools/per_class_f1.py --run-dir experiment_logs/eval-wilds/RUN-seed0 --root-path ./wilds_data --output-prefix experiment_logs/analysis/RUN-seed0
```

Outputs are `RUN-seed0_per_class.csv`, `RUN-seed0_summary.csv`, and a
LaTeX/booktabs `RUN-seed0.tex`. Tail/mid/head are count-sorted terciles over
non-empty Source classes. Add `--download` only when the WILDS metadata is not
already local. Tests/offline analysis can instead pass `--labels-npz` containing
`train`, `test`, and `id_test` arrays.

Effective rank and top-k centered PCA variance (defaults: 5,000 images and
k=1,5,10,50,100):

```bash
python3 tools/feature_rank.py --checkpoint experiment_logs/eval-wilds/RUN-seed0/best_model.pth.tar --config configs/supervised_dinov2_vitl14.yaml --root-path ./wilds_data --split val --output experiment_logs/analysis/dinov2_vitl14-rank.json --save-features experiment_logs/analysis/dinov2_vitl14-features.npy
```

Use the corresponding existing I-JEPA supervised config/checkpoint to compare
ImageNet and iWildCam-pretrained I-JEPA representations. A saved matrix can be
reanalyzed without WILDS or torch model imports via
`python3 tools/feature_rank.py --features FEATURES.npy --output rank.json`.

Extract pgfplots-compatible validation curves after a run:

```bash
python3 tools/plot_val_curve.py --log experiment_logs/eval-wilds/RUN-seed0/linear_probe_r0.csv --output experiment_logs/analysis/RUN-seed0-val-f1.csv
```

Use `table[col sep=comma, x=epoch, y=val_f1]` for OOD val or `y=id_val_f1` for
ID val. Repeated headers after submitit resumes are handled.

## Weight/code licenses

Check the upstream terms before redistribution; this is a summary, not legal
advice.

- **DINOv2** encoders/weights: Apache-2.0.
- **MAE** code and pretrained weights: CC BY-NC 4.0.
- **timm** and the supervised original ViT IN-21k weights: Apache-2.0.
- **OpenAI CLIP**: MIT.
- **I-JEPA** comparison checkpoints/code: CC BY-NC 4.0.

The dataset has its own license and is not covered by these model licenses.
