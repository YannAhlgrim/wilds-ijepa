# WILDS-IJEPA

This fork of [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) evaluates frozen I-JEPA representations on iWildCam2020-WILDS, including linear probes under camera-location distribution shift, label efficiency, and frozen DINOv2/MAE/supervised-ViT/CLIP baselines. The primary metric is OOD test macro-F1; retention is `R = OOD F1 / ID F1`.

## Results

Final cached-protocol results with OOD-validation-selected hyperparameters; mean ± sample standard deviation over five seeds. Retention is `R = OOD F1 / ID F1`.

| Model | OOD macro-F1 | ID macro-F1 | R | Readout |
|---|---:|---:|---:|---|
| I-JEPA ViT-H/14 IN-1K | 0.242 ± 0.010 | 0.387 ± 0.009 | 0.62 | Avg last layer |
| I-JEPA ViT-H/16@448 IN-1K | 0.255 ± 0.011 | 0.420 ± 0.019 | 0.61 | Avg last 4 |
| I-JEPA ViT-H/14 IN-22K | 0.256 ± 0.006 | 0.403 ± 0.014 | 0.63 | Avg last 4 |
| I-JEPA ViT-g/16 IN-22K | 0.278 ± 0.006 | 0.419 ± 0.008 | 0.67 | Avg last 4 |
| I-JEPA ViT-H/14 iWildCam-pretrained | 0.081 ± 0.002 | 0.255 ± 0.013 | 0.32 | — |
| I-JEPA ViT-B/16@448 iWildCam-pretrained | 0.046 ± 0.006 | 0.201 ± 0.012 | 0.23 | — |
| ViT-H/14 random init | 0.047 ± 0.005 | 0.105 ± 0.013 | 0.45 | — |
| DINOv2 ViT-L/14 | 0.463 ± 0.006 | 0.569 ± 0.016 | 0.81 | CLS |
| DINOv2 ViT-g/14 | 0.488 ± 0.004 | 0.600 ± 0.006 | 0.81 | CLS |
| CLIP ViT-L/14@224 | 0.348 ± 0.009 | 0.449 ± 0.020 | 0.78 | CLS |
| CLIP ViT-L/14@336 | 0.368 ± 0.010 | 0.516 ± 0.015 | 0.71 | CLS |
| Supervised IN-21k ViT-H/14 | 0.259 ± 0.006 | 0.378 ± 0.003 | 0.68 | CLS |
| MAE ViT-H/14 | 0.214 ± 0.002 | 0.380 ± 0.010 | 0.56 | CLS |

Older numbers in previous commits (thesis) used test-split-ranked hyperparameters and loss-based early stopping and are superseded.

## Install and data

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Use the CUDA PyTorch wheel appropriate for the cluster. The first run downloads WILDS iWildCam into `./wilds_data`; set `data.root_path` in YAML for an existing copy. Runs use the official `train`, `val`, `id_test`, and `test` splits.

Set `PARTITION` and `TIME_MIN` (minutes) for the following SLURM/submitit commands. Run everything from the repository root.

## Training and evaluation

**(a) I-JEPA pretraining on iWildCam unlabeled data:**
```bash
python3 main_distributed.py --fname configs/wilds_vith14_ep300.yaml --folder submitit_logs --partition "$PARTITION" --nodes 1 --tasks-per-node 4 --time "$TIME_MIN"
```

**(b) One frozen-encoder linear probe:**
```bash
python3 main_distributed_supervised.py --fname configs/supervised_vith14_224_in22k.yaml --folder submitit_logs --partition "$PARTITION" --nodes 1 --tasks-per-node 1 --time "$TIME_MIN"
```

**(c) Five-seed sweep and aggregation:**
```bash
bash tools/run_seed_sweep.sh --partition "$PARTITION" --time "$TIME_MIN"
python3 tools/aggregate_seeds.py --root experiment_logs/eval-wilds --out experiment_logs/seed-runs --min-seeds 5
```

**(d) Label-efficiency sweep:**
```bash
bash tools/run_label_efficiency.sh --partition "$PARTITION" --time "$TIME_MIN"
python3 tools/aggregate_label_efficiency.py --root experiment_logs/eval-wilds
```

**(e) Frozen external baselines:** identifiers, six HP grids, six seed grids, and GPU budgets are in [docs/baselines.md](docs/baselines.md). Example:
```bash
python3 tools/run_grid.py --grid configs/grids/hp/dinov2_vitl14.yaml --partition "$PARTITION" --time "$TIME_MIN"
```

**(f) Select HPs on OOD validation from existing runs (no retraining):**
```bash
python3 tools/select_hparams.py --root experiment_logs/eval-wilds --train-root experiment_logs --out experiment_logs/hparam-selection
```

The selector prefers complete per-seed OOD-val macro-F1, falls back to minimum OOD-val loss with accuracy tiebreaks, and reports test selection only as a diagnostic. `tools/summarize_grid.py` now defaults to `--split val`.

**(g) Analysis tools:** per-class/frequency-bin F1, feature rank, and validation curves:
```bash
python3 tools/per_class_f1.py --run-dir experiment_logs/eval-wilds/RUN-seed0 --root-path wilds_data --output-prefix experiment_logs/analysis/RUN-seed0
python3 tools/feature_rank.py --features experiment_logs/analysis/features.npy --output experiment_logs/analysis/rank.json
python3 tools/plot_val_curve.py --log experiment_logs/eval-wilds/RUN-seed0/linear_probe_r0.csv --output experiment_logs/analysis/RUN-seed0-val-f1.csv
```
See [docs/baselines.md](docs/baselines.md#analysis-commands) for checkpoint-based feature extraction and retention details.

**(h) Export small paper artifacts (no checkpoints or images):**
```bash
bash tools/export_artifacts.sh --root experiment_logs --output-dir .
```

## Cached-feature protocol

For the unified validation-selected protocol across all 13 encoders, extract frozen features once and probe them on CPU or one GPU:

```bash
python3 tools/extract_features.py --config configs/supervised_vith14_224_in22k.yaml --out experiment_logs/features/vith14_224_in22k --dtype float16
DEVICE=cuda bash tools/run_cached_protocol.sh vith14_224_in22k experiment_logs/features/vith14_224_in22k experiment_logs/cached-runs
```

This runs the 160-config seed-0 OOD-val grid, five final seeds, and label fractions. See [docs/cached_protocol.md](docs/cached_protocol.md) for every encoder command, resume behavior, disk/time estimates, and paper outputs.

## WILDS evaluation

The public WILDS leaderboard no longer accepts submissions; predictions can be evaluated locally with WILDS' examples/evaluate.py.

## Licenses

I-JEPA and MAE code/weights are CC BY-NC 4.0; DINOv2, timm, and supervised original ViT IN-21k weights are Apache-2.0; OpenAI CLIP is MIT. iWildCam/WILDS has separate dataset terms. See `LICENSE` and upstream weight licenses before redistribution.

## Citation

Paper under review; citation information will be added later.
