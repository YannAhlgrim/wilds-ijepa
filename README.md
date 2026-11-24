# WILDS-IJEPA

This fork of [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) evaluates frozen I-JEPA representations on iWildCam2020-WILDS, including linear probes under camera-location distribution shift, label efficiency, and frozen DINOv2/MAE/supervised-ViT/CLIP baselines. The primary metric is OOD test macro-F1; retention is `R = OOD F1 / ID F1`.

## Results

Mean ± sample standard deviation over five seeds for ImageNet-pretrained I-JEPA checkpoints. Gap is `ID F1 − OOD F1`.

| Model | ID F1 | OOD F1 | Gap | R |
|---|---:|---:|---:|---:|
| ViT-H/14 (224, IN-1K) | 0.338 ± 0.015 | 0.214 ± 0.006 | 0.124 | 0.63 |
| ViT-H/16 (448, IN-1K) | 0.385 ± 0.005 | 0.247 ± 0.003 | 0.138 | 0.64 |
| **ViT-H/14 (224, IN-22K)** | 0.348 ± 0.003 | **0.260 ± 0.003** | 0.088 | 0.75 |
| ViT-g/16 (224, IN-22K) | 0.371 ± 0.002 | 0.255 ± 0.003 | 0.116 | 0.69 |

**Hyperparameter re-selection on OOD validation is pending; these numbers are retained from the existing runs.**

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

## WILDS submission packaging

```bash
python3 tools/generate_wilds_submission.py --submission-name vith14_224_in22k --eval-root experiment_logs/eval-wilds --out-dir .
```
This creates the upload tarball, verification F1 file, and per-seed checkpoints; see [docs/reproducibility.md](docs/reproducibility.md) for run outputs and details.

## Licenses

I-JEPA and MAE code/weights are CC BY-NC 4.0; DINOv2, timm, and supervised original ViT IN-21k weights are Apache-2.0; OpenAI CLIP is MIT. iWildCam/WILDS has separate dataset terms. See `LICENSE` and upstream weight licenses before redistribution.

## Citation

Citation information will be added when the paper is public.
