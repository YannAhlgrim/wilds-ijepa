# WILDS-IJEPA

Fork of the official I-JEPA repo, adapted for WILDS-iWildCam.

Reference: official I-JEPA README https://github.com/facebookresearch/ijepa/blob/main/README.md

- SSL pretraining on WILDS-iWildCam unlabeled dataset: https://arxiv.org/abs/2112.05090 (Extending the WILDS Benchmark for Unsupervised Adaptation)
- Supervised training on WILDS-iWildCam labeled dataset: https://arxiv.org/abs/2012.07421 (WILDS: A Benchmark of in-the-Wild Distribution Shifts)
- Supervised learning supports full fine-tuning or freezing the encoder

<!-- Optional: add a pipeline figure (SSL -> SL) here -->

## Models

- ViT-H, 14x14 patches, 224x224 resolution (trained)
- ViT-H, 16x16 patches, 448x448 resolution (trained)
- Plan: add a graph comparing models with the WILDS leaderboard https://wilds.stanford.edu/leaderboard/#with-unlabeled-data-1

<!-- Optional: add a WILDS leaderboard comparison graph here -->

## Repo layout

- `src/`: core model, masks, and training utilities
- `src/train.py`: SSL training loop
- `src/train_supervised.py`: supervised training loop
- `configs/`: training configs
- `configs/wilds_vith14_ep300.yaml`: SSL config used here
- `configs/supervised_vith14_224.yaml`: supervised config used here (see `configs/` for all supervised linear-probe configs)
- `main_distributed.py`: entrypoint for distributed SSL training
- `main_distributed_supervised.py`: entrypoint for distributed supervised training
- `configs/grids/seeds/`: per-model seed grids for multi-seed paper runs
- `tools/run_seed_sweep.sh`: launch each model across all seeds (one by one)
- `tools/aggregate_seeds.py`: aggregate seed runs into mean +/- std (ID + OOD)
- `requirements.txt`: dependencies

<!-- Optional: add a sample iWildCam image grid here -->

## Requirements

- Python 3.8+ (compatible and newer)
- PyTorch (CUDA 12.1 wheel index): https://download.pytorch.org/whl/cu121
- Key deps: torchvision, submitit, wilds, PyYAML, numpy
- Full list: `requirements.txt`

## SLURM commands

SSL pretraining:

```
python3 main_distributed.py --fname configs/wilds_vith14_ep300.yaml --folder $submitit_folder --partition $slurm_partition --nodes $nodes --tasks-per-node $tasks_per_node --time $time
```

Supervised fine-tuning:

```
python3 main_distributed_supervised.py --fname configs/supervised_vith14_224.yaml --folder $submitit_folder --partition $slurm_partition --nodes $nodes --tasks-per-node $tasks_per_node --time $time
```

Evaluation on iWildCam test split:

```
python3 main_eval_wilds.py --fname configs/eval_wilds_vith14.yaml --folder $submitit_folder --partition $slurm_partition --nodes $nodes --tasks-per-node $tasks_per_node --time $time
```

Evaluation metrics are written to `experiment_logs/eval-wilds-vith14/iwildcam_test_metrics.json` by default.

Variable hints: set `$submitit_folder`, `$slurm_partition`, `$nodes`, `$tasks_per_node`, and `$time` to match your SLURM cluster.

## Multi-seed runs (paper results)

To report mean +/- std over seeds, each supervised model is trained across 5
seeds (0-4). Seeding is config-driven via `meta.seed` (applied in
`src/train_supervised.py`), and the run folder name includes `-seed{N}` so seeds
do not collide.

Each run automatically:
- evaluates on **both** WILDS splits: `id_test` (ID) and `test` (OOD), so the
  generalization gap can be measured;
- records the WILDS metrics, the wall-clock **training time**, and the number of
  **epochs run** (accounting for early stopping) into the per-split metrics JSON
  and into `params.yaml` in the eval folder.

The four leaderboard columns are: Test ID Macro F1, Test ID Avg Acc,
Test OOD Macro F1, Test OOD Avg Acc (headline metric: `F1-macro_all`).

Launch all models, one at a time, each across all seeds (SLURM/submitit):

```
bash tools/run_seed_sweep.sh --partition $slurm_partition --time $time
```

Run a subset of models:

```
bash tools/run_seed_sweep.sh --partition $slurm_partition --models "vith14_224 vith16_448"
```

Per-model seed grids live in `configs/grids/seeds/` (each sets
`meta.seed: [0, 1, 2, 3, 4]` over the corresponding `configs/supervised_*.yaml`
base config). They are launched via `tools/run_grid.py`.

Aggregate mean +/- std across seeds after the jobs finish:

```
python3 tools/aggregate_seeds.py --root experiment_logs/eval-wilds
```

Outputs:
- `experiment_logs/seed-runs/<model>/summary.json` (per-seed rows + mean/std for
  all metrics, training time, epochs, and ID-OOD generalization gap)
- `experiment_logs/seed-runs/summary_all.csv` (one row per model, paper-ready)

## License

See the `LICENSE` file for details about the license under which this code is made available.

## Citation

To be defined.
