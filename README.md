# WILDS-IJEPA

Fork of the official I-JEPA repo, adapted for WILDS-iWildCam.

Reference: official I-JEPA README https://github.com/facebookresearch/ijepa/blob/main/README.md

- SSL pretraining on WILDS-iWildCam unlabeled dataset: https://arxiv.org/abs/2112.05090 (Extending the WILDS Benchmark for Unsupervised Adaptation)
- Supervised training on WILDS-iWildCam labeled dataset: https://arxiv.org/abs/2012.07421 (WILDS: A Benchmark of in-the-Wild Distribution Shifts)
- Supervised learning supports full fine-tuning or freezing the encoder

<!-- Optional: add a pipeline figure (SSL -> SL) here -->

## Models

- ViT-H, 14x14 patches, 224x224 resolution (trained)
- ViT-H, 16x16 patches, 448x448 resolution (planned)
- Plan: add a graph comparing models with the WILDS leaderboard https://wilds.stanford.edu/leaderboard/#with-unlabeled-data-1

<!-- Optional: add a WILDS leaderboard comparison graph here -->

## Repo layout

- `src/`: core model, masks, and training utilities
- `src/train.py`: SSL training loop
- `src/train_supervised.py`: supervised training loop
- `configs/`: training configs
- `configs/wilds_vith14_ep300.yaml`: SSL config used here
- `configs/supervised_wilds_vith14_ep300.yaml`: supervised config used here
- `main_distributed.py`: entrypoint for distributed SSL training
- `main_distributed_supervised.py`: entrypoint for distributed supervised training
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
python3 main_distributed_supervised.py --fname configs/supervised_wilds_vith14_ep300.yaml --folder $submitit_folder --partition $slurm_partition --nodes $nodes --tasks-per-node $tasks_per_node --time $time
```

Variable hints: set `$submitit_folder`, `$slurm_partition`, `$nodes`, `$tasks_per_node`, and `$time` to match your SLURM cluster.

## License

See the `LICENSE` file for details about the license under which this code is made available.

## Citation

To be defined.
