from logging import getLogger

import numpy as np
import torch
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

logger = getLogger()


def _stratified_subset_indices(labels, fraction, seed):
    """Return a deterministic, seed-dependent stratified subset of indices.

    For each class, keep ceil(count * fraction) samples, guaranteeing every
    class is represented even for very small fractions.

    Parameters
    ----------
    labels : array-like of int
        Per-sample class labels.
    fraction : float
        Fraction of samples to keep (0 < fraction <= 1).
    seed : int
        Seed for NumPy's RNG; different seeds give different subsets.

    Returns
    -------
    indices : np.ndarray
        Sorted array of selected global indices.
    """
    if not (0 < fraction <= 1.0):
        raise ValueError(f"label_fraction must be in (0, 1], got {fraction}")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)

    selected = []
    for cls in classes:
        cls_idx = np.nonzero(labels == cls)[0]
        n_keep = max(1, int(np.ceil(len(cls_idx) * fraction)))
        n_keep = min(n_keep, len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=n_keep, replace=False).tolist())

    indices = np.array(sorted(selected), dtype=np.int64)
    return indices


def make_iwildcam(
    transform,
    batch_size,
    collator=None,
    split="extra_unlabeled",
    num_workers=8,
    world_size=1,
    rank=0,
    root_path="./wilds_data",
    download=True,
    pin_mem=True,
    drop_last=True,
    label_fraction=None,
    seed=0,
):
    unlabeled = True if split == "extra_unlabeled" else False
    shuffle = True if split == "extra_unlabeled" or split == "train" else False

    full_dataset = get_dataset(
        dataset="iwildcam", download=download, root_dir=root_path, unlabeled=unlabeled
    )

    dataset = full_dataset.get_subset(split, transform=transform)

    # Subset the labeled training data for label-efficiency experiments.
    if label_fraction is not None and split == "train":
        labels = dataset.y_array
        indices = _stratified_subset_indices(
            labels=labels, fraction=float(label_fraction), seed=int(seed)
        )
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(
            f"iWildCam {split} subset created with {len(dataset)} samples "
            f"(label_fraction={label_fraction}, seed={seed})"
        )

    # Use the unified wrapper that mimics the ImageNet structure
    dataset = WildsToTorchWrapper(dataset, is_unlabeled=unlabeled)

    logger.info(f"iWildCam {split} dataset created with {len(dataset)} samples")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    return dataset, data_loader, dist_sampler


def make_iwildcam_eval(
    transform,
    batch_size,
    split="test",
    num_workers=8,
    root_path="./wilds_data",
    download=True,
    pin_mem=True,
):
    full_dataset = get_dataset(
        dataset="iwildcam", download=download, root_dir=root_path
    )

    dataset = full_dataset.get_subset(split, transform=transform)

    logger.info(f"iWildCam {split} eval dataset created with {len(dataset)} samples")

    data_loader = get_eval_loader(
        "standard",
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return full_dataset, dataset, data_loader


class WildsToTorchWrapper(torch.utils.data.Dataset):
    """
    Mimics the ImageNet wrapper by always returning (image, target).
    """

    def __init__(self, wilds_subset, is_unlabeled=False):
        self.dataset = wilds_subset
        self.is_unlabeled = is_unlabeled

    def __getitem__(self, i):
        if self.is_unlabeled:
            # metadata is at index 1 for unlabeled WILDS
            x, _ = self.dataset[i]
            target = -1  # Dummy target to match (img, target) signature
        else:
            # image, target, metadata
            x, y, _ = self.dataset[i]
            target = y

        return x, target

    def __len__(self):
        return len(self.dataset)
