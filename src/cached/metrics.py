"""Official WILDS iWildCam evaluation on cached arrays, without image files."""

import numpy as np
import torch


def evaluate_iwildcam(y_pred, y_true, metadata):
    """Call the unmodified official ``IWildCamDataset.eval`` implementation.

    ``IWildCamDataset.eval`` only constructs Accuracy/Recall/F1 metric objects;
    it never reads dataset attributes or images. Constructing the object without
    calling ``__init__`` therefore provides the exact official evaluator for
    arrays loaded from a feature cache. Metadata is shape-checked because the
    real WILDS API requires it, although this dataset's metrics do not use it.
    """
    y_pred = torch.as_tensor(np.asarray(y_pred), dtype=torch.long).reshape(-1)
    y_true = torch.as_tensor(np.asarray(y_true), dtype=torch.long).reshape(-1)
    metadata = torch.as_tensor(np.asarray(metadata))
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction/label shape mismatch: {y_pred.shape} vs {y_true.shape}")
    if metadata.shape[0] != y_true.shape[0]:
        raise ValueError("Metadata length does not match cached labels")
    if y_true.numel() == 0:
        raise ValueError("Cannot evaluate an empty split")

    from wilds.datasets.iwildcam_dataset import IWildCamDataset

    evaluator = object.__new__(IWildCamDataset)
    metrics, summary = evaluator.eval(y_pred, y_true, metadata)
    return {key: float(value) for key, value in metrics.items()}, summary
