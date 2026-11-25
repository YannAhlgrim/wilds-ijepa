import json

import numpy as np
import pytest
import torch
import yaml
from wilds.common.metrics.all_metrics import Accuracy, F1
from wilds.datasets.iwildcam_dataset import IWildCamDataset

from src.cached.features import representations_for_batch
from src.cached.metrics import evaluate_iwildcam
from src.models.head import ViTClassifier
from src.models.vision_transformer import vit_tiny
from tools.extract_features import extract_features


class SyntheticVisionDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        metadata = torch.arange(9, dtype=torch.long) + index
        metadata[-1] = self.labels[index]
        return self.images[index], int(self.labels[index]), metadata


def _tiny_config(tmp_path):
    config = {
        "meta": {
            "model_name": "vit_tiny", "embed_dim": 192, "num_classes": 3,
            "load_checkpoint": False, "seed": 11, "use_bfloat16": False,
        },
        "data": {"crop_size": 32, "normalization": "imagenet", "root_path": "unused"},
        "mask": {"patch_size": 16},
    }
    path = tmp_path / "supervised_tiny.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_extraction_shapes_resume_and_representation_equivalence(tmp_path):
    torch.manual_seed(7)
    encoder = vit_tiny(img_size=[32], patch_size=16, use_checkpoint=False).eval()
    images = torch.randn(5, 3, 32, 32)
    labels = [0, 1, 2, 1, 0]
    dataset = SyntheticVisionDataset(images, labels)
    out = tmp_path / "features"

    class Interrupted(RuntimeError):
        pass

    def interrupt_after_first_chunk(next_index):
        if next_index == 2:
            raise Interrupted()

    with pytest.raises(Interrupted):
        extract_features(
            str(_tiny_config(tmp_path)), str(out), splits=["train"], batch_size=2,
            num_workers=0, device="cpu", encoder=encoder,
            dataset_factory=lambda _split, _transform: dataset,
            batch_observer=interrupt_after_first_chunk,
        )
    progress = json.loads((out / ".train_progress.json").read_text())
    assert progress["next_index"] == 2
    assert not (out / "train_last_avgpool.npy").exists()

    manifest = extract_features(
        str(_tiny_config(tmp_path)), str(out), splits=["train"], batch_size=2,
        num_workers=0, device="cpu", encoder=encoder,
        dataset_factory=lambda _split, _transform: dataset,
    )
    avg = np.load(out / "train_last_avgpool.npy", mmap_mode="r")
    last4 = np.load(out / "train_last4_avgpool_concat.npy", mmap_mode="r")
    cached_y = np.load(out / "train_y.npy")
    metadata = np.load(out / "train_metadata.npy")
    assert avg.shape == (5, 192) and avg.dtype == np.float16
    assert last4.shape == (5, 768) and last4.dtype == np.float16
    assert np.array_equal(cached_y, labels)
    assert metadata.shape == (5, 9) and metadata.dtype == np.int64
    assert manifest["splits"]["train"]["num_images"] == 5
    assert not (out / ".train_progress.json").exists()

    for representation, cached in (("last_avgpool", avg), ("last4_avgpool_concat", last4)):
        classifier = ViTClassifier(
            encoder, num_classes=3, embed_dim=192, representation_type=representation
        ).eval()
        with torch.no_grad():
            expected = classifier._extract_representation(images).numpy()
        assert np.allclose(cached, expected, atol=0.02, rtol=0.02)

    # A third call finalizes nothing and skips before touching the dataset.
    # A fresh, uninterrupted cache must also finalize without a progress file.
    fresh = tmp_path / "features-fresh"
    fresh_manifest = extract_features(
        str(_tiny_config(tmp_path)), str(fresh), splits=["train"], batch_size=5,
        num_workers=0, device="cpu", encoder=encoder,
        dataset_factory=lambda _split, _transform: dataset,
    )
    assert fresh_manifest["splits"]["train"]["complete"] is True


def test_official_wilds_evaluator_matches_metric_classes_on_synthetic_labels():
    y_true = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 6])
    y_pred = torch.tensor([0, 1, 1, 1, 2, 4, 4, 5, 0])
    metadata = torch.zeros(len(y_true), 9, dtype=torch.long)
    evaluator = object.__new__(IWildCamDataset)
    expected, expected_summary = evaluator.eval(y_pred, y_true, metadata)
    actual, actual_summary = evaluate_iwildcam(y_pred, y_true, metadata)
    assert actual == {key: float(value) for key, value in expected.items()}
    assert actual_summary == expected_summary
    assert actual["acc_avg"] == pytest.approx(float(Accuracy().compute(y_pred, y_true)["acc_avg"]))
    assert actual["F1-macro_all"] == pytest.approx(
        float(F1(average="macro").compute(y_pred, y_true)["F1-macro_all"])
    )
    # Hand calculation: class-wise F1 values average to 17/30.
    assert actual["F1-macro_all"] == pytest.approx(17.0 / 30.0)


def test_external_adapter_one_pass_includes_preprojection_cls():
    timm = pytest.importorskip("timm")
    from src.models.external_encoders import TimmExternalEncoder

    backbone = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=0, img_size=32
    )
    encoder = TimmExternalEncoder(backbone, 192).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        outputs = representations_for_batch(encoder, images)
        expected_cls = encoder.forward_cls(images)
    assert outputs["last_avgpool"].shape == (2, 192)
    assert outputs["last4_avgpool_concat"].shape == (2, 768)
    assert outputs["cls"].shape == (2, 192)
    assert torch.allclose(outputs["cls"], expected_cls)
