import hashlib
import json
import os

import numpy as np
import pytest
import torch
import yaml

from src.models.external_encoders import EXTERNAL_ENCODERS
from src.transforms import NORMALIZATION_STATS, resolve_normalization
from tools.extract_features import extract_features


@pytest.mark.parametrize("key", sorted(EXTERNAL_ENCODERS))
def test_external_normalization_matches_timm_pretrained_cfg(key):
    timm = pytest.importorskip("timm")
    spec = EXTERNAL_ENCODERS[key]
    cfg = timm.get_pretrained_cfg(spec.timm_name)
    mean, std = resolve_normalization(spec.normalization)
    assert np.allclose(mean, cfg.mean)
    assert np.allclose(std, cfg.std)
    assert spec.normalization in NORMALIZATION_STATS


def test_supervised_in21k_config_uses_native_half_stats():
    with open("configs/supervised_sup_vith14_in21k.yaml") as handle:
        config = yaml.safe_load(handle)
    assert config["data"]["normalization"] == "inception"
    assert resolve_normalization(config["data"]["normalization"]) == (
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )


def test_changed_normalization_config_hash_rejects_old_feature_cache(tmp_path):
    source = "configs/supervised_sup_vith14_in21k.yaml"
    current_text = open(source).read()
    old_text = current_text.replace("normalization: inception", "normalization: imagenet")
    old_hash = hashlib.sha256(old_text.encode()).hexdigest()
    current_hash = hashlib.sha256(current_text.encode()).hexdigest()
    assert old_hash != current_hash

    config_path = tmp_path / os.path.basename(source)
    config_path.write_text(current_text)
    features = tmp_path / "features"
    features.mkdir()
    (features / "manifest.json").write_text(
        json.dumps({"signature": {"config_sha256": old_hash}})
    )
    with pytest.raises(ValueError, match="different extraction configuration"):
        extract_features(
            str(config_path), str(features), splits=["train"],
            encoder=torch.nn.Identity(),
            dataset_factory=lambda *_args: (_ for _ in ()).throw(
                AssertionError("Dataset must not be touched before signature rejection")
            ),
        )
