import copy
import os
import subprocess
import sys

import pytest
import yaml

from src.models.external_encoders import EXTERNAL_ENCODERS, validate_external_encoder_config
from src.transforms import NORMALIZATION_STATS, resolve_normalization
from tools.run_grid import _deep_update, _expand_grid


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_yaml(relative_path):
    with open(os.path.join(ROOT, relative_path), "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def validate_base(config):
    meta = config["meta"]
    validate_external_encoder_config(
        meta["model_name"], meta, config["data"], config["mask"]
    )
    assert config["optimization"]["freeze_weights"] is True
    assert config["data"]["batch_size"] * config["optimization"]["gradient_accumulation_steps"] == 512
    assert config["optimization"]["weight_decay"] == pytest.approx(5.0e-4)
    assert config["validation"]["log_per_epoch_f1"] is True
    for key in (
        "use_random_resized_crop",
        "use_horizontal_flip",
        "use_color_distortion",
        "use_gaussian_blur",
    ):
        assert config["data"][key] is False


@pytest.mark.parametrize("key", sorted(EXTERNAL_ENCODERS))
def test_base_config_matches_verified_architecture_and_protocol(key):
    config = load_yaml(f"configs/supervised_{key}.yaml")
    validate_base(config)
    spec = EXTERNAL_ENCODERS[key]
    assert config["meta"]["embed_dim"] == spec.embed_dim
    assert config["meta"]["pretrained"] is True
    assert config["meta"]["load_checkpoint"] is False
    assert config["data"]["crop_size"] == spec.crop_size
    assert config["data"]["normalization"] == spec.normalization
    assert config["mask"]["patch_size"] == spec.patch_size


@pytest.mark.parametrize("key", sorted(EXTERNAL_ENCODERS))
def test_reduced_hp_grid_expands_to_exactly_eight_valid_runs(key):
    grid = load_yaml(f"configs/grids/hp/{key}.yaml")
    base = load_yaml(grid["base_config"])
    expanded = list(_expand_grid(grid["grid"]))
    assert len(expanded) == 8
    assert len({tuple(sorted(row.items())) for row in expanded}) == 8
    for overrides in expanded:
        config = _deep_update(copy.deepcopy(base), grid.get("constants", {}))
        _deep_update(config, overrides)
        validate_base(config)
        assert config["meta"]["seed"] == 0
        assert config["optimization"]["lr"] in (0.001, 0.01)
        assert config["meta"]["head_type"] in ("linear", "bn_linear")
        assert config["meta"]["representation_type"] in (
            "last_avgpool",
            "last4_avgpool_concat",
        )


@pytest.mark.parametrize("key", sorted(EXTERNAL_ENCODERS))
def test_seed_grid_has_five_launchable_placeholder_runs(key):
    grid = load_yaml(f"configs/grids/seeds/{key}.yaml")
    assert grid["grid"] == {"meta.seed": [0, 1, 2, 3, 4]}
    base = load_yaml(grid["base_config"])
    for overrides in _expand_grid(grid["grid"]):
        config = _deep_update(copy.deepcopy(base), grid["constants"])
        _deep_update(config, overrides)
        validate_base(config)


def test_every_existing_yaml_still_uses_full_loader():
    for directory, _, filenames in os.walk(os.path.join(ROOT, "configs")):
        for filename in filenames:
            if filename.endswith((".yaml", ".yml")):
                with open(os.path.join(directory, filename), "r") as handle:
                    assert isinstance(yaml.load(handle, Loader=yaml.FullLoader), dict)


def test_legacy_normalization_default_is_identical():
    assert resolve_normalization() == ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert resolve_normalization("clip") == NORMALIZATION_STATS["clip"]
    with pytest.raises(ValueError, match="normalization"):
        resolve_normalization("unknown")


def test_training_entrypoint_help_instantiates_imports_without_data_or_cuda():
    result = subprocess.run(
        [sys.executable, "main_distributed_supervised.py", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--fname" in result.stdout
