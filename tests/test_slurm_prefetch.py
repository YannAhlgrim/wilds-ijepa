import os
import subprocess
import sys

import pytest

from tools import prefetch_weights


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


SLURM_SCRIPTS = (
    "tools/slurm/extract_features.sbatch",
    "tools/slurm/probe_cached.sbatch",
)


def test_slurm_scripts_have_required_directives_and_syntax():
    for filename in SLURM_SCRIPTS:
        path = os.path.join(ROOT, filename)
        text = open(path).read()
        assert "set -euo pipefail" in text
        assert "#SBATCH --nodes=1" in text
        assert "#SBATCH --ntasks=1" in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --cpus-per-task=8" in text
        assert "#SBATCH --mem=64G" in text
        assert "#SBATCH --output=slurm_logs/%x-%j.out" in text
        assert 'cd "${SLURM_SUBMIT_DIR' in text
        assert "nvidia-smi -L" in text
        assert 'JobName="' in text and '${KEY}' in text
        assert "#SBATCH --partition" not in text
        subprocess.run(["bash", "-n", path], check=True)
    assert "#SBATCH --time=03:00:00" in open(os.path.join(ROOT, SLURM_SCRIPTS[0])).read()
    assert "#SBATCH --time=12:00:00" in open(os.path.join(ROOT, SLURM_SCRIPTS[1])).read()


def test_slurm_scripts_check_key_and_refuse_incomplete_manifest(tmp_path):
    env = dict(os.environ)
    env.pop("KEY", None)
    env.pop("VENV", None)
    env["SLURM_SUBMIT_DIR"] = ROOT
    missing_key = subprocess.run(
        ["bash", os.path.join(ROOT, SLURM_SCRIPTS[0])], env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert missing_key.returncode != 0
    assert "Set KEY" in missing_key.stderr

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia = fake_bin / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\necho 'Fake GPU for local validation'\n")
    nvidia.chmod(0o755)
    env.update(KEY="missing", FEATURES_ROOT=str(tmp_path / "features"), PYTHON=sys.executable)
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    incomplete = subprocess.run(
        ["bash", os.path.join(ROOT, SLURM_SCRIPTS[1])], env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert incomplete.returncode != 0
    assert "Incomplete extraction" in incomplete.stderr
    assert "missing" in incomplete.stderr


def test_prefetch_uses_external_registry_and_prints_cache_sizes(tmp_path, monkeypatch, capsys):
    downloaded = tmp_path / "weights.safetensors"
    downloaded.write_bytes(b"synthetic")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    result = prefetch_weights.prefetch_key("clip_vitl14", downloader=fake_download)
    assert result["key"] == "clip_vitl14"
    assert result["filename"] == "model.safetensors"
    assert result["path"] == str(downloaded)
    assert result["bytes"] == 9
    assert calls[0]["repo_id"] == "timm/vit_large_patch14_clip_224.openai"
    with pytest.raises(ValueError, match="Unknown external encoder"):
        prefetch_weights.prefetch_key("vith14_224", downloader=fake_download)
    assert len(calls) == 1

    monkeypatch.setattr(prefetch_weights, "prefetch_key", lambda _key: result)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    returned = prefetch_weights.main(["--keys", "clip_vitl14"])
    assert returned == [result]
    output = capsys.readouterr().out
    assert str(tmp_path / "hf") in output
    assert str(tmp_path / "torch") in output
    assert "9 bytes" in output
