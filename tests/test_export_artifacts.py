import os
import subprocess
import tarfile


def test_export_artifacts_includes_only_small_result_files(tmp_path):
    root = tmp_path / "experiment_logs"
    run = root / "eval-wilds" / "run-seed0"
    pretrain = root / "vith14.224-bs.128-ep.300"
    run.mkdir(parents=True)
    pretrain.mkdir(parents=True)
    allowed = {
        run / "iwildcam_val_metrics.json": "{}",
        run / "iwildcam_test_predictions.csv": "0\n",
        run / "linear_probe_r0.csv": "epoch,val_loss\n1,0.5\n",
        run / "params.yaml": "meta: {}",
        root / "seed-runs" / "summary_all.csv": "model\nrun\n",
        pretrain / "jepa_r0.csv": "epoch,itr,loss\n1,0,0.5\n",
        root / "cached-runs" / "enc" / "seeds" / "seed0" / "cfg" / "summary.json": "{}",
        root / "cached-runs" / "enc" / "seeds" / "seed0" / "cfg" / "epochs.csv": "epoch,val_f1\n1,0.5\n",
        root / "cached-runs" / "enc" / "seeds" / "seed0" / "cfg" / "test_predictions.csv": "0\n",
        root / "cached-runs" / "enc" / "selection" / "cached_selection.json": "{}",
        root / "cached-runs" / "enc" / "selection" / "cached_selection.csv": "encoder_key\nenc\n",
        root / "cached-runs" / "enc" / "selection" / "cached_selection.tex": "% rows\n",
        root / "cached-runs" / "enc" / "selection" / "enc_selected.yaml": "grid: {}\n",
        root / "features" / "enc" / "manifest.json": "{}",
    }
    (root / "seed-runs").mkdir()
    for path, content in allowed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "features" / "enc" / "train_last_avgpool.npy").write_bytes(b"not exported")
    (run / "best_model.pth.tar").write_bytes(b"checkpoint")
    (run / "image.png").write_bytes(b"image")
    wilds = root / "wilds_data"
    wilds.mkdir()
    (wilds / "params.yaml").write_text("excluded: true")

    output = tmp_path / "archives"
    script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools", "export_artifacts.sh"))
    completed = subprocess.run(
        ["bash", script, "--root", str(root), "--output-dir", str(output)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    archives = list(output.glob("tmlr_artifacts_*.tar.gz"))
    assert len(archives) == 1
    assert "archive size:" in completed.stdout
    with tarfile.open(archives[0], "r:gz") as archive:
        names = archive.getnames()
    assert len(names) == len(allowed)
    assert any(name.endswith("jepa_r0.csv") for name in names)
    assert any(name.endswith("linear_probe_r0.csv") for name in names)
    assert any(name.endswith("features/enc/manifest.json") for name in names)
    assert any(name.endswith("cached_selection.csv") for name in names)
    assert not any(name.endswith((".pth", ".pth.tar", ".png", ".npy")) for name in names)
    assert not any("wilds_data" in name for name in names)
