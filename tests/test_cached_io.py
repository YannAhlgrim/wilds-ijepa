import numpy as np
import pytest
import torch

from src.cached.probe import evaluate_head, make_cached_head, train_cached_head
from tools import probe_cached


def test_process_cache_loads_arrays_only_once(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    np.save(tmp_path / "train_last_avgpool.npy", rng.normal(size=(9, 6)).astype(np.float16))
    np.save(tmp_path / "train_y.npy", np.arange(9) % 3)
    np.save(tmp_path / "train_metadata.npy", np.zeros((9, 9), dtype=np.int64))
    calls = []
    real_load = np.load

    def counted_load(*args, **kwargs):
        calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(probe_cached.np, "load", counted_load)
    probe_cached._SPLIT_CACHE.clear()
    first = probe_cached._load_split(str(tmp_path), "train", "last_avgpool", device="cpu")
    second = probe_cached._load_split(str(tmp_path), "train", "last_avgpool", device="cpu")
    assert len(calls) == 3  # features, labels, metadata; all without mmap
    assert all(kwargs.get("mmap_mode") is None for _args, kwargs in calls)
    assert first[0] is second[0] and first[1] is second[1] and first[2] is second[2]
    assert first[0].dtype == torch.float16
    assert first[1].dtype == torch.long
    probe_cached._SPLIT_CACHE.clear()


def test_index_batching_matches_explicit_fixed_seed_reference():
    rng = np.random.default_rng(21)
    train_x = rng.normal(size=(13, 6)).astype(np.float16)
    train_y = np.arange(13) % 3
    val_x = rng.normal(size=(8, 6)).astype(np.float16)
    val_y = np.arange(8) % 3
    id_val_x = rng.normal(size=(7, 6)).astype(np.float16)
    id_val_y = np.arange(7) % 3
    val_metadata = np.zeros((8, 9), dtype=np.int64)
    id_val_metadata = np.zeros((7, 9), dtype=np.int64)
    train_indices = np.array([0, 1, 3, 5, 7, 8, 10, 12])
    seed = 17
    config = {
        "lr": 0.003, "weight_decay": 0.0001, "batch_size": 3,
        "head_type": "linear", "representation_type": "last_avgpool",
        "max_epochs": 1, "patience": 2, "min_delta": 1e-4, "min_epochs": 0,
    }
    actual_head, history, _, _ = train_cached_head(
        train_x, train_y, val_x, val_y, val_metadata,
        id_val_x, id_val_y, id_val_metadata,
        num_classes=3, config=config, seed=seed, device="cpu",
        train_indices=train_indices,
    )

    # Explicit reference: no helper batching, DataLoader, or training loop calls.
    torch.manual_seed(seed)
    np.random.seed(seed)
    reference_head = make_cached_head(6, 3, "linear")
    reference_optimizer = torch.optim.AdamW(
        reference_head.parameters(), lr=0.003, weight_decay=0.0001
    )
    active = torch.as_tensor(train_indices, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = active[torch.randperm(len(active), generator=generator)]
    reference_loss_sum = torch.zeros(())
    reference_count = 0
    for start in range(0, len(order), 3):
        indices = order[start : start + 3]
        batch_x = torch.as_tensor(train_x).index_select(0, indices).float()
        batch_y = torch.as_tensor(train_y).index_select(0, indices).long()
        loss = torch.nn.functional.cross_entropy(reference_head(batch_x), batch_y)
        loss.backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()
        reference_loss_sum += loss.detach() * len(indices)
        reference_count += len(indices)
    assert torch.allclose(actual_head.weight, reference_head.weight, atol=1e-7, rtol=1e-7)
    assert torch.allclose(actual_head.bias, reference_head.bias, atol=1e-7, rtol=1e-7)
    assert history[0]["train_loss"] == pytest.approx(
        float(reference_loss_sum.item()) / reference_count
    )
    assert history[0]["epoch_time_seconds"] >= 0

    metrics, predictions, _ = evaluate_head(
        actual_head, val_x, val_y, val_metadata, batch_size=3, device="cpu"
    )
    reference_logits = actual_head(torch.as_tensor(val_x).float())
    reference_predictions = reference_logits.argmax(dim=1).numpy()
    assert np.array_equal(predictions, reference_predictions)
    assert metrics["loss"] == pytest.approx(
        float(torch.nn.functional.cross_entropy(reference_logits, torch.as_tensor(val_y).long()).item())
    )
