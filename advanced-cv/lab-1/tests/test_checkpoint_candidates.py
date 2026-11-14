"""EMD controls stopping; camera disagreement only selects an optional third epoch."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from soilcv.bags import one_bag
from soilcv.checkpoints import (
    candidate_paths,
    candidate_predictions,
    load_candidates,
    restore_candidates,
    selected_candidates,
    update_candidates,
)
from soilcv.config import ExperimentConfig
from soilcv.curves import TARGET_COLUMNS
from soilcv.nn import EMDLoss
from soilcv.training import fit_supervised_fold, move_images


def curves(value, samples=("soil-b", "soil-a")):
    frame = pd.DataFrame(
        np.tile(np.r_[np.full(10, value), 100], (len(samples), 1)),
        columns=TARGET_COLUMNS,
    )
    frame.insert(0, "sample_id", samples)
    return frame


@pytest.mark.parametrize(
    ("third_camera", "expected"),
    [(9.0, [0, 1, 2]), (10.0, [0, 1]), (11.0, [0, 1]), (None, [0, 1])],
)
def test_third_checkpoint_needs_lower_camera_gap_than_second(third_camera, expected):
    records = [
        {"epoch": 0, "emd": 20, "camera_disagreement": 5},
        {"epoch": 1, "emd": 21, "camera_disagreement": 10},
        {"epoch": 2, "emd": 22, "camera_disagreement": third_camera},
    ]
    assert [row["epoch"] for row in selected_candidates(records)] == expected


def test_pool_keeps_three_states_and_aligns_selected_oof(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    records = []
    for epoch, (emd, camera) in enumerate([(40, 30), (30, 40), (20, 20), (10, 10)]):
        with torch.no_grad():
            model.weight.fill_(epoch)
        predictions = curves(10 * epoch)
        if epoch % 2:
            predictions = predictions.iloc[::-1]
        records = update_candidates(
            tmp_path,
            records,
            model,
            epoch,
            {"emd": emd, "camera_disagreement": camera},
            predictions,
        )
    assert [row["epoch"] for row in load_candidates(tmp_path, selected_only=False)] == [
        3,
        2,
        1,
    ]
    assert [path.name for path in candidate_paths(tmp_path)] == [
        "epoch_0003.pt",
        "epoch_0002.pt",
    ]
    assert not (tmp_path / "candidates/epoch_0000.pt").exists()
    assert not (tmp_path / "candidates/epoch_0000_oof.csv").exists()
    selected = candidate_predictions(tmp_path)
    assert selected.sample_id.tolist() == ["soil-a", "soil-b"]
    np.testing.assert_allclose(selected[list(TARGET_COLUMNS)].iloc[:, :-1], 25)
    for row in records:
        state = torch.load(tmp_path / row["weights"], weights_only=True)
        assert float(state["weight"].item()) == row["epoch"]
    index = json.loads((tmp_path / "candidates.json").read_text())
    assert index["selected_epochs"] == [3, 2]
    path = candidate_paths(tmp_path)[0]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed candidate"):
        candidate_paths(tmp_path)


def test_resume_copies_and_checks_all_retained_candidate_sidecars(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    model = nn.Linear(1, 1)
    records = []
    for epoch in range(3):
        records = update_candidates(
            source,
            records,
            model,
            epoch,
            {"emd": 10 + epoch, "camera_disagreement": 5 - epoch},
            curves(epoch),
        )
    restore_candidates(source, destination, records, 3)
    assert (
        load_candidates(destination, selected_only=False, verify_predictions=True)
        == records
    )
    (source / records[-1]["predictions"]).unlink()
    with pytest.raises(ValueError, match="saved candidate sidecars"):
        restore_candidates(source, tmp_path / "missing", records, 3)


def test_interrupted_index_update_keeps_previous_last_checkpoint_resumable(tmp_path):
    model = nn.Linear(1, 1)
    previous = []
    for epoch in range(3):
        previous = update_candidates(
            tmp_path,
            previous,
            model,
            epoch,
            {"emd": 30 - epoch, "camera_disagreement": epoch},
            curves(epoch),
        )
    # A new index can be written before last.pt is replaced. Old sidecars must remain.
    update_candidates(
        tmp_path,
        previous,
        model,
        3,
        {"emd": 20, "camera_disagreement": 0},
        curves(3),
        prune=False,
    )
    assert (tmp_path / previous[-1]["weights"]).exists()
    restore_candidates(tmp_path, tmp_path, previous, 3)
    assert (
        load_candidates(tmp_path, selected_only=False, verify_predictions=True)
        == previous
    )
    assert not (tmp_path / "candidates/epoch_0003.pt").exists()


class OneSoil:
    def __init__(self, sample, fold):
        self.sample = sample
        self.manifest = pd.DataFrame({"sample_id": [sample], "fold": [fold]})

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"image": torch.ones(1, 1), "target": torch.ones(1, 11)}


def test_best_and_patience_follow_emd_when_camera_error_grows(tmp_path, monkeypatch):
    from soilcv import training

    train = DataLoader(OneSoil("train", 1), batch_size=1, collate_fn=one_bag)
    valid = DataLoader(OneSoil("valid", 0), batch_size=1, collate_fn=one_bag)
    truth = curves(20, ("valid",))
    values = iter([(45, 12), (39, 15), (31, 17), (28, 18), (29, 19), (30, 20)])
    epoch = -1

    def train_epoch(model, *args, **kwargs):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            model.weight.fill_(epoch)
        return {"global_step": epoch}

    def validate(model, loader, labels, **kwargs):
        emd, camera = next(values)
        return {"emd": emd, "camera_disagreement": camera}, curves(
            20 + epoch, ("valid",)
        )

    monkeypatch.setattr(training, "train_one_epoch", train_epoch)
    config = ExperimentConfig("emd-candidates", fold=0, output_root=str(tmp_path))
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    kwargs = {
        "validation_fn": validate,
        "selection": "emd",
        "keep_top_k": 3,
        "patience": 2,
    }
    fit_supervised_fold(
        model, train, valid, truth, optimizer, EMDLoss(), config, epochs=3, **kwargs
    )
    _, history = fit_supervised_fold(
        model,
        train,
        valid,
        truth,
        optimizer,
        EMDLoss(),
        config,
        epochs=10,
        resume_from=config.output_dir / "last.pt",
        **kwargs,
    )
    assert len(history) == 6
    assert float(model.weight.item()) == 3
    state = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert state["metadata"]["selected_epoch"] == 3
    assert state["metadata"]["selected_metrics"] == {
        "emd": 28,
        "camera_disagreement": 18,
    }
    assert state["metadata"]["bad_epochs"] == 2
    assert [row["epoch"] for row in state["metadata"]["candidate_records"]] == [3, 4, 5]
    assert [row["epoch"] for row in load_candidates(config.output_dir)] == [3, 4]
    with pytest.raises(ValueError, match="Existing checkpoint candidates"):
        fit_supervised_fold(
            model,
            train,
            valid,
            truth,
            optimizer,
            EMDLoss(),
            config,
            epochs=10,
            **kwargs,
        )


def test_move_images_handles_nested_multiscale_views():
    image = torch.ones(3, 8, 9)
    local = torch.ones(2, 3, 8, 8)
    views = [
        {"global": image, "local": local, "local_weights": torch.tensor([0.3, 0.7])}
    ]
    moved = move_images(views, "cpu")
    assert torch.equal(moved[0]["global"], image)
    assert torch.equal(moved[0]["local"], local)
    torch.testing.assert_close(moved[0]["local_weights"], torch.tensor([0.3, 0.7]))


@pytest.mark.parametrize("scheduler_step", ["epoch", "metric"])
def test_epoch_scheduler_waits_when_amp_skips_every_optimizer_update(
    tmp_path, monkeypatch, scheduler_step
):
    from soilcv import training

    train = DataLoader(OneSoil("train", 1), batch_size=1, collate_fn=one_bag)
    valid = DataLoader(OneSoil("valid", 0), batch_size=1, collate_fn=one_bag)
    truth = curves(20, ("valid",))
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    if scheduler_step == "epoch":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)
    calls = []
    original_step = scheduler.step
    epoch = -1

    def record_step(self, *args):
        calls.append((epoch, args))
        original_step(*args)

    def train_epoch(model, loader, optimizer, *args, **kwargs):
        nonlocal epoch
        epoch += 1
        if epoch == 1:
            optimizer.zero_grad()
            model.weight.square().mean().backward()
            optimizer.step()
        return {
            "global_step": epoch,
            "optimizer_steps": epoch,
            "skipped_steps": 1 - epoch,
        }

    def validate(model, loader, labels, **kwargs):
        return {"emd": 10 - 5 * epoch, "camera_disagreement": epoch}, labels.copy()

    monkeypatch.setattr(training, "train_one_epoch", train_epoch)
    monkeypatch.setattr(type(scheduler), "step", record_step)
    config = ExperimentConfig("amp-scheduler", fold=0, output_root=str(tmp_path))
    _, history = fit_supervised_fold(
        model,
        train,
        valid,
        truth,
        optimizer,
        EMDLoss(),
        config,
        epochs=2,
        scheduler=scheduler,
        scheduler_step=scheduler_step,
        validation_fn=validate,
    )
    assert len(history) == 2
    assert history[0]["train"]["optimizer_steps"] == 0
    assert calls == [(1, ())] if scheduler_step == "epoch" else calls == [(1, (5,))]
    state = torch.load(config.output_dir / "best.pt", weights_only=False)
    assert state["metadata"]["selected_epoch"] == 1
    assert state["metadata"]["selected_metrics"]["emd"] == 5
