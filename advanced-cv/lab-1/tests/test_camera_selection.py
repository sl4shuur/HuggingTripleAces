"""Paired camera errors and two-metric checkpoint selection, without W&B."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from soilcv.bags import one_bag
from soilcv.camera import camera_metrics, domain_metrics, improves_checkpoint
from soilcv.cnn.predict import validate_bags_camera
from soilcv.config import ExperimentConfig
from soilcv.curves import TARGET_COLUMNS
from soilcv.nn import EMDLoss, MassCurveHead
from soilcv.training import fit_supervised_fold, train_one_epoch


def curve(value):
    return np.r_[np.full(10, value), 100.0]


def labels(ids):
    return pd.DataFrame(
        [
            {"sample_id": sample, **dict(zip(TARGET_COLUMNS, curve(20)))}
            for sample in ids
        ]
    )


def camera_rows():
    return pd.DataFrame(
        [
            {
                "sample_id": sample,
                "device": device,
                **dict(zip(TARGET_COLUMNS, curve(value))),
            }
            for sample, device, value in [
                ("paired", "Motorola G", 22),
                ("paired", "Samsung A", 26),
                ("one-camera", "Motorola G", 90),
            ]
        ]
    )


def test_camera_gap_uses_exact_same_heldout_samples():
    truth = labels(["one-camera", "paired"])
    metrics = camera_metrics(truth, camera_rows(), require_pairs=True)
    assert metrics["camera_paired_samples"] == 1
    assert metrics["motorola_emd"] == pytest.approx(10)
    assert metrics["samsung_emd"] == pytest.approx(30)
    assert metrics["camera_disagreement"] == pytest.approx(20)
    motorola, pairs = domain_metrics(truth, camera_rows(), "Motorola")
    samsung, _ = domain_metrics(truth, camera_rows(), "Samsung")
    assert pairs.sample_id.tolist() == ["paired"]
    assert motorola["e_same"] == pytest.approx(10)
    assert motorola["e_cross"] == pytest.approx(30)
    assert motorola["delta_domain"] == pytest.approx(20)
    assert samsung["delta_domain"] == pytest.approx(-20)
    with pytest.raises(ValueError, match="at least one paired"):
        camera_metrics(truth, camera_rows().iloc[[0]], require_pairs=True)
    with pytest.raises(ValueError, match="one pooled"):
        camera_metrics(truth, pd.concat([camera_rows(), camera_rows().iloc[[0]]]))


def test_pair_selection_rejects_tradeoffs_and_cannot_oscillate():
    best = {"emd": 10.0, "camera_disagreement": 5.0}
    assert improves_checkpoint(best, None, selection="emd_camera")
    assert not improves_checkpoint(best, best, selection="emd_camera")
    assert improves_checkpoint(
        {"emd": 9, "camera_disagreement": 5}, best, selection="emd_camera"
    )
    assert improves_checkpoint(
        {"emd": 10, "camera_disagreement": 4}, best, selection="emd_camera"
    )
    assert not improves_checkpoint(
        {"emd": 9, "camera_disagreement": 6}, best, selection="emd_camera"
    )
    assert not improves_checkpoint(
        {"emd": 11, "camera_disagreement": 4}, best, selection="emd_camera"
    )
    assert improves_checkpoint(
        {"emd": 9, "camera_disagreement": 6}, best, selection="emd"
    )
    with pytest.raises(ValueError, match="finite camera_disagreement"):
        improves_checkpoint({"emd": 9}, best, selection="emd_camera")


class TinyPhotos:
    def __init__(self, sample, fold):
        self.manifest = pd.DataFrame({"sample_id": [sample], "fold": [fold]})
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "sample_id": self.sample,
            "image": [torch.ones(3, 5, 7), torch.full((3, 7, 4), 2.0)],
            "bag_weights": torch.tensor([0.5, 0.5]),
            "device": ["Motorola", "Samsung"],
            "target": torch.tensor(curve(20), dtype=torch.float32)[None],
        }


class TinyPhotoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = MassCurveHead(3)

    def encode(self, images):
        return torch.stack([image.mean(dim=(-1, -2)) for image in images])

    def pool(self, features, weights):
        return self.head(
            (features * weights[:, None]).sum(0, keepdim=True) / weights.sum()
        )

    def forward(self, images, weights):
        return self.pool(self.encode(images), weights)


def test_variable_photo_bags_train_and_validate_once():
    model = TinyPhotoModel()
    loader = DataLoader(TinyPhotos("paired", 0), batch_size=1, collate_fn=one_bag)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    before = model.head.projection.weight.detach().clone()
    train_one_epoch(model, loader, optimizer, EMDLoss())
    assert not torch.equal(model.head.projection.weight, before)
    metrics, predictions = validate_bags_camera(model, loader, labels(["paired"]))
    assert len(predictions) == 1
    assert metrics["camera_paired_samples"] == 1
    assert np.isfinite(metrics["camera_disagreement"])


def test_pair_state_survives_resume_and_patience_counts_rejected_epochs(
    tmp_path, monkeypatch
):
    from soilcv import training

    train = DataLoader(TinyPhotos("train", 1), batch_size=1, collate_fn=one_bag)
    valid = DataLoader(TinyPhotos("valid", 0), batch_size=1, collate_fn=one_bag)
    truth = labels(["valid"])
    values = iter([(10, 10), (8, 12), (9, 9), (8, 10), (7, 11)])

    def validate(model, loader, labels, **kwargs):
        emd, disagreement = next(values)
        return {"emd": emd, "camera_disagreement": disagreement}, labels.copy()

    monkeypatch.setattr(
        training, "train_one_epoch", lambda *args, **kwargs: {"global_step": 0}
    )
    config = ExperimentConfig("pair", fold=0, output_root=str(tmp_path))
    model = TinyPhotoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    kwargs = {
        "patience": 2,
        "selection": "emd_camera",
        "validation_fn": validate,
    }
    fit_supervised_fold(
        model, train, valid, truth, optimizer, EMDLoss(), config, epochs=3, **kwargs
    )
    state = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert state["metadata"]["selected_epoch"] == 2
    assert state["metadata"]["selected_metrics"]["camera_disagreement"] == 9
    _, history = fit_supervised_fold(
        model,
        train,
        valid,
        truth,
        optimizer,
        EMDLoss(),
        config,
        epochs=8,
        resume_from=config.output_dir / "last.pt",
        **kwargs,
    )
    assert len(history) == 5
    state = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert state["metadata"]["selected_epoch"] == 2
    assert state["metadata"]["best_emd"] == 9
    assert state["metadata"]["bad_epochs"] == 2
    settings = json.loads((config.output_dir / "training_settings.json").read_text())
    assert settings["selection"] == "emd_camera"


def test_warmup_does_not_select_prior_and_can_resume_without_best(
    tmp_path, monkeypatch
):
    from soilcv import training

    train = DataLoader(TinyPhotos("train", 1), batch_size=1, collate_fn=one_bag)
    valid = DataLoader(TinyPhotos("valid", 0), batch_size=1, collate_fn=one_bag)
    truth = labels(["valid"])
    calls = 0

    def interrupted_validation(model, loader, labels, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("test interruption")
        return {"emd": 1, "camera_disagreement": 0}, labels.copy()

    monkeypatch.setattr(
        training, "train_one_epoch", lambda *args, **kwargs: {"global_step": 0}
    )
    config = ExperimentConfig("warmup-pair", fold=0, output_root=str(tmp_path))
    model = TinyPhotoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    kwargs = {"selection": "emd_camera", "selection_start_epoch": 3, "patience": 1}
    with pytest.raises(RuntimeError, match="test interruption"):
        fit_supervised_fold(
            model,
            train,
            valid,
            truth,
            optimizer,
            EMDLoss(),
            config,
            epochs=6,
            validation_fn=interrupted_validation,
            **kwargs,
        )
    state = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert state["metadata"]["selected_metrics"] is None
    assert state["metadata"]["bad_epochs"] == 0
    assert not (config.output_dir / "best.pt").exists()
    values = iter([(1, 0), (10, 10), (9, 9), (8, 11)])

    def resumed_validation(model, loader, labels, **kwargs):
        emd, disagreement = next(values)
        return {"emd": emd, "camera_disagreement": disagreement}, labels.copy()

    _, history = fit_supervised_fold(
        model,
        train,
        valid,
        truth,
        optimizer,
        EMDLoss(),
        config,
        epochs=6,
        validation_fn=resumed_validation,
        resume_from=config.output_dir / "last.pt",
        **kwargs,
    )
    assert len(history) == 6
    state = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert state["metadata"]["selected_epoch"] == 4
    assert state["metadata"]["best_emd"] == 9
    assert state["metadata"]["selected_metrics"]["camera_disagreement"] == 9
