"""Part 3 leakage, deterministic transfer, representation and reload contracts."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from soilcv.artifacts import package_identity
from soilcv.bags import cache_grid, make_grid_manifest
from soilcv.curves import TARGET_COLUMNS, validate_curves
from soilcv.ssl.adaptation import ADAPTATION, adapt_encoder, dino_loss, ema_update
from soilcv.ssl.data import (
    TileTransform,
    adaptation_loader,
    camera_features,
    pool_features,
)
from soilcv.ssl.evaluation import frozen_predict, linear_cka, rankme, weighted_knn
from soilcv.ssl.inference import predict_fold
from soilcv.ssl.model import SSLEncoder


def tiny_config(name):
    common = {
        "hidden_size": 12,
        "intermediate_size": 24,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "image_size": 224,
        "patch_size": 16,
    }
    if name == "dinov3":
        common["num_register_tokens"] = 2
    return common


@pytest.mark.parametrize("name", ["dinov3", "mae"])
def test_encoder_is_deterministic_and_differentiable(name):
    torch.set_num_threads(1)
    model = SSLEncoder(name, config=tiny_config(name)).eval()
    x = torch.randn(2, 3, 224, 224)
    rng = torch.get_rng_state().clone()
    before, after = model(x), model(x)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert before.shape == (2, 24)
    before.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert set(model.layer_ids()) == set(dict(model.named_parameters()))


def test_pooling_uses_area_then_equal_photos():
    ids = pd.DataFrame(
        {"sample_id": ["a"] * 3, "photo": ["x", "x", "y"], "tile_id": ["0", "1", "0"]}
    )
    manifest = ids.assign(photo_weight=[0.25, 0.75, 1.0])
    samples, pooled = pool_features(ids, np.array([[0.0], [4.0], [9.0]]), manifest)
    assert samples.sample_id.tolist() == ["a"]
    np.testing.assert_allclose(pooled, [[6.0]])
    camera_ids, values = camera_features(
        ids,
        np.array([[0.0], [4.0], [9.0]]),
        manifest.assign(device=["Motorola", "Motorola", "Samsung"]),
        ["a"],
    )
    assert camera_ids.device.tolist() == ["motorola", "samsung"]
    np.testing.assert_allclose(values, [[3.0], [9.0]])


@pytest.mark.parametrize("folds", [[0, 1], [0, np.nan], [1, 1]])
def test_adaptation_rejects_heldout_or_test(folds):
    manifest = pd.DataFrame(
        {
            "fold": folds,
            "sample_id": ["a", "b"],
            "photo": ["x", "y"],
            "photo_weight": [1, 1],
        }
    )
    if folds == [1, 1]:
        loader = adaptation_loader(manifest, "/tmp", fold=0, batch_size=2)
        assert len(loader.dataset) == 2
    else:
        with pytest.raises(ValueError, match="training photos"):
            adaptation_loader(manifest, "/tmp", fold=0)


def test_dino_teacher_is_detached_and_ema_is_exact():
    student = [torch.randn(4, 8, requires_grad=True) for _ in range(2)]
    teacher = [torch.randn(4, 8, requires_grad=True) for _ in range(2)]
    loss, probabilities = dino_loss(student, teacher, torch.zeros(1, 8))
    loss.backward()
    assert all(value.grad is not None for value in student)
    assert all(value.grad is None for value in teacher)
    torch.testing.assert_close(probabilities[0].sum(-1), torch.ones(4))
    a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    initial = a.weight.detach().clone()
    ema_update(a, b, 0.75)
    torch.testing.assert_close(a.weight, initial * 0.75 + b.weight * 0.25)


def test_cka_invariance_collapse_and_matched_rows():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert linear_cka(x, x @ rotation * 12 + 17) == pytest.approx(1)
    assert linear_cka(x, np.ones_like(x)) is None
    assert linear_cka(x, rng.normal(size=(30, 8))) < 0.7
    assert rankme(np.ones((20, 8))) == 0
    with pytest.raises(ValueError, match="matched"):
        linear_cka(x, x[:3])


def test_frozen_baselines_fit_train_only_and_return_valid_curves():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(8, 12))
    targets = np.sort(rng.uniform(0, 100, size=(8, 11)), axis=1)
    targets[:, -1] = 100
    query = rng.normal(size=(3, 12))
    predicted, fitted = frozen_predict(train, targets, query, "ridge")
    np.testing.assert_allclose(fitted.steps[0][1].mean_, train.mean(0))
    validate_curves(predicted, atol=0)
    validate_curves(weighted_knn(train, targets, query), atol=0)


def test_dihedral_transform_preserves_pixel_grid():
    image = Image.fromarray(
        np.random.default_rng(4).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    )
    before = TileTransform()(image)
    after = TileTransform(train=True, photometric=False)(image)
    for channel in range(3):
        torch.testing.assert_close(
            before[channel].flatten().sort().values,
            after[channel].flatten().sort().values,
        )


@pytest.mark.parametrize("encoder_name", ["dinov3", "mae"])
def test_complete_tiny_source_and_inference_reload(tmp_path, monkeypatch, encoder_name):
    from soilcv.ssl import experiments as study

    torch.set_num_threads(1)
    cache, folder = tmp_path / "cache", tmp_path / encoder_name
    runs, published, active = [], [], []

    class RecordedRun:
        def __init__(self, config):
            self.config, self.summary, self.history = config, {}, []
            self.id, self.url = str(len(runs)), f"https://example.com/run/{len(runs)}"

        def __enter__(self):
            assert not active
            active.append(self)
            return self

        def __exit__(self, *args):
            assert active.pop() is self
            return False

        def log(self, data):
            self.history.append(data)

        def define_metric(self, name, *, step_metric):
            self.defined_metric = (name, step_metric)

    def start(config, **kwargs):
        run = RecordedRun(config)
        runs.append(run)
        return run

    monkeypatch.setattr(study, "start_run", start)
    monkeypatch.setattr(study, "publish_fold", lambda run, *args: published.append(run))
    monkeypatch.setattr(study, "CACHE", cache)
    monkeypatch.setitem(study.SETTINGS, "lp_epochs", 2)
    monkeypatch.setitem(study.SETTINGS, "ft_epochs", 1)
    photos = []
    for i in range(5):
        path = tmp_path / f"sample_{i}.png"
        Image.fromarray(
            np.random.default_rng(i).integers(0, 255, (128, 128, 3), dtype=np.uint8)
        ).save(path)
        photos.append(
            {
                "sample_id": f"soil-{i}",
                "photo": path.name,
                "path": str(path),
                "ppm_est": 1.0,
                "device": "Samsung",
            }
        )
    photos = pd.DataFrame(photos)
    tiles = make_grid_manifest(photos, tile_mm=64, output_px=224)
    cache_grid(tiles, cache)
    folds = pd.DataFrame(
        {"sample_id": [f"soil-{i}" for i in range(4)], "fold": range(4)}
    )
    tiles = tiles.merge(folds, on="sample_id", how="left")
    labels = folds[["sample_id"]].copy()
    labels[list(TARGET_COLUMNS)] = np.stack(
        [np.linspace(5 + i * 2, 100, 11) for i in range(4)]
    )
    encoder = SSLEncoder(encoder_name, config=tiny_config(encoder_name))
    metadata = {"identity": {"soilcv": package_identity()}, "study_id": "tiny-test"}
    before, diagnostic = study.run_source(
        encoder_name,
        encoder_name,
        encoder,
        tiles,
        labels,
        folds,
        0,
        folder,
        metadata,
        "cpu",
        None,
        tracking_root=tmp_path / "tracking",
    )
    assert len(before) == len(diagnostic) == 4
    initial = torch.load(folder / "encoder.pt", weights_only=True)
    tuned = torch.load(folder / "full/weights.pt", weights_only=True)
    assert any(
        not torch.equal(value, tuned["encoder." + name])
        for name, value in initial.items()
    )
    strategies = ["knn", "ridge", "lp", "lpft", "full"]
    if encoder_name == "dinov3":
        strategies += ["ridge-cls", "lpft-uniform-lr", "lpft-no-color"]
    assert len(runs) == len(strategies)
    assert {run.config.experiment_id for run in runs} == {
        f"{encoder_name}-{strategy}" for strategy in strategies
    }
    assert all(run.config.fold == 0 for run in runs)
    assert not active
    for run in runs:
        assert run.history[0]["epoch"] == 0
        assert {
            "val/emd",
            "val/mae_pp",
            "val/worst_point_pp",
            "val/d50_abs_log10",
            "val/motorola_emd",
            "val/samsung_emd",
        } <= run.history[0].keys()
        assert "best/emd" in run.summary
        assert not any(
            key.startswith(encoder_name + "-") for row in run.history for key in row
        )
        if run.config.params["strategy"] not in {"knn", "ridge"}:
            assert "train/loss" in run.history[0]
            assert "train/lr_0" in run.history[0]
            assert "selected/epoch" in run.summary
            assert "selected/camera_disagreement" in run.summary
        if run.config.params["strategy"] == "lp":
            assert [row["epoch"] for row in run.history] == [0, 1]
            assert (
                run.history[1]["train/global_step"]
                > run.history[0]["train/global_step"]
            )
    assert len(published) == (1 if encoder_name == "mae" else 0)
    if published:
        assert published[0].config.experiment_id == "mae-full"
    for strategy in strategies:
        result = predict_fold(
            folder / strategy,
            photos[photos.sample_id.eq("soil-4")],
            cache,
            device="cpu",
        )
        saved = pd.read_csv(folder / strategy / "test_predictions.csv")
        np.testing.assert_allclose(
            result[list(TARGET_COLUMNS)], saved[list(TARGET_COLUMNS)], atol=2e-4
        )
    if encoder_name != "dinov3":
        return
    train = tiles[tiles.fold.notna() & tiles.fold.ne(0)]
    settings = {**ADAPTATION, "epochs": 1, "batch_size": 4}
    adaptation_run = RecordedRun(None)
    _, history = adapt_encoder(
        encoder,
        adaptation_loader(train, cache, fold=0, batch_size=4),
        tmp_path / "adaptation",
        fold=0,
        device="cpu",
        settings=settings,
        run=adaptation_run,
    )
    assert np.isfinite(history[0]["loss"])
    assert adaptation_run.defined_metric == ("adaptation/*", "epoch")
    assert adaptation_run.history[0]["epoch"] == 0
    assert "adaptation/loss" in adaptation_run.history[0]
    assert "adaptation/epoch" not in adaptation_run.history[0]
    assert "train/loss" not in adaptation_run.history[0]
    config = json.loads((tmp_path / "adaptation" / "config.json").read_text())
    assert "soil-0" not in config["train_ids"] and "soil-4" not in config["train_ids"]


def test_report_collects_complete_oof_and_rejects_missing_fold(tmp_path):
    from soilcv.artifacts import file_hash, write_json
    from soilcv.ssl.analysis import collect_results, expected_experiments
    from soilcv.ssl.data import PREPROCESSING

    folds = pd.DataFrame(
        {"sample_id": [f"s-{i}" for i in range(8)], "fold": [0, 0, 1, 1, 2, 2, 3, 3]}
    )
    labels = folds[["sample_id"]].copy()
    labels[list(TARGET_COLUMNS)] = np.linspace(4, 100, 11)[None].repeat(8, axis=0)
    template = labels.iloc[:2].copy()
    template["sample_id"] = ["test-a", "test-b"]
    labels.to_json(tmp_path / "labels.json", orient="records")
    template.to_json(tmp_path / "template.json", orient="records")
    folds.to_csv(tmp_path / "sample_folds.csv", index=False)
    for experiment, (encoder, source, strategy) in expected_experiments().items():
        for fold in range(4):
            parent = tmp_path / "folds" / encoder / f"fold_{fold}"
            folder = parent / source / strategy
            folder.mkdir(parents=True, exist_ok=True)
            write_json(
                parent / "complete.json",
                {"study_id": "test", "run_url": "https://example.com/run"},
            )
            write_json(
                folder / "config.json",
                {
                    "identity": {"soilcv": package_identity()},
                    "preprocessing": PREPROCESSING,
                    "experiment": experiment,
                    "strategy": "ridge" if strategy == "ridge-cls" else strategy,
                    "fold": fold,
                    "study_id": "test",
                },
            )
            valid = labels[
                labels.sample_id.isin(folds.loc[folds.fold.eq(fold), "sample_id"])
            ].copy()
            valid["fold"] = fold
            valid.to_csv(folder / "oof.csv", index=False)
            template.to_csv(folder / "test_predictions.csv", index=False)
            write_json(
                folder / "result.json",
                {"experiment": experiment, "fold": fold, "emd": 0.0},
            )
            write_json(folder / "compute.json", {"gpu_hours": 0.01})
            write_json(folder.parent / "feature_compute.json", {"gpu_hours": 0.02})
            write_json(folder / "representation.json", {"cka": 0.9})
            weights = (
                "regressor.npz"
                if strategy in {"knn", "ridge", "ridge-cls"}
                else "weights.pt"
            )
            (folder / weights).write_bytes(b"test")
            write_json(
                folder / "hashes.json",
                {
                    path.name: file_hash(path)
                    for path in folder.iterdir()
                    if path.is_file()
                },
            )
    for fold in range(4):
        folder = tmp_path / "folds" / "dinov3" / f"fold_{fold}" / "adaptation"
        write_json(folder / "compute.json", {"gpu_hours": 0.05})
        write_json(folder / "representation.json", {"cka": 0.8})
        write_json(
            folder / "history.json",
            [{"epoch": 0, "loss": 2.0, "teacher_entropy": 3.0, "feature_std": 0.5}],
        )
    table = collect_results(tmp_path)
    assert len(table) == 18
    np.testing.assert_allclose(table.cv_emd, 0, atol=1e-10)
    assert (tmp_path / "submission.csv").is_file()
    assert (tmp_path / "results/adaptation_curves.png").is_file()
    (tmp_path / "folds/mae/fold_3/mae/full/oof.csv").unlink()
    with pytest.raises(FileNotFoundError):
        collect_results(tmp_path)
