import copy
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

torch.set_num_threads(1)

from soilcv.artifacts import load_checkpoint, save_checkpoint
from soilcv.config import ExperimentConfig, load_config, save_config, seed_everything
from soilcv.curves import (
    TARGET_COLUMNS,
    aggregate_predictions,
    align_samples,
    make_submission,
    repair_curves,
    validate_curves,
)
from soilcv.data import (
    DEFAULT_FOLDS_PATH,
    TileDataset,
    load_folds,
    load_metadata,
    sample_weights,
    validate_folds,
)
from soilcv.features import (
    extract_features,
    feature_identity,
    load_features,
    save_features,
)
from soilcv.metrics import (
    emd_per_sample,
    evaluate_samples,
    grain_size,
    range_errors,
    summarize_cv,
)
from soilcv.nn import EMDLoss, MassCurveHead, freeze_bn_stats
from soilcv.tiles import build_tile_cache, check_cache, estimate_ppm, make_tile_manifest
from soilcv.tracking import start_run
from soilcv.training import fit_supervised_fold, train_one_epoch, validate_epoch


def curves_frame(ids, values):
    frame = pd.DataFrame(np.asarray(values), columns=TARGET_COLUMNS)
    frame.insert(0, "sample_id", ids)
    return frame


def test_plain_import_is_light():
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import soilcv.metrics,sys; assert 'torch' not in sys.modules; assert 'wandb' not in sys.modules",
        ],
        env=env,
        check=True,
    )


def test_metric_matches_physical_interval_and_torch_gradient():
    fine = np.full(11, 100.0)
    coarse = np.r_[np.zeros(10), 100.0]
    assert emd_per_sample(fine, coarse) == pytest.approx(500)
    wrong_end = fine.copy()
    wrong_end[-1] = 80
    assert emd_per_sample(fine, wrong_end) == 0
    with pytest.raises(ValueError):
        validate_curves(wrong_end)
    changed = coarse.copy()
    changed[0] = 10
    assert emd_per_sample(coarse, changed) == pytest.approx(
        10 * np.log10(0.0063 / 0.002)
    )
    seed_everything(4)
    head = MassCurveHead(3)
    pred = head(torch.randn(5, 3))
    true = torch.tensor(np.tile(coarse, (5, 1)), dtype=torch.float32)
    validate_curves(pred.detach().numpy())
    loss = EMDLoss()(pred, true)
    assert loss.item() == pytest.approx(
        emd_per_sample(true.numpy(), pred.detach().numpy()).mean(), rel=1e-6
    )
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
    assert sum(p.grad.abs().sum() for p in head.parameters()) > 0


def test_repair_does_not_modify_input_and_reports_censored_d10():
    raw = np.array([-2, 3, 2, 8, 6, 30, 50, 70, 90, 99, 98.0])
    original = raw.copy()
    fixed = repair_curves(raw)
    validate_curves(fixed)
    assert np.array_equal(raw, original)
    assert np.isnan(grain_size(np.full(11, 100.0), 10))
    frame = curves_frame(["a"], [np.full(11, 100.0)])
    scores = evaluate_samples(frame, frame)
    assert scores["d10_valid_pairs"] == 0 and scores["d10_abs_log10"] is None


def test_aggregation_has_equal_photo_weight_and_strict_ids():
    a = np.full(11, 100.0)
    b = np.r_[np.zeros(10), 100.0]
    frame = curves_frame(["s"] * 4, [a, a, a, b])
    frame["photo"] = ["many", "many", "many", "one"]
    frame["tile_id"] = [0, 1, 2, 0]
    result = aggregate_predictions(frame)
    assert np.allclose(result[list(TARGET_COLUMNS)].iloc[0, :-1], 50)
    weights = sample_weights(frame)
    assert weights[:3].sum() == pytest.approx(weights[-1])
    with pytest.raises(ValueError):
        aggregate_predictions(pd.concat([frame, frame.iloc[[0]]]))
    labels = curves_frame(["s", "t"], [a, b])
    reversed_pred = labels.iloc[::-1]
    assert align_samples(labels, reversed_pred).sample_id.tolist() == ["s", "t"]
    assert make_submission(reversed_pred, labels).sample_id.tolist() == ["s", "t"]
    assert evaluate_samples(labels, reversed_pred)["emd"] == 0
    with pytest.raises(ValueError):
        align_samples(labels, reversed_pred.iloc[[0]])
    assert range_errors(labels, labels)["emd_part"].sum() == 0


def test_folds_and_complete_oof_are_checked():
    folds = load_folds(DEFAULT_FOLDS_PATH, pd.read_csv(DEFAULT_FOLDS_PATH).sample_id)
    assert len(folds) == 24 and folds.fold.value_counts().eq(6).all()
    labels = curves_frame(["a", "b"], [np.linspace(0, 100, 11)] * 2)
    small = pd.DataFrame({"sample_id": ["a", "b"], "fold": [0, 1]})
    pred = labels.copy()
    pred["fold"] = [0, 1]
    assert summarize_cv(labels, pred, small, expected_folds=(0, 1))["complete"]
    pred["fold"] = [1, 0]
    with pytest.raises(ValueError):
        summarize_cv(labels, pred, small, expected_folds=(0, 1))
    with pytest.raises(ValueError):
        validate_folds(
            pd.concat([small, small.iloc[[0]]]), labels.sample_id, expected_folds=(0, 1)
        )


def make_photo(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (80, 60), (120, 80, 30)).save(path)
    return pd.DataFrame(
        [
            {
                "sample_id": "a",
                "photo": "source.png",
                "path": str(path),
                "ppm_est": 1.0,
                "fold": 0,
            }
        ]
    )


def test_physical_cache_changes_with_scale_and_is_portable(tmp_path):
    assert estimate_ppm(11.492, (1600, 720), (4000, 1800)) == pytest.approx(4.5968)
    assert estimate_ppm(11.492, (720, 1600), (4000, 1800)) == pytest.approx(4.5968)
    photos = make_photo(tmp_path)
    manifest = make_tile_manifest(
        photos, tile_mm=20, output_px=8, centres=((0.5, 0.5),)
    )
    root = tmp_path / "cache"
    build_tile_cache(manifest, root)
    assert check_cache(manifest, root)
    assert manifest.output_ppm_est.iloc[0] == pytest.approx(0.4)
    changed = make_tile_manifest(
        photos.assign(ppm_est=1.1), tile_mm=20, output_px=8, centres=((0.5, 0.5),)
    )
    assert changed.cache_id.iloc[0] != manifest.cache_id.iloc[0]
    changed2 = manifest.copy()
    changed2["output_px"] = 9
    with pytest.raises(ValueError):
        check_cache(changed2, root)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    labels = curves_frame(["a"], [np.linspace(0, 100, 11)])
    data = TileDataset(manifest, moved, labels)
    assert data[0]["image"].shape == (3, 8, 8)
    assert check_cache(manifest, moved)
    Image.new("RGB", (80, 60), (0, 0, 0)).save(photos.path.iloc[0])
    with pytest.raises(ValueError):
        build_tile_cache(manifest, root)


def test_metadata_camera_scale_and_template_are_separate(tmp_path):
    train = tmp_path / "Training-All_Photos_updated"
    train.mkdir()
    test = tmp_path / "Test_All_Photos"
    test.mkdir()
    Image.new("RGB", (16, 12)).save(train / "Samsung_A52_F827_01.jpg")
    exif = Image.Exif()
    exif[272] = "iPhone 14"
    Image.new("RGB", (16, 12)).save(test / "iPhone14_HPC_Test Site (1).jpg", exif=exif)
    curves_frame(["F827"], [np.linspace(0, 100, 11)]).to_csv(
        tmp_path / "Training_labels_updated.csv", index=False
    )
    curves_frame(["HPC_Test Site"], [np.zeros(11)]).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    pd.DataFrame(
        [
            ["Samsung A52", "SM-A525F", 160, 120, 10],
            ["iPhone 14", "iPhone 14", 400, 300, 10],
        ],
        columns=["phone", "camera", "width", "height", "ppm"],
    ).to_csv(tmp_path / "ppm_updated.csv", index=False)
    labels, photos, template = load_metadata(tmp_path)
    assert len(photos) == 2 and labels.sample_id.tolist() == ["F827"]
    assert photos.query("split=='train'").ppm_est.iloc[0] == pytest.approx(1.0)
    assert not photos.absolute_scale_verified.any()
    assert template["200"].iloc[0] == 0


def test_feature_cache_binds_actual_weights_and_input(tmp_path):
    photos = make_photo(tmp_path)
    manifest = make_tile_manifest(
        photos, tile_mm=20, output_px=8, centres=((0.5, 0.5),)
    )
    root = tmp_path / "tiles"
    build_tile_cache(manifest, root)
    loader = DataLoader(TileDataset(manifest, root), batch_size=1)
    encoder = torch.nn.Sequential(
        torch.nn.BatchNorm2d(3), torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4)
    )
    encoder.train()
    freeze_bn_stats(encoder)
    before = encoder[0].running_mean.clone()
    ids, features = extract_features(encoder, loader)
    assert features.shape == (1, 4) and encoder.training and not encoder[0].training
    assert torch.equal(before, encoder[0].running_mean)
    identity = feature_identity(
        encoder,
        name="tiny",
        weights_revision="test-v1",
        preprocessing={"norm": "unit"},
        tile_cache_id=manifest.cache_id.iloc[0],
        pooling="flat",
    )
    folder = tmp_path / "features"
    save_features(folder, ids, features, identity)
    loaded_ids, loaded_features = load_features(folder, identity)
    assert loaded_ids.equals(ids) and np.array_equal(loaded_features, features)
    with torch.no_grad():
        encoder[-1].weight.add_(0.1)
    changed = feature_identity(
        encoder,
        name="tiny",
        weights_revision="test-v1",
        preprocessing={"norm": "unit"},
        tile_cache_id=manifest.cache_id.iloc[0],
        pooling="flat",
    )
    with pytest.raises(ValueError):
        load_features(folder, changed)
    with pytest.raises(ValueError):
        feature_identity(
            encoder,
            name="tiny",
            weights_revision="test-v1",
            preprocessing={},
            tile_cache_id="x",
            pooling="flat",
            fold=0,
        )


def test_checkpoint_restores_rng_optimizer_and_scheduler(tmp_path):
    seed_everything(8)
    model = torch.nn.Linear(2, 11)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    x = torch.tensor([[0.2, 0.4]])
    target = torch.ones(1, 11)

    def step(model, optimizer, scheduler):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), target).backward()
        optimizer.step()
        scheduler.step()

    step(model, optimizer, scheduler)
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=7,
        metadata={"identity": "a"},
    )
    expected_random = (random.random(), np.random.rand(), torch.rand(1))
    step(model, optimizer, scheduler)
    expected = copy.deepcopy(model.state_dict())
    restored = torch.nn.Linear(2, 11)
    opt = torch.optim.Adam(restored.parameters(), lr=10)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.8)
    state = load_checkpoint(
        checkpoint,
        restored,
        optimizer=opt,
        scheduler=sched,
        expected_metadata={"identity": "a"},
    )
    assert state["epoch"] == 2 and state["global_step"] == 7
    assert (
        random.random() == expected_random[0] and np.random.rand() == expected_random[1]
    )
    assert torch.equal(torch.rand(1), expected_random[2])
    step(restored, opt, sched)
    assert all(
        torch.equal(expected[k], value) for k, value in restored.state_dict().items()
    )
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, restored, expected_metadata={"identity": "b"})


def test_accumulation_handles_short_final_window():
    torch.manual_seed(4)
    x = torch.linspace(-1, 1, 7)[:, None]
    target = torch.arange(7.0)[:, None].repeat(1, 11)
    samples = [{"image": a, "target": b} for a, b in zip(x, target)]
    model = torch.nn.Linear(1, 11)
    reference = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.01)
    stats = train_one_epoch(
        model,
        DataLoader(samples, batch_size=2),
        opt,
        torch.nn.MSELoss(),
        accumulation_steps=2,
    )
    for start, end in [(0, 4), (4, 7)]:
        ref_opt.zero_grad()
        torch.nn.functional.mse_loss(
            reference(x[start:end]), target[start:end]
        ).backward()
        ref_opt.step()
    assert stats["optimizer_steps"] == 2
    assert all(
        torch.allclose(a, b, atol=1e-7)
        for a, b in zip(model.parameters(), reference.parameters())
    )


def test_tiny_fit_resume_and_leakage_guard(tmp_path):
    rows = []
    for i in range(4):
        image_path = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4), (30 + i * 40, 60, 80)).save(image_path)
        rows.append(
            {
                "sample_id": str(i),
                "photo": f"{i}.png",
                "tile_id": 0,
                "tile_path": image_path.name,
                "fold": 1 if i < 2 else 0,
            }
        )
    manifest = pd.DataFrame(rows)
    labels = curves_frame([str(i) for i in range(4)], [np.linspace(0, 100, 11)] * 4)
    train = DataLoader(TileDataset(manifest.iloc[:2], tmp_path, labels), batch_size=2)
    valid = DataLoader(TileDataset(manifest.iloc[2:], tmp_path, labels), batch_size=2)
    config = ExperimentConfig("tiny-fit", fold=0, output_root=str(tmp_path / "runs"))

    def make_model():
        return torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(48, 11), MassCurveHead()
        )

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    predictions, history = fit_supervised_fold(
        model, train, valid, labels.iloc[2:], optimizer, EMDLoss(), config, epochs=1
    )
    assert (
        len(history) == 1
        and set(predictions.sample_id) == {"2", "3"}
        and predictions.fold.eq(0).all()
    )
    fresh = make_model()
    opt = torch.optim.Adam(fresh.parameters(), lr=0.01)
    predictions, history = fit_supervised_fold(
        fresh,
        train,
        valid,
        labels.iloc[2:],
        opt,
        EMDLoss(),
        config,
        epochs=2,
        resume_from=config.output_dir / "last.pt",
    )
    metrics, _ = validate_epoch(fresh, valid, labels.iloc[2:])
    assert len(history) == 2 and metrics["emd"] == pytest.approx(
        min(row["val"]["emd"] for row in history)
    )
    assert (config.output_dir / "oof.csv").exists() and (
        config.output_dir / "compute.json"
    ).exists()
    with pytest.raises(ValueError):
        fit_supervised_fold(
            fresh, valid, valid, labels.iloc[2:], opt, EMDLoss(), config, epochs=1
        )


def test_wandb_defaults_online_and_secret_does_not_enter_config(tmp_path, monkeypatch):
    calls = {}

    class Run:
        settings = SimpleNamespace(mode="online")

        def define_metric(self, *args, **kwargs):
            pass

        def finish(self, **kwargs):
            calls["finish"] = kwargs

    def login(**kwargs):
        calls["login"] = kwargs
        return True

    def init(**kwargs):
        calls["init"] = kwargs
        return Run()

    sdk = SimpleNamespace(run=None, login=login, init=init)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class Secrets:
        def get_secret(self, label):
            calls["secret_label"] = label
            return "test-key-value"

    monkeypatch.setitem(
        sys.modules, "kaggle_secrets", SimpleNamespace(UserSecretsClient=Secrets)
    )
    cfg = ExperimentConfig("tracking-test", fold=0, output_root=str(tmp_path))
    start_run(cfg)
    assert calls["secret_label"] == "WANDB_API_KEY"
    assert (
        calls["init"]["entity"] == "public-team" and calls["init"]["mode"] == "online"
    )
    assert calls["init"]["group"] == "tracking-test"
    assert "test-key-value" not in (cfg.output_dir / "config.json").read_text()
    assert calls["login"] == {"key": "test-key-value", "verify": True}
    sdk.login = lambda **kwargs: False
    with pytest.raises(RuntimeError):
        start_run(cfg)
    with pytest.raises(ValueError):
        start_run(cfg, resume=True)
    with pytest.raises(ValueError):
        save_config(
            ExperimentConfig("bad", params={"api_key": "secret"}), tmp_path / "bad.json"
        )
    other = ExperimentConfig("override", wandb_entity="another-team")
    save_config(other, tmp_path / "config.json")
    assert load_config(tmp_path / "config.json").wandb_entity == "another-team"


def test_curve_head_extreme_logits_and_strict_submission():
    torch.manual_seed(10)
    values = MassCurveHead()(torch.randn(2048, 11) * 100).detach().numpy()
    validate_curves(values, atol=0)
    almost = curves_frame(["a"], [np.linspace(0, 100, 11)])
    almost.loc[0, "0.002"] = -1e-6
    with pytest.raises(ValueError):
        make_submission(almost, almost)


def test_failed_compute_stage_still_saves_usage(tmp_path):
    from soilcv.artifacts import ComputeMeter

    path = tmp_path / "compute.json"
    with pytest.raises(RuntimeError), ComputeMeter("cpu", output_path=path):
        raise RuntimeError("test failure")
    record = json.loads(path.read_text())
    assert (
        not record["completed"]
        and record["wall_seconds"] >= 0
        and record["gpu_hours"] == 0
    )


def test_layer_groups_require_explicit_mapping_and_have_no_duplicates():
    from soilcv.nn import make_param_groups

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 11)
    )
    layers = {
        name: 0 if name.startswith("0.") else 1 for name, _ in model.named_parameters()
    }
    groups = make_param_groups(
        model,
        lr=0.01,
        weight_decay=0.1,
        layer_ids=layers,
        layer_decay=0.5,
        no_weight_decay=("2.weight",),
    )
    params = [p for group in groups for p in group["params"]]
    assert len(params) == len(set(map(id, params))) == len(list(model.parameters()))
    for group in groups:
        for parameter in group["params"]:
            if parameter.ndim == 1 or parameter is model[2].weight:
                assert group["weight_decay"] == 0
            if parameter is model[0].weight:
                assert group["lr"] == pytest.approx(0.005)
    with pytest.raises(ValueError):
        make_param_groups(model, lr=0.1, layer_decay=0.5)


def test_upload_bundle_contains_cnn_code_environment_and_original_folds(tmp_path):
    from zipfile import ZipFile

    from package_soilcv import build_bundle

    path = build_bundle(tmp_path / "soilcv.zip")
    with ZipFile(path) as archive:
        names = archive.namelist()
        release = json.loads(archive.read("release.json"))
        assert release["default_wandb_entity"] == "public-team"
        assert "soilcv/assets/sample_folds.csv" in names
        assert "soilcv/tracking.py" in names and "soilcv/training.py" in names
        for name in [
            "experiments.py",
            "launcher.py",
            "analysis.py",
            "inference.py",
            "outputs.py",
            "model.py",
            "predict.py",
        ]:
            assert "soilcv/cnn/" + name in names
        assert "soilcv/cnn/assets/uv.lock" in names
        assert "soilcv/cnn/assets/pyproject.toml" in names
        assert "pyproject.toml" in names and "HYPOTHESES.md" in names
        assert "soilcv/inference/runner.py" in names
        assert "inference/runner.py" not in names
        assert not any("legacy" in name or name.endswith(".zip") for name in names)
        assert not any(
            "__pycache__" in name or name.endswith(".ipynb") for name in names
        )
        assert (
            archive.read("soilcv/assets/sample_folds.csv")
            == DEFAULT_FOLDS_PATH.read_bytes()
        )
