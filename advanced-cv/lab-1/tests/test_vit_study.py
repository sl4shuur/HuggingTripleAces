"""Small report fixtures check fold alignment, ensembles and provenance."""

import json

import numpy as np
import pandas as pd
import pytest

from soilcv.artifacts import file_hash, identity_hash, package_identity, write_json
from soilcv.checkpoints import write_candidate_index
from soilcv.curves import TARGET_COLUMNS
from soilcv.vit.analysis import collect_results


def curves(samples, value):
    frame = pd.DataFrame(
        np.tile(np.r_[np.full(10, value), 100.0], (len(samples), 1)),
        columns=TARGET_COLUMNS,
    )
    frame.insert(0, "sample_id", samples)
    return frame


@pytest.fixture
def study(tmp_path, monkeypatch):
    from soilcv.vit import experiments

    tmp_path = tmp_path / "study"
    tmp_path.mkdir()
    recipe = {"id": "test-vit", "arch": "scratch_vit", "view_mode": "multiscale"}
    monkeypatch.setattr(experiments, "DOMAIN_BASELINE", recipe["id"])
    metadata = {"run_id": "study-1", "identity": {"soilcv": package_identity()}}
    write_json(tmp_path / "metadata.json", metadata)
    write_json(tmp_path / "recipes.json", [recipe])
    samples = [f"soil-{index}" for index in range(8)]
    folds = pd.DataFrame({"sample_id": samples, "fold": np.repeat(np.arange(4), 2)})
    folds.to_csv(tmp_path / "sample_folds.csv", index=False)
    curves(samples, 20).to_json(tmp_path / "labels.json", orient="records")
    curves(["test-b", "test-a"], 0).to_json(
        tmp_path / "template.json", orient="records"
    )
    write_json(
        tmp_path / "status.json", {"failed": [], "jobs": [{"state": "complete"}] * 12}
    )
    for train_camera in (None, "Motorola", "Samsung"):
        run_name = (
            recipe["id"]
            if train_camera is None
            else recipe["id"] + "__" + train_camera.lower()
        )
        for fold in range(4):
            folder = tmp_path / "runs" / run_name / f"fold_{fold}"
            folder.mkdir(parents=True)
            heldout = folds.loc[folds.fold.eq(fold), "sample_id"].tolist()
            oof = curves(heldout, 22 + fold)
            oof["fold"] = fold
            oof.to_csv(folder / "oof.csv", index=False)
            curves(["test-a", "test-b"], 10 * (fold + 1)).to_csv(
                folder / "test_predictions.csv", index=False
            )
            motorola = curves(heldout, 22)
            motorola["device"] = "Motorola"
            samsung = curves(heldout, 26)
            samsung["device"] = "Samsung"
            camera = pd.concat([motorola, samsung], ignore_index=True)
            camera["fold"] = fold
            camera.to_csv(folder / "camera_predictions.csv", index=False)
            write_json(
                folder / "history.json",
                [{"epoch": 0, "val": {"emd": 10, "camera_disagreement": 20}}],
            )
            for name in (
                "config.json",
                "compute.json",
                "inference_check.json",
                "padding_diagnostic.json",
                "checkpoint_summary.json",
                "padding_pairs.csv",
            ):
                (folder / name).write_text("fixture")
            padding_emd = 0.01 * (fold + 1)
            write_json(
                folder / "padding_diagnostic.json",
                {
                    "padding_samples": 1,
                    "padding_max_emd": padding_emd,
                    "padding_mean_emd": padding_emd,
                    "padding_extra_pixels": 16,
                },
            )
            pd.DataFrame(
                {"sample_id": [heldout[0]], "padding_emd": [padding_emd]}
            ).to_csv(folder / "padding_pairs.csv", index=False)
            records = []
            (folder / "candidates").mkdir()
            for rank, epoch in enumerate((3, 2, 1)):
                weights = folder / "candidates" / f"epoch_{epoch:04d}.pt"
                weights.write_text(f"fixture weights {epoch}")
                predictions = folder / "candidates" / f"epoch_{epoch:04d}_oof.csv"
                candidate_oof = curves(heldout, 21 + 2 * rank + fold)
                candidate_oof.to_csv(predictions, index=False)
                if rank == 0:
                    candidate_oof["fold"] = fold
                    candidate_oof.to_csv(folder / "best_only_oof.csv", index=False)
                records.append(
                    {
                        "epoch": epoch,
                        "emd": 5 * (1 + 2 * rank + fold),
                        "camera_disagreement": 20 if rank < 2 else 25,
                        "weights": str(weights.relative_to(folder)),
                        "weights_sha256": file_hash(weights),
                        "predictions": str(predictions.relative_to(folder)),
                        "predictions_sha256": file_hash(predictions),
                    }
                )
            write_candidate_index(folder, records, 3)
            names = [
                str(path.relative_to(folder))
                for path in folder.rglob("*")
                if path.is_file()
            ]
            write_json(
                folder / "artifact_hashes.json",
                {name: file_hash(folder / name) for name in names},
            )
            write_json(
                folder / "result.json",
                {
                    "experiment": recipe["id"],
                    "fold": fold,
                    "purpose": "standard" if train_camera is None else "domain",
                    "train_camera": train_camera,
                    "run_id": metadata["run_id"],
                    "recipe_hash": identity_hash(
                        {
                            "recipe": recipe,
                            "identity": metadata["identity"],
                            "fold": fold,
                            "train_camera": train_camera,
                        }
                    ),
                    "gpu_hours": 0.01,
                    "peak_gpu_memory_mb": 100,
                    "epochs_ran": 4,
                    "best_epoch": 3,
                    "best_emd": 5 * (1 + fold),
                    "selected_epochs": [3, 2],
                    "selected_checkpoint_count": 2,
                    "run_url": f"https://wandb.test/{run_name}/{fold}",
                    "checkpoint_artifact": f"public-team/test/{run_name}-fold-{fold}:v0",
                },
            )
    return tmp_path, [recipe]


def test_study_report_scores_oof_and_averages_complete_four_fold_test(study):
    root, recipes = study
    table, predictions, domain = collect_results(root, recipes)
    assert table.cv_emd.iloc[0] == pytest.approx(17.5)
    assert table.best_only_cv_emd.iloc[0] == pytest.approx(12.5)
    assert table.selected_checkpoints_per_fold.iloc[0] == "2,2,2,2"
    assert table.padding_max_emd.iloc[0] == pytest.approx(0.04)
    assert table.padding_samples.iloc[0] == 4
    padding = pd.read_csv(root / "results" / "padding_checks.csv")
    assert padding.fold.tolist() == [0, 1, 2, 3]
    assert padding.checked_samples.tolist() == ["soil-0", "soil-2", "soil-4", "soil-6"]
    assert "## Padding check" in (root / "VIT_RESULTS.md").read_text()
    assert len(predictions["test-vit"]) == 8
    assert table.camera_disagreement.iloc[0] == pytest.approx(20)
    assert table.gpu_hours.iloc[0] == pytest.approx(0.04)
    assert domain.train_camera.tolist() == ["Motorola", "Samsung"]
    assert domain.delta_domain.tolist() == pytest.approx([20, -20])
    submission = pd.read_csv(root / "submission.csv")
    assert submission.sample_id.tolist() == ["test-b", "test-a"]
    np.testing.assert_allclose(submission[list(TARGET_COLUMNS)].iloc[:, :-1], 25)
    assert (root / "VIT_RESULTS.md").exists()
    assert len(pd.read_csv(root / "results" / "vit_checkpoints.csv")) == 4


def test_study_rejects_changed_prediction_artifact(study):
    root, recipes = study
    path = root / "runs" / "test-vit" / "fold_0" / "oof.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed run artifact"):
        collect_results(root, recipes)


def test_study_rejects_mixed_run_ids_and_incomplete_status(study):
    root, recipes = study
    path = root / "runs" / "test-vit" / "fold_0" / "result.json"
    result = json.loads(path.read_text())
    result["run_id"] = "other-study"
    write_json(path, result)
    with pytest.raises(ValueError, match="Mixed study"):
        collect_results(root, recipes)
    write_json(
        root / "status.json", {"failed": [], "jobs": [{"state": "running"}] * 12}
    )
    with pytest.raises(RuntimeError, match="Wait for every"):
        collect_results(root, recipes)


def export_fixture(root, recipes, monkeypatch):
    from soilcv.vit import outputs

    monkeypatch.setattr(outputs, "ROOT", root)
    planned = [
        (recipes[0]["id"], fold, camera)
        for camera in (None, "Motorola", "Samsung")
        for fold in range(4)
    ]
    monkeypatch.setattr(outputs, "jobs", lambda: planned)
    return outputs.export_outputs


def test_export_keeps_chosen_four_fold_csv_and_saves_source_manifest(
    study, monkeypatch
):
    root, recipes = study
    collect_results(root, recipes)
    export = export_fixture(root, recipes, monkeypatch)
    assert export() == root
    published = root.parent / "submission.csv"
    assert published.read_bytes() == (root / "submission.csv").read_bytes()
    submission = pd.read_csv(published)
    np.testing.assert_allclose(submission[list(TARGET_COLUMNS)].iloc[:, :-1], 25)
    manifest = json.loads((root / "output_manifest.json").read_text())
    assert manifest["submission.csv"]["sha256"] == file_hash(published)
    assert "source/soilcv/vit/model.py" in manifest
    assert "source/environment/runtime.json" in manifest
    assert "results/domain_results.csv" in manifest
    outputs = json.loads((root / "outputs.json").read_text())
    assert outputs["submission"] == str(published)
    assert outputs["competition_submission_made"] is False


def test_failed_export_does_not_publish_top_level_submission(study, monkeypatch):
    root, recipes = study
    collect_results(root, recipes)
    export = export_fixture(root, recipes, monkeypatch)
    (root / "runs" / "test-vit" / "fold_0" / "candidates" / "epoch_0003.pt").write_text(
        "changed"
    )
    with pytest.raises(ValueError, match="changed candidate"):
        export()
    assert not (root.parent / "submission.csv").exists()
    assert not (root / "outputs.json").exists()


def test_study_rejects_missing_selected_candidate_weight(study):
    root, recipes = study
    (root / "runs" / "test-vit" / "fold_0" / "candidates" / "epoch_0002.pt").unlink()
    with pytest.raises(ValueError, match="candidate file"):
        collect_results(root, recipes)


def test_epoch_curve_average_aligns_sample_and_camera_ids():
    from soilcv.vit.experiments import mean_curves

    first = curves(["sample-a", "sample-b"], 20)
    second = curves(["sample-b", "sample-a"], 30)
    result = mean_curves([first, second])
    assert result.sample_id.tolist() == ["sample-a", "sample-b"]
    np.testing.assert_allclose(result[list(TARGET_COLUMNS)].iloc[:, :-1], 25)
    second.loc[0, "sample_id"] = "wrong-sample"
    with pytest.raises(ValueError, match="different sample"):
        mean_curves([first, second])


def test_swin_tile_comparison_changes_schedule_without_changing_local_scale():
    from soilcv.vit.experiments import TILE_RECIPES

    frozen, constant, cosine = TILE_RECIPES
    assert frozen["tuning"] == "frozen"
    for recipe in TILE_RECIPES:
        assert recipe["view_mode"] == "tiles"
        assert recipe["tile_mm"] == 64
        assert recipe["pixels"] == 288
        assert recipe["selection"] == "emd"
        assert recipe["keep_top_k"] == 3
    difference = {key for key in constant if constant[key] != cosine[key]}
    assert difference == {"id", "schedule", "warmup"}


def test_padding_check_rejects_a_training_sample(study):
    from soilcv.vit.analysis import read_padding_check

    root, _ = study
    folder = root / "runs" / "test-vit" / "fold_0"
    pairs = pd.read_csv(folder / "padding_pairs.csv")
    pairs.loc[0, "sample_id"] = "soil-2"
    pairs.to_csv(folder / "padding_pairs.csv", index=False)
    with pytest.raises(ValueError, match="held-out samples"):
        read_padding_check(
            folder, "test-vit", 0, pd.read_csv(root / "sample_folds.csv")
        )
