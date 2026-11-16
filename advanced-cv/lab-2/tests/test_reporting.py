"""Таблиці та графіки notebook на малих fixtures, без training і GPU."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CELLS = {
    cell["id"]: "".join(cell["source"])
    for cell in json.loads((ROOT / "normis.ipynb").read_text())["cells"]
}
NS = {}
for cell_id in (
    "imports",
    "pixel-function",
    "aggregation-function",
    "7db30633",
    "cdae6c95",
    "5279c8ea",
    "db3e36f2",
    "134ad660",
    "c550166e",
    "68f1110b",
    "c3c6f48b",
    "2bee71a8",
):
    exec(compile(CELLS[cell_id], f"normis.ipynb:{cell_id}", "exec"), NS)

plt = NS["plt"]
ENCODER = NS["EncoderName"].DINO_VITB16
BUDGETS = NS["BENCHMARK_BUDGETS"]


def write_run(root, encoder=ENCODER, method="random", rounds=None):
    full = method == "full_data"
    config = NS["benchmark_config"](encoder, "random" if full else method, full)
    directory = root / encoder.value / method
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(config.model_dump_json())
    budgets = config.budgets if rounds is None else config.budgets[:rounds]
    rows = []
    for index, budget in enumerate(budgets):
        offset = 0 if full else NS["BENCHMARK_METHODS"].index(method) * 0.01
        miou = 0.9 if full else 0.2 + 0.6 * (budget - 21) / (672 - 21) + offset
        rows.append(
            {
                "encoder": encoder.value,
                "method": config.acquisition.method.value,
                "budget": budget,
                "round": index,
                "seed": 42,
                "epochs": 20,
                "smoke": False,
                "val_images": 1449,
                "n_fit": budget,
                "n_calibration": 0,
                "miou": miou,
                "pixel_accuracy": 0.9,
                "mean_class_accuracy": 0.7,
                "val_loss": 1.0 - miou / 2,
                "acquisition_budget": 0 if not index else budgets[index - 1],
                "sigma_used": 0.8 if method in ("maxherding", "uherding") else np.nan,
                "tau_used": 1.0 if method == "random" else 2.0 if index else np.nan,
                "selection_seconds": 1.0,
                "feature_seconds": 0.1,
                "train_seconds": 2.0,
                "eval_seconds": 0.5,
            }
        )
    for index, budget in enumerate(budgets):
        round_dir = directory / f"round-{index:02d}-budget-{budget}"
        round_dir.mkdir()
        (round_dir / "config.json").write_text(
            json.dumps(
                {
                    "encoder_identity": {
                        "name": encoder.value,
                        "image_size": 448,
                        "weights_sha256": NS["VIT16_WEIGHTS"][encoder]["sha256"],
                    },
                    "runtime": {"precision": "32-true"},
                    "task": config.task.model_dump(),
                }
            )
        )
        pd.DataFrame({
            "epoch": range(20),
            "train/loss": np.linspace(2.0, 0.5, 20),
            "train/miou": np.linspace(0.1, 0.8, 20),
            "train/pixel_accuracy": np.linspace(0.5, 0.9, 20),
        }).to_csv(round_dir / "training.csv", index=False)
    pd.DataFrame(rows).to_csv(directory / "results.csv", index=False)
    (directory / "state.json").write_text(
        json.dumps({"status": "completed", "budget": budgets[-1]})
    )
    ids = {
        "pool_ids": [f"train_{i}" for i in range(1464)],
        "val_ids": [f"val_{i}" for i in range(1449)],
    }
    digest = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps({**ids, "sha256": digest}))
    return directory, config


class TestReporting(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_grid_has_eight_trajectories_and_two_full_data_runs(self):
        plan = list(NS["benchmark_runs"](Path("unused")))
        self.assertEqual(len(plan), 10)
        self.assertEqual(sum(method == "full_data" for _, _, method in plan), 2)
        for _, config, method in plan:
            self.assertFalse(config.smoke)
            self.assertEqual(config.task.epochs, 20)
            self.assertEqual(
                config.budgets, [1464] if method == "full_data" else list(BUDGETS)
            )
            self.assertEqual(config.acquisition.uncertainty_method, "margin")
            self.assertFalse(str(config.paths.cache_dir).startswith("/kaggle"))

    def test_empty_grid_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "missing"
            data, status = NS["collect_benchmark"](root)
            self.assertTrue(data.empty)
            self.assertEqual(status.status.tolist(), ["missing"] * 10)
            self.assertFalse(root.exists())

    def test_aubc_gains_and_full_data_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for method in (*NS["BENCHMARK_METHODS"], "full_data"):
                write_run(root, method=method)
            data, _ = NS["collect_benchmark"](root)
            detailed, summary = NS["benchmark_tables"](data)
            self.assertAlmostEqual(
                summary.loc[(ENCODER.value, "random"), "normalized_AUBC"], 0.5
            )
            gain = detailed.loc[detailed.method == "uherding", "gain_over_random_pp"]
            np.testing.assert_allclose(gain, 3.0)
            self.assertTrue(
                np.isnan(summary.loc[(ENCODER.value, "full_data"), "normalized_AUBC"])
            )
            self.assertTrue(
                detailed.loc[detailed.method == "full_data", "gain_over_random_pp"]
                .isna()
                .all()
            )
            self.assertTrue(data.loc[data.method == "random", "tau_used"].isna().all())

    def test_partial_curve_has_no_aubc_or_invented_random_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            write_run(Path(temp), method="uherding", rounds=2)
            data, status = NS["collect_benchmark"](Path(temp))
            detailed, summary = NS["benchmark_tables"](data)
            self.assertTrue(summary.normalized_AUBC.isna().all())
            self.assertTrue(detailed.gain_over_random_pp.isna().all())
            self.assertEqual(
                status.loc[status.method == "uherding", "status"].iloc[0], "partial"
            )

    def test_rejects_recipe_seed_smoke_and_duplicate_budgets(self):
        for change in ("recipe", "weights", "seed", "smoke", "duplicate", "nan", "val_loss"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                directory, config = write_run(Path(temp))
                if change == "recipe":
                    actual = json.loads((directory / "config.json").read_text())
                    actual["task"]["epochs"] = 2
                    (directory / "config.json").write_text(json.dumps(actual))
                elif change == "weights":
                    path = directory / "round-00-budget-21/config.json"
                    actual = json.loads(path.read_text())
                    actual["encoder_identity"]["weights_sha256"] = "wrong-checkpoint"
                    path.write_text(json.dumps(actual))
                else:
                    data = pd.read_csv(directory / "results.csv")
                    if change == "seed":
                        data.loc[0, "seed"] = 99
                    elif change == "smoke":
                        data.loc[0, "smoke"] = True
                    elif change == "nan":
                        data.loc[0, "miou"] = np.nan
                    elif change == "val_loss":
                        data.loc[0, "val_loss"] = np.nan
                    else:
                        data.loc[1, "budget"] = 21
                    data.to_csv(directory / "results.csv", index=False)
                with self.assertRaises(ValueError):
                    NS["read_benchmark_run"](directory, config, "random")

    def test_different_manifests_are_not_merged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root)
            directory, _ = write_run(root, method="uherding")
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["pool_ids"] = manifest["pool_ids"][::-1]
            ids = {k: manifest[k] for k in ("pool_ids", "val_ids")}
            manifest["sha256"] = hashlib.sha256(
                json.dumps(ids, sort_keys=True).encode()
            ).hexdigest()
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                NS["collect_benchmark"](root)

    def test_mask_boundary_cost_excludes_void(self):
        counts, length = NS["mask_summary"](
            np.array([[0, 1, 255], [0, 1, 1]], dtype=np.uint8)
        )
        self.assertEqual(length, 2)
        np.testing.assert_array_equal(counts[:2], [2, 3])
        self.assertEqual(counts.sum(), 5)
        with self.assertRaises(ValueError):
            NS["mask_summary"](np.array([[21]], dtype=np.uint8))

    def test_snapshots_use_u_only_and_do_not_read_masks(self):
        from alseg.core.predictions import LogitBatch

        logits = torch.zeros(4, 21, 2, 3)
        logits[:, 0] = torch.tensor([4.0, 3.0, 2.0, 1.0])[:, None, None]
        scores = NS["mean_image_uncertainty"](
            NS["pixel_uncertainty"](logits, "margin"),
            torch.ones(4, 2, 3, dtype=torch.bool),
        )
        captured = []

        def predictions(head, pool, encoder, ids, image_size, batch_size, device):
            captured.extend(ids)
            yield LogitBatch(
                ids, logits[ids], torch.ones(len(ids), 2, 3, dtype=torch.bool)
            )

        with tempfile.TemporaryDirectory() as temp, patch.dict(
            NS, {"iter_model_logits": predictions}
        ):
            name = NS["save_uncertainty_snapshot"](
                Path(temp),
                1,
                scores,
                [0],
                object(),
                range(4),
                object(),
                1.0,
                "margin",
                3,
                4,
                "cpu",
            )
            self.assertNotIn(0, captured)
            self.assertEqual(set(captured), {1, 2, 3})
            with np.load(Path(temp) / name, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["ids"], captured)
                np.testing.assert_allclose(
                    saved["scores"], saved["uncertainty"].mean(axis=(1, 2))
                )
                self.assertEqual(saved["predictions"].dtype, np.uint8)
            with self.assertRaises(FileExistsError):
                NS["save_uncertainty_snapshot"](
                    Path(temp),
                    1,
                    scores,
                    [0],
                    object(),
                    range(4),
                    object(),
                    1.0,
                    "margin",
                    3,
                    4,
                    "cpu",
                )

    def test_training_history_preserves_head_boundaries_and_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, _ = write_run(Path(temp))
            data, _ = NS["collect_benchmark"](Path(temp))
            history = NS["collect_training_history"](data)
            self.assertEqual(len(history), 6 * 20)
            self.assertEqual(history.epoch.tolist(), list(range(1, 21)) * 6)
            self.assertEqual(history.budget.unique().tolist(), list(BUDGETS))
            self.assertEqual(history.method.unique().tolist(), ["random"])
            self.assertEqual(NS["collect_training_history"](pd.DataFrame()).shape[0], 0)
            path = directory / "round-00-budget-21/training.csv"
            broken = pd.read_csv(path).iloc[:-1]
            broken.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                NS["collect_training_history"](data)

    def test_training_history_rejects_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, _ = write_run(Path(temp))
            data, _ = NS["collect_benchmark"](Path(temp))
            path = directory / "round-00-budget-21/training.csv"
            broken = pd.read_csv(path)
            broken.loc[3, "train/loss"] = np.nan
            broken.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                NS["collect_training_history"](data)

    def test_figure_rendering_and_export_in_both_themes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for encoder in NS["BENCHMARK_ENCODERS"]:
                for method in (*NS["BENCHMARK_METHODS"], "full_data"):
                    write_run(root, encoder, method)
            data, status = NS["collect_benchmark"](root)
            self.assertTrue((status.status == "complete").all())
            self.assertEqual(len(data), 50)
            history = NS["collect_training_history"](data)
            self.assertEqual(len(history), 1000)
            for theme in ("light", "dark"):
                for metric in ("train/loss", "train/miou"):
                    figure = NS["plot_training_history"](history, ENCODER.value, metric, theme)
                    figure.canvas.draw()
                    self.assertEqual(len(figure.axes), 6)
                    self.assertTrue(all(len(ax.lines) == 4 for ax in figure.axes))
                    self.assertEqual(figure.axes[0].lines[0].get_xdata().tolist(), list(range(1, 21)))
                    plt.close(figure)
                for name in ("plot_learning_curves", "plot_adaptive_parameters",
                             "plot_validation_loss", "plot_full_data_history"):
                    figure = NS[name](history if name == "plot_full_data_history" else data, theme)
                    figure.canvas.draw()
                    if name == "plot_learning_curves":
                        self.assertEqual(len(figure.axes[0].get_xticks(minor=True)), 0)
                    path = root / f"{name}-{theme}.png"
                    figure.savefig(path, dpi=70)
                    self.assertGreater(path.stat().st_size, 1000)
                    plt.close(figure)

    def test_posthoc_tables_and_image_figures_on_tiny_voc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            voc = root / "VOC"
            stems = ["train_0", "train_1", "train_2"]
            for folder in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
                (voc / folder).mkdir(parents=True)
            for index, stem in enumerate([*stems, "val_0"]):
                Image.fromarray(
                    np.full((8, 8, 3), 70 + index * 20, dtype=np.uint8)
                ).save(voc / "JPEGImages" / f"{stem}.jpg")
                Image.fromarray(
                    np.array([[0, index % 3], [0, index % 3]], dtype=np.uint8)
                ).save(voc / "SegmentationClass" / f"{stem}.png")
            (voc / "ImageSets/Segmentation/train.txt").write_text("\n".join(stems))
            (voc / "ImageSets/Segmentation/val.txt").write_text("val_0")
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(
                json.dumps({"pool_ids": stems, "val_ids": ["val_0"]})
            )
            selections = [
                {"budget": 2, "ids": [0, 1], "new_ids": [0, 1]},
                {"budget": 3, "ids": [0, 1, 2], "new_ids": [2]},
            ]
            (run / "selections.json").write_text(json.dumps(selections))
            data = pd.DataFrame(
                {
                    "encoder": [ENCODER.value] * 2,
                    "method": ["random"] * 2,
                    "budget": [2, 3],
                    "miou": [0.3, 0.4],
                    "run_dir": [str(run)] * 2,
                }
            )
            cost, distribution = NS["posthoc_tables"](data, voc)
            self.assertEqual(cost.boundary_edges.tolist(), [2, 4])
            final = distribution.loc[
                (distribution.method == "random") & (distribution.budget == 3)
            ]
            self.assertAlmostEqual(final.pixel_share.sum(), 1)
            self.assertEqual(
                final.loc[final["class"] == "background", "image_presence_pct"].iloc[0],
                100,
            )
            for figure in (
                NS["plot_annotation_cost"](cost),
                NS["plot_class_distribution"](
                    distribution.loc[distribution.budget == 3].assign(budget=672)
                ),
                NS["plot_first_batch"](voc, stems, [0, 1, 2], "SYNTHETIC TEST ONLY"),
            ):
                figure.canvas.draw()
                plt.close(figure)
            snapshot = root / "snapshot.npz"
            np.savez_compressed(
                snapshot,
                ids=np.array([0, 1]),
                scores=np.array([0.1, 0.9]),
                predictions=np.array([[[0, 1], [0, 1]]] * 2),
                uncertainty=np.array([[[0.1] * 2] * 2, [[0.9] * 2] * 2]),
                most_uncertain=np.array([False, True]),
                temperature=np.array(2.0),
            )
            figure = NS["plot_uncertainty_snapshot"](
                snapshot, voc, stems, "SYNTHETIC TEST ONLY"
            )
            figure.canvas.draw()
            plt.close(figure)

    def test_report_cell_exports_tables_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for encoder in NS["BENCHMARK_ENCODERS"]:
                for method in (*NS["BENCHMARK_METHODS"], "full_data"):
                    write_run(root, encoder, method)
            with patch.dict(
                NS, {"BENCHMARK_ROOT": root, "display": Mock()}
            ), patch.object(plt, "show"):
                exec(compile(CELLS["ce048116"], "report-cell", "exec"), NS)
            for name in (
                "per-round.csv",
                "summary.csv",
                "regimes.csv",
                "uherding-comparison.csv",
                "timings.csv",
                "learning-curves.png",
                "adaptive-parameters.png",
                "training-history.csv",
                "validation-loss.png",
                "full-data-training.png",
                "training-loss-dino_vitb16.png",
                "training-miou-dino_vitb16.png",
                "training-loss-ibot_vitb16.png",
                "training-miou-ibot_vitb16.png",
            ):
                self.assertTrue((root / "report" / name).is_file(), name)
            result = pd.read_csv(root / "report/per-round.csv")
            self.assertEqual(len(result), 50)
            history = pd.read_csv(root / "report/training-history.csv")
            self.assertEqual(len(history), 1000)

    def test_partial_run_blocks_benchmark_before_any_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root / "runs", method="uherding", rounds=2)
            features = root / "features"
            features.mkdir()
            (features / "extraction.json").write_text(
                '{"complete": true, "reusable": true}'
            )
            runner = Mock(side_effect=AssertionError("Unexpected training"))
            with patch.dict(
                NS,
                {
                    "BENCHMARK_ROOT": root / "runs",
                    "FEATURES_ROOT": features,
                    "BENCHMARK_DEVICE": "cpu",
                    "RUN_BENCHMARK": True,
                    "run_experiment": runner,
                },
            ):
                with self.assertRaises(RuntimeError):
                    exec(compile(CELLS["578391c6"], "benchmark-cell", "exec"), NS)
            runner.assert_not_called()

    def test_default_run_cells_do_not_start_training_or_posthoc(self):
        for cell_id in ("ead57a63", "578391c6", "77a55da2"):
            guard = {
                "RUN_EXPERIMENT": False,
                "RUN_BENCHMARK": False,
                "RUN_POSTHOC": False,
            }
            exec(compile(CELLS[cell_id], cell_id, "exec"), guard)


if __name__ == "__main__":
    unittest.main()
