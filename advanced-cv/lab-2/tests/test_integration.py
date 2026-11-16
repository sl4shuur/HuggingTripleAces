"""Контракти notebook callback і alseg 0.9.0 без навчання та downloads."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "normis.ipynb").read_text())
cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
namespace = {}
for cell_id in (
    "imports", "pixel-function", "aggregation-function", "tests",
    "b85f0a67", "22f84e14", "13c6b63a", "bc006cc5", "38ffa237", "674bd275",
    "7db30633", "6bec99ae", "cdae6c95", "312f7139",
):
    exec(compile(cells[cell_id], f"normis.ipynb:{cell_id}", "exec"), namespace)

from alseg.config import ExperimentConfig
from alseg.core.features import EncoderFeatures
from alseg.core.selection import AcquisitionMethod, SelectionRequest
from alseg.interfaces.data import ImageSource
from alseg.interfaces.encoder import FeatureSource
from alseg.nn.heads import LinearSegmentationHead


class TinyPool(ImageSource):
    image_ids = tuple(f"train_{i}" for i in range(48))

    def __len__(self):
        return len(self.image_ids)

    def get_image(self, image_id):
        raise AssertionError("Готові features не повинні декодувати RGB")


class TinyFeatures(FeatureSource):
    def __init__(self):
        ids = torch.arange(48, dtype=torch.float32)
        self.patches = torch.stack([ids, ids % 3, ids % 7, torch.ones(48)], dim=1)
        self.embeddings = torch.nn.functional.normalize(self.patches, dim=1)

    def get_features(self, source, ids):
        return EncoderFeatures(self.embeddings[ids], self.patches[ids, :, None, None])

    def describe(self):
        return {"name": "synthetic", "feature_dim": 4, "image_size": 16}


class QueriedMasks:
    def __init__(self, labeled_ids):
        self.labeled_ids = labeled_ids.copy()
        self.reads = []

    def get_mask(self, image_id):
        if image_id not in self.labeled_ids:
            raise AssertionError("Читання прихованої маски U")
        self.reads.append(image_id)
        return torch.full((16, 16), image_id % 21, dtype=torch.int64)

    def query(self, ids):
        raise AssertionError("Query належить pipeline, не notebook callback")


def config_for(method):
    return ExperimentConfig(
        device="cpu", budgets=[21, 42], tracking={"mode": "disabled"},
        encoder={"image_size": 16, "batch_size": 8}, acquisition={"method": method},
    ).model_dump(mode="json")


class TestNotebookIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(min(threads, 2))
        cls.addClassCleanup(torch.set_num_threads, threads)

    def test_selector_all_methods_and_stale_embeddings(self):
        embeddings = torch.eye(5)
        selector = namespace["NotebookSelector"](embeddings)
        for method in AcquisitionMethod:
            request = SelectionRequest(
                method=method, embeddings=embeddings, labeled_ids=[0],
                unlabeled_ids=[1, 2, 3, 4], n_select=2, seed=42,
                uncertainty=torch.tensor([0.0, 0.9, 0.2, 0.8, 0.1]), sigma=1.0,
            )
            selected = selector.select(request)
            self.assertEqual(len(set(selected)), 2)
            self.assertTrue(set(selected) <= {1, 2, 3, 4})
        request = SelectionRequest(
            method=AcquisitionMethod.MAXHERDING, embeddings=embeddings.flip(0),
            labeled_ids=[], unlabeled_ids=list(range(5)), n_select=1, seed=42, sigma=1.0,
        )
        with self.assertRaises(ValueError):
            selector.select(request)

    def test_common_random_cold_start_without_training_or_masks(self):
        selected = {}
        for method in ("random", "uncertainty", "uherding", "maxherding"):
            oracle = QueriedMasks([])
            with patch.dict(namespace, {"fit_head": Mock(side_effect=AssertionError("Unexpected training"))}):
                ids, metadata = namespace["make_acquisition"]()(
                    config=config_for(method), pool=TinyPool(), oracle=oracle,
                    encoder=TinyFeatures(), round_index=0, budget=21,
                )
            selected[method] = ids
            self.assertEqual(len(set(ids)), 21)
            self.assertEqual(oracle.reads, [])
            self.assertEqual(metadata["selection_seed"], 42)
            self.assertEqual(metadata["acquisition_budget"], 0)
        self.assertEqual(selected["random"], selected["uncertainty"])
        self.assertEqual(selected["random"], selected["uherding"])

    def check_auxiliary(self, method, fraction=0.2, void_calibration=False):
        oracle, encoder = QueriedMasks(list(range(21))), TinyFeatures()
        config = config_for(method)
        config["protocol"]["calibration_fraction"] = fraction
        fit_calls, inference_calls = [], []
        real_inference = namespace["iter_model_logits"]
        real_temperature = namespace["fit_temperature"]
        fit_ids, cal_ids = namespace["split_calibration_ids"](
            oracle.labeled_ids, 20042, fraction,
        )
        original_mask = oracle.get_mask
        if void_calibration:
            def get_mask(image_id):
                mask = original_mask(image_id)
                return torch.full_like(mask, 255) if image_id in cal_ids else mask
            oracle.get_mask = get_mask

        def fake_fit(features, targets, task_config, seed, device, logger=False):
            head = LinearSegmentationHead(4)
            with torch.no_grad():
                for parameter in head.parameters():
                    parameter.zero_()
            fit_calls.append((features, targets, seed, device, logger, head))
            return SimpleNamespace(head=head)

        def audited_inference(head, pool, encoder, ids, *args):
            inference_calls.append((head, ids.copy()))
            return real_inference(head, pool, encoder, ids, *args)

        temperature = Mock(wraps=real_temperature)
        with patch.dict(namespace, {
            "fit_head": fake_fit, "iter_model_logits": audited_inference,
            "fit_temperature": temperature,
        }):
            selected, metadata = namespace["make_acquisition"]()(
                config=config, pool=TinyPool(), oracle=oracle,
                encoder=encoder, round_index=1, budget=42,
            )
        self.assertEqual(len(fit_calls), 1)
        features, targets, seed, device, logger, head = fit_calls[0]
        torch.testing.assert_close(features, encoder.patches[fit_ids, :, None, None])
        torch.testing.assert_close(targets[:, 0, 0], torch.tensor(fit_ids) % 21)
        self.assertEqual((seed, device, logger), (10042, "cpu", False))
        self.assertEqual(metadata["calibration_fit_ids"], fit_ids)
        self.assertEqual(metadata["calibration_ids"], cal_ids)
        self.assertFalse(set(fit_ids) & set(cal_ids))
        self.assertTrue(all(used_head is head for used_head, _ in inference_calls))
        self.assertEqual(inference_calls[-1][1], list(range(48)))
        self.assertEqual(temperature.call_count, bool(cal_ids))
        if cal_ids:
            self.assertEqual(inference_calls[0][1], cal_ids)
        self.assertTrue(set(oracle.reads) <= set(range(21)))
        self.assertEqual(len(set(selected)), 21)
        self.assertTrue(set(selected) <= set(range(21, 48)))
        self.assertEqual(oracle.labeled_ids, list(range(21)))
        self.assertEqual(metadata["selection_seed"], 43)
        self.assertEqual(metadata["acquisition_budget"], 21)
        return metadata

    def test_same_auxiliary_head_for_calibration_and_scoring(self):
        for method in ("uncertainty", "uherding"):
            with self.subTest(method=method):
                metadata = self.check_auxiliary(method)
                self.assertEqual(len(metadata["calibration_fit_ids"]), 17)
                self.assertEqual(len(metadata["calibration_ids"]), 4)
                self.assertIn(metadata["tau"], namespace["TEMPERATURE_GRID"])
                self.assertIsNone(metadata["fallback"])

    def test_tau_fallbacks_are_explicit(self):
        for fraction, void, reason in (
            (0, False, "no_calibration_images"),
            (0.2, True, "no_valid_calibration_pixels"),
        ):
            metadata = self.check_auxiliary("uncertainty", fraction, void)
            self.assertEqual(metadata["tau"], 1.0)
            self.assertEqual(metadata["ece_by_temperature"], {})
            self.assertEqual(metadata["fallback"], reason)

    def test_maxherding_round_needs_no_masks_or_head(self):
        oracle = QueriedMasks(list(range(21)))
        with patch.dict(namespace, {"fit_head": Mock(side_effect=AssertionError("Unexpected training"))}):
            selected, metadata = namespace["make_acquisition"]()(
                config=config_for("maxherding"), pool=TinyPool(), oracle=oracle,
                encoder=TinyFeatures(), round_index=1, budget=42,
            )
        self.assertEqual(oracle.reads, [])
        self.assertEqual(len(selected), 21)
        self.assertGreater(metadata["sigma"], 0)
        self.assertIsNone(metadata["tau"])

    def test_cached_embedding_load_does_not_scan_patch_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            expected = TinyFeatures().embeddings
            np.save(root / "train/embeddings.npy", expected.numpy())
            config = config_for("maxherding")
            config["encoder"]["features_dir"] = str(root)
            encoder = TinyFeatures()
            encoder.get_features = Mock(wraps=encoder.get_features)
            with patch.dict(namespace, {"iter_features": Mock(side_effect=AssertionError("Unexpected full patch scan"))}):
                embeddings = namespace["extract_pool_embeddings"](config, TinyPool(), encoder)
            torch.testing.assert_close(embeddings, expected)
            self.assertEqual(encoder.get_features.call_count, 1)
            self.assertEqual(encoder.get_features.call_args.args[1], [0])

    def test_notebook_callback_runs_inside_shared_pipeline_without_training(self):
        from alseg import pipeline
        from alseg.core.pool import PoolManifest

        for method in ("random", "uncertainty", "maxherding", "uherding"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                pool, oracle, encoder = TinyPool(), QueriedMasks([]), TinyFeatures()
                oracle.query = lambda ids: setattr(oracle, "labeled_ids", oracle.labeled_ids + ids)
                manifest = PoolManifest(list(pool.image_ids), ["validation"])
                experiment = ExperimentConfig.model_validate(config_for(method))
                config = experiment.model_dump(mode="json")
                config["data"]["voc_root"] = directory
                reports = []

                def fake_report(run_config, *args):
                    reports.append(run_config)
                    return {"budget": run_config["budget"], "miou": 0.0}

                with patch.multiple(
                    pipeline,
                    load_manifest=Mock(return_value=manifest),
                    VocImages=Mock(return_value=pool),
                    VocOracle=Mock(return_value=oracle),
                    VocEvaluation=Mock(),
                    load_encoder=Mock(return_value=encoder),
                    _run_round=Mock(side_effect=fake_report),
                    log_experiment_history=Mock(),
                ), patch.dict(namespace, {
                    "fit_head": Mock(side_effect=lambda *a, **k: SimpleNamespace(
                        head=LinearSegmentationHead(4),
                    )),
                }):
                    results = namespace["run_experiment"](
                        config, Path(directory) / "run", "cpu", experiment.tracking,
                        acquisition=namespace["make_acquisition"](),
                    )
                self.assertEqual([row["budget"] for row in results], [21, 42])
                self.assertEqual([len(report["fit_ids"]) for report in reports], [21, 42])
                self.assertTrue(all(report["calibration_ids"] == [] for report in reports))
                self.assertEqual(len(set(oracle.labeled_ids)), 42)
                selection = json.loads((Path(directory) / "run/selections.json").read_text())
                self.assertEqual(selection[-1]["acquisition"]["method"], method)
                self.assertEqual(selection[-1]["acquisition"]["acquisition_budget"], 21)

    def test_notebook_run_cell_is_disabled_by_default(self):
        scope = dict(namespace)
        exec(cells["db3e36f2"], scope)
        self.assertFalse(scope["RUN_EXPERIMENT"])
        self.assertEqual(scope["experiment"].task.epochs, 2)
        self.assertEqual(scope["experiment"].acquisition.method, AcquisitionMethod.UHERDING)
        with patch.dict(scope, {"run_experiment": Mock(side_effect=AssertionError("Unexpected experiment"))}):
            exec(cells["ead57a63"], scope)


if __name__ == "__main__":
    unittest.main(verbosity=2)
