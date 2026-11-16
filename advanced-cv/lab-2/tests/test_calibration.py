"""CPU-тести image-level holdout, глобального ECE та temperature search."""

import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "normis.ipynb").read_text())
cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
namespace = {}
for cell_id in ("imports", "pixel-function", "bc006cc5", "38ffa237", "674bd275"):
    exec(compile(cells[cell_id], f"normis.ipynb:{cell_id}", "exec"), namespace)

split_calibration_ids = namespace["split_calibration_ids"]
ece_bin_sums = namespace["ece_bin_sums"]
expected_calibration_error = namespace["expected_calibration_error"]
fit_temperature = namespace["fit_temperature"]
pixel_uncertainty = namespace["pixel_uncertainty"]


def overconfident_batch():
    probabilities = torch.full((21,), 0.01 / 20)
    probabilities[0] = 0.99
    logits = probabilities.log()[None, :, None, None].expand(2, -1, 1, 2)
    targets = torch.tensor([[[0, 0]], [[0, 1]]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    return logits, targets, valid


class TestCalibrationSplit(unittest.TestCase):
    def test_budget_schedule_disjoint_partition(self):
        for budget in (21, 42, 84, 168, 336, 672):
            labeled = list(range(1000, 1000 + budget))
            fit, calibration = split_calibration_ids(labeled, seed=20042)
            self.assertEqual(len(calibration), max(4, int(0.2 * budget)))
            self.assertFalse(set(fit) & set(calibration))
            self.assertEqual(sorted(fit + calibration), labeled)
            self.assertEqual(fit, sorted(fit))
            self.assertEqual(calibration, sorted(calibration))

    def test_reproducibility_order_and_rng_isolation(self):
        labeled = list(range(21))
        original = labeled.copy()
        rng_state = torch.random.get_rng_state().clone()
        expected = split_calibration_ids(labeled, seed=20042)
        self.assertEqual(split_calibration_ids(labeled[::-1], seed=20042), expected)
        self.assertNotEqual(split_calibration_ids(labeled, seed=20043), expected)
        self.assertEqual(labeled, original)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
        self.assertEqual(tuple(map(len, expected)), (17, 4))

    def test_small_labeled_set_uses_shared_capped_holdout(self):
        from alseg.core.calibration import split_labeled_ids

        self.assertEqual(split_calibration_ids([], seed=42), ([], []))
        for labeled in ([8], [8, 2, 4, 6], list(range(21))):
            self.assertEqual(
                split_calibration_ids(labeled, seed=42),
                split_labeled_ids(labeled, fraction=0.2, min_images=4, seed=42),
            )
        self.assertEqual(
            split_calibration_ids([8, 2, 4, 6], seed=42, calibration_fraction=0),
            ([2, 4, 6, 8], []),
        )

    def test_invalid_split_inputs(self):
        for labeled in ([0, 0], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                split_calibration_ids(labeled, seed=42)
        for fraction in (-0.1, 1.0, float("nan")):
            with self.assertRaises(ValueError):
                split_calibration_ids([0, 1], seed=42, calibration_fraction=fraction)
        for minimum in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                split_calibration_ids([0, 1], seed=42, min_calibration_images=minimum)
        for seed in (-1, True, 2**63):
            with self.assertRaises(ValueError):
                split_calibration_ids([0, 1], seed=seed)


class TestECE(unittest.TestCase):
    def test_known_bin_statistics(self):
        confidence = torch.tensor([0.9, 0.85, 0.65, 0.55])
        probabilities = torch.stack([confidence, 1 - confidence])
        logits = probabilities.log()[None, :, None, :].requires_grad_()
        targets = torch.tensor([[[0, 1, 0, 1]]])
        valid = torch.ones_like(targets, dtype=torch.bool)
        stats = ece_bin_sums(logits, targets, valid)
        torch.testing.assert_close(stats[0], torch.tensor([0, 0, 1, 1, 2]).double())
        self.assertAlmostEqual(expected_calibration_error(stats), 0.4125, places=6)
        self.assertEqual(stats.dtype, torch.float64)
        self.assertFalse(stats.requires_grad)

    def test_confidence_one_and_empty_bins(self):
        logits = torch.tensor([[[[10000.0, 10000.0]], [[-10000.0, -10000.0]]]])
        targets = torch.zeros(1, 1, 2, dtype=torch.int64)
        valid = torch.ones_like(targets, dtype=torch.bool)
        stats = ece_bin_sums(logits, targets, valid)
        torch.testing.assert_close(stats[0], torch.tensor([0, 0, 0, 0, 2]).double())
        self.assertEqual(expected_calibration_error(stats), 0.0)
        self.assertEqual(expected_calibration_error(ece_bin_sums(logits, targets + 1, valid)), 1.0)

    def test_void_and_geometry_pixels_are_excluded(self):
        logits = torch.tensor([[[[100.0, 100.0, 100.0]], [[-100.0, -100.0, -100.0]]]])
        targets = torch.tensor([[[0, 255, 1]]])
        valid = torch.tensor([[[True, True, False]]])
        stats = ece_bin_sums(logits, targets, valid)
        self.assertEqual(stats[0].sum().item(), 1)
        self.assertEqual(expected_calibration_error(stats), 0.0)

    def test_accumulation_is_not_average_batch_ece(self):
        probabilities = torch.tensor([0.75, 0.25])
        logits = probabilities.log()[None, :, None, None].expand(4, -1, 1, 1)
        targets = torch.tensor([[[1]], [[0]], [[0]], [[0]]])
        valid = torch.ones_like(targets, dtype=torch.bool)
        first = ece_bin_sums(logits[:1], targets[:1], valid[:1])
        rest = ece_bin_sums(logits[1:], targets[1:], valid[1:])
        together = ece_bin_sums(logits, targets, valid)
        torch.testing.assert_close(first + rest, together)
        self.assertAlmostEqual(expected_calibration_error(together), 0.0, places=6)
        wrong_average = (expected_calibration_error(first) + 3 * expected_calibration_error(rest)) / 4
        self.assertAlmostEqual(wrong_average, 0.375, places=6)

    def test_ece_undefined_without_valid_pixels(self):
        logits, targets, valid = overconfident_batch()
        for mask, labels in (
            (torch.zeros_like(valid), targets),
            (valid, torch.full_like(targets, 255)),
        ):
            stats = ece_bin_sums(logits, labels, mask)
            torch.testing.assert_close(stats, torch.zeros(3, 5, dtype=torch.float64))
            with self.assertRaises(ValueError):
                expected_calibration_error(stats)

    def test_invalid_temperature_bins_and_targets(self):
        logits, targets, valid = overconfident_batch()
        for temperature in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                ece_bin_sums(logits, targets, valid, temperature=temperature)
        for n_bins in (0, -1, True):
            with self.assertRaises(ValueError):
                ece_bin_sums(logits, targets, valid, n_bins=n_bins)
        for labels in (targets.float(), targets[:, :, :1], torch.full_like(targets, -1), torch.full_like(targets, 21)):
            with self.assertRaises(ValueError):
                ece_bin_sums(logits, labels, valid)
        with self.assertRaises(ValueError):
            ece_bin_sums(logits, targets, valid.float())
        with self.assertRaises(ValueError):
            ece_bin_sums(torch.full_like(logits, float("nan")), targets, valid)


class TestTemperatureSearch(unittest.TestCase):
    def test_overconfident_example_and_unchanged_predictions(self):
        logits, targets, valid = overconfident_batch()
        original = logits.clone(), targets.clone(), valid.clone()
        temperature, errors = fit_temperature([(logits, targets, valid)])
        self.assertEqual(temperature, 2.0)
        # Bin sums — float64, але confidence надходить із float32 softmax.
        self.assertAlmostEqual(errors[1.0], 0.24, delta=1e-6)
        self.assertLess(errors[temperature], errors[1.0])
        torch.testing.assert_close(logits.argmax(dim=1), (logits / temperature).argmax(dim=1))
        self.assertGreater(
            pixel_uncertainty(logits / temperature, "margin").mean().item(),
            pixel_uncertainty(logits, "margin").mean().item(),
        )
        for before, after in zip(original, (logits, targets, valid)):
            torch.testing.assert_close(before, after)

    def test_one_pass_generator_matches_combined_batch(self):
        logits, targets, valid = overconfident_batch()
        visited = []

        def batches():
            for i in range(2):
                visited.append(i)
                yield logits[i:i + 1], targets[i:i + 1], valid[i:i + 1]

        expected = fit_temperature([(logits, targets, valid)])
        actual = fit_temperature(batches())
        self.assertEqual(visited, [0, 1])
        self.assertEqual(actual[0], expected[0])
        for temperature in expected[1]:
            self.assertAlmostEqual(actual[1][temperature], expected[1][temperature], places=12)

    def test_variable_spatial_sizes(self):
        logits, targets, valid = overconfident_batch()
        split_pixels = [
            (logits[:1], targets[:1], valid[:1]),
            (logits[1:, :, :, :1], targets[1:, :, :1], valid[1:, :, :1]),
            (logits[1:, :, :, 1:], targets[1:, :, 1:], valid[1:, :, 1:]),
        ]
        actual = fit_temperature(iter(split_pixels))
        expected = fit_temperature([(logits, targets, valid)])
        self.assertEqual(actual[0], expected[0])
        for temperature in expected[1]:
            self.assertAlmostEqual(actual[1][temperature], expected[1][temperature], places=12)

    def test_ties_follow_grid_order(self):
        logits = torch.zeros(1, 2, 1, 2)
        targets = torch.zeros(1, 1, 2, dtype=torch.int64)
        valid = torch.ones_like(targets, dtype=torch.bool)
        temperature, errors = fit_temperature([(logits, targets, valid)], temperatures=(2.0, 1.0, 0.5))
        self.assertEqual(temperature, 2.0)
        self.assertEqual(errors, {2.0: 0.5, 1.0: 0.5, 0.5: 0.5})

    def test_no_calibration_data_uses_explicit_fallback(self):
        self.assertEqual(fit_temperature(iter(())), (1.0, {}))
        logits, targets, valid = overconfident_batch()
        self.assertEqual(fit_temperature([(logits, torch.full_like(targets, 255), valid)]), (1.0, {}))

    def test_invalid_temperature_grid(self):
        for grid in ((), (0.0,), (-1.0,), (True,), (float("nan"),), (float("inf"),), (1.0, 1.0)):
            with self.assertRaises(ValueError):
                fit_temperature([], temperatures=grid)
        with self.assertRaises(ValueError):
            fit_temperature([], n_bins=0)

    def test_alseg_calibration_batch_contract(self):
        sys.path.insert(0, str(ROOT / "src/source"))
        try:
            from alseg.core.calibration import CalibrationBatch
            from alseg.core.predictions import LogitBatch
        finally:
            sys.path.pop(0)

        fit_ids, calibration_ids = split_calibration_ids(list(range(21)), seed=20042)
        logits, targets, valid = overconfident_batch()
        batch = CalibrationBatch(LogitBatch(calibration_ids[:2], logits, valid), targets)
        self.assertFalse(set(batch.predictions.ids) & set(fit_ids))
        result = fit_temperature([(
            batch.predictions.logits, batch.targets, batch.predictions.valid_pixels,
        )])
        self.assertEqual(result[0], 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
