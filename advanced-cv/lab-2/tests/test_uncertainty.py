"""CPU-тести функцій з normis.ipynb без запуску експериментальних комірок.

Запуск із кореня проєкту: python -m unittest discover -s tests -v
"""

import json
import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "normis.ipynb").read_text())
cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
namespace = {}
# Виконуємо лише imports і визначення, ніколи training або демонстрації.
for cell_id in ("imports", "pixel-function", "aggregation-function", "tests"):
    exec(compile(cells[cell_id], f"normis.ipynb:{cell_id}", "exec"), namespace)

pixel_uncertainty = namespace["pixel_uncertainty"]
mean_image_uncertainty = namespace["mean_image_uncertainty"]
uncertainty_top_b = namespace["uncertainty_top_b"]
NUM_CLASSES = namespace["NUM_CLASSES"]
SEED = namespace["SEED"]


class TestUncertainty(unittest.TestCase):
    def test_uniform_predictions(self):
        logits = torch.zeros(2, NUM_CLASSES, 2, 3)
        for method in ("margin", "entropy"):
            torch.testing.assert_close(
                pixel_uncertainty(logits, method), torch.ones(2, 2, 3)
            )

    def test_confident_extreme_logits(self):
        logits = torch.full((1, NUM_CLASSES, 2, 3), -10_000.0)
        logits[:, 0] = 10_000.0
        for method in ("margin", "entropy"):
            torch.testing.assert_close(
                pixel_uncertainty(logits, method), torch.zeros(1, 2, 3)
            )

    def test_known_distribution(self):
        probabilities = torch.tensor([0.7, 0.2, 0.1])
        logits = probabilities.log().reshape(1, 3, 1, 1)
        entropy = -sum(p * math.log(p) for p in (0.7, 0.2, 0.1)) / math.log(3)
        self.assertAlmostEqual(
            pixel_uncertainty(logits, "margin").item(), 0.5, places=6
        )
        self.assertAlmostEqual(
            pixel_uncertainty(logits, "entropy").item(), entropy, places=6
        )

    def test_two_competing_classes(self):
        logits = torch.full((1, NUM_CLASSES, 1, 1), -100.0)
        logits[:, :2] = 0.0
        self.assertAlmostEqual(
            pixel_uncertainty(logits, "margin").item(), 1.0, places=6
        )
        self.assertAlmostEqual(
            pixel_uncertainty(logits, "entropy").item(),
            math.log(2) / math.log(NUM_CLASSES),
            places=6,
        )

    def test_shape_range_and_invariance(self):
        generator = torch.Generator().manual_seed(SEED)
        logits = torch.randn(2, NUM_CLASSES, 3, 4, generator=generator)
        logits.requires_grad_()
        original = logits.detach().clone()
        permutation = torch.randperm(NUM_CLASSES, generator=generator)
        for method in ("margin", "entropy"):
            scores = pixel_uncertainty(logits, method)
            self.assertEqual(scores.shape, (2, 3, 4))
            self.assertEqual(scores.dtype, torch.float32)
            self.assertFalse(scores.requires_grad)
            self.assertTrue(torch.isfinite(scores).all().item())
            self.assertTrue(((scores >= 0) & (scores <= 1)).all().item())
            torch.testing.assert_close(
                scores, pixel_uncertainty(logits[:, permutation], method)
            )
            torch.testing.assert_close(
                scores, pixel_uncertainty(logits + 100.0, method)
            )
        torch.testing.assert_close(logits.detach(), original)

    def test_geometry_mask_and_image_denominator(self):
        scores = torch.tensor(
            [[[0.2, 0.8], [1.0, 1.0]], [[0.4, 0.0], [0.0, 0.0]]]
        )
        valid = torch.tensor(
            [[[True, True], [False, False]], [[True, False], [False, False]]]
        )
        torch.testing.assert_close(
            mean_image_uncertainty(scores, valid), torch.tensor([0.5, 0.4])
        )
        all_valid = torch.ones_like(valid)
        torch.testing.assert_close(
            mean_image_uncertainty(scores, all_valid), scores.mean(dim=(1, 2))
        )

    def test_invalid_logits(self):
        bad_inputs = (
            torch.zeros(2, 3, 4),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(0, NUM_CLASSES, 2, 2),
            torch.zeros(1, NUM_CLASSES, 2, 2, dtype=torch.int64),
            torch.full((1, NUM_CLASSES, 2, 2), float("nan")),
            torch.full((1, NUM_CLASSES, 2, 2), float("inf")),
        )
        for logits in bad_inputs:
            with self.subTest(shape=logits.shape, dtype=logits.dtype):
                with self.assertRaises(ValueError):
                    pixel_uncertainty(logits, "margin")
        with self.assertRaises(ValueError):
            pixel_uncertainty(torch.zeros(1, NUM_CLASSES, 2, 2), "unknown")

    def test_invalid_geometry_masks_and_scores(self):
        scores = torch.zeros(2, 2, 3)
        valid = torch.ones_like(scores, dtype=torch.bool)
        invalid_row = valid.clone()
        invalid_row[1] = False
        for mask in (invalid_row, valid[:, :, :1], valid.float()):
            with self.assertRaises(ValueError):
                mean_image_uncertainty(scores, mask)
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mean_image_uncertainty(torch.full_like(scores, value), valid)


class TestUncertaintyTopB(unittest.TestCase):
    def test_ranking_excludes_labeled_and_preserves_inputs(self):
        scores = torch.tensor([0.1, 0.95, 0.8, 1.0, 0.95, 0.6])
        unlabeled = [0, 1, 2, 4, 5]
        original_scores, original_ids = scores.clone(), unlabeled.copy()
        selected = uncertainty_top_b(scores, unlabeled, 3)
        self.assertEqual(selected, [1, 4, 2])
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(all(type(i) is int for i in selected))
        torch.testing.assert_close(scores, original_scores)
        self.assertEqual(unlabeled, original_ids)

    def test_ties_use_smallest_id_even_for_unsorted_candidates(self):
        scores = torch.ones(6)
        self.assertEqual(uncertainty_top_b(scores, [5, 2, 4, 0], 3), [0, 2, 4])

    def test_zero_budget_full_budget_and_empty_pool(self):
        scores = torch.tensor([0.2, 0.9, 0.5])
        self.assertEqual(uncertainty_top_b(scores, [0, 1, 2], 0), [])
        self.assertEqual(uncertainty_top_b(scores, [0, 1, 2], 3), [1, 2, 0])
        self.assertEqual(uncertainty_top_b(scores, [], 0), [])
        self.assertEqual(uncertainty_top_b(torch.empty(0), [], 0), [])

    def test_invalid_budget(self):
        for budget in (-1, 3, 1.5, True):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    uncertainty_top_b(torch.ones(3), [0, 2], budget)

    def test_invalid_candidate_ids(self):
        for ids in ([1, 1], [-1], [3], [True], [1.0]):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    uncertainty_top_b(torch.ones(3), ids, 1)

    def test_invalid_scores(self):
        for scores in (
            torch.ones(2, 2),
            torch.tensor([0.1, float("nan")]),
            torch.tensor([0.1, float("inf")]),
            torch.tensor([0.1, -0.2]),
            torch.ones(2, dtype=torch.int64),
        ):
            with self.assertRaises(ValueError):
                uncertainty_top_b(scores, [0, 1], 1)

    def test_matches_reference_on_full_voc_sized_pool(self):
        generator = torch.Generator().manual_seed(SEED)
        scores = torch.rand(1464, generator=generator)
        unlabeled = [i for i in range(1464) if i % 3 != 0]
        expected = sorted(unlabeled, key=lambda i: (-scores[i].item(), i))[:336]
        self.assertEqual(uncertainty_top_b(scores, unlabeled, 336), expected)

    def test_alseg_contract_and_shuffled_prediction_batches(self):
        # Лише локальні контракти; encoder, датасет і training не імпортуємо.
        sys.path.insert(0, str(ROOT / "src/source"))
        try:
            from alseg.core.predictions import LogitBatch
            from alseg.core.scores import assemble_scores
            from alseg.core.selection import (
                AcquisitionMethod,
                SelectionRequest,
                validate_selection,
            )
        finally:
            sys.path.pop(0)

        logits = torch.zeros(2, NUM_CLASSES, 2, 3)
        logits[1, 0] = 100.0
        batches = [
            LogitBatch([2, 0], logits, torch.ones(2, 2, 3, dtype=torch.bool)),
            LogitBatch(
                [1], torch.zeros(1, NUM_CLASSES, 2, 3),
                torch.ones(1, 2, 3, dtype=torch.bool),
            ),
        ]
        for method in ("margin", "entropy"):
            batch_scores = [
                (
                    batch.ids,
                    mean_image_uncertainty(
                        pixel_uncertainty(batch.logits, method), batch.valid_pixels
                    ),
                )
                for batch in batches
            ]
            scores = assemble_scores(3, batch_scores)
            torch.testing.assert_close(scores, torch.tensor([0.0, 1.0, 1.0]))
            request = SelectionRequest(
                method=AcquisitionMethod.UNCERTAINTY,
                embeddings=torch.eye(3),
                labeled_ids=[1],
                unlabeled_ids=[0, 2],
                n_select=1,
                seed=SEED,
                uncertainty=scores,
            )
            selected = uncertainty_top_b(
                request.uncertainty, request.unlabeled_ids, request.n_select
            )
            self.assertEqual(selected, [2])
            validate_selection(request, selected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
