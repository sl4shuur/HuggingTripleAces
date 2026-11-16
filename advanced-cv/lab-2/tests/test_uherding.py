"""Синтетичні CPU-тести UHerding та adaptive sigma без Oracle і training."""

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
for cell_id in ("imports", "tests", "b85f0a67", "22f84e14", "13c6b63a"):
    exec(compile(cells[cell_id], f"normis.ipynb:{cell_id}", "exec"), namespace)

pairwise_distances = namespace["pairwise_distances"]
median_kernel_sigma = namespace["median_kernel_sigma"]
gaussian_kernel = namespace["gaussian_kernel"]
adaptive_kernel_sigma = namespace["adaptive_kernel_sigma"]
maxherding = namespace["maxherding"]
uherding = namespace["uherding"]
uncertainty_top_b = namespace["uncertainty_top_b"]
SIGMA_FLOOR = namespace["SIGMA_FLOOR"]


class TestAdaptiveSigma(unittest.TestCase):
    def test_uses_only_distinct_labeled_pairs(self):
        distances = pairwise_distances(torch.tensor([[0.0], [10.0], [0.1], [20.0]]))
        before = distances.clone()
        self.assertEqual(adaptive_kernel_sigma(distances, [1, 0], 2.0), 10.0)
        torch.testing.assert_close(distances, before)
        # Зміна відстаней до unlabeled точок не змінює labeled minimum.
        distances[2:, :] = 0.001
        distances[:, 2:] = 0.001
        self.assertEqual(adaptive_kernel_sigma(distances, [0, 1], 2.0), 10.0)

    def test_fallback_for_fewer_than_two_labels(self):
        distances = pairwise_distances(torch.tensor([[0.0], [1.0], [4.0]]))
        fallback = median_kernel_sigma(distances)
        self.assertEqual(adaptive_kernel_sigma(distances, [], fallback), fallback)
        self.assertEqual(adaptive_kernel_sigma(distances, [2], fallback), fallback)

    def test_nonincreasing_trajectory_and_duplicate_floor(self):
        distances = pairwise_distances(torch.tensor([[0.0], [5.0], [6.0], [6.0]]))
        trajectory = [
            adaptive_kernel_sigma(distances, list(range(n)), 3.0)
            for n in (2, 3, 4)
        ]
        self.assertEqual(trajectory, [5.0, 1.0, SIGMA_FLOOR])
        self.assertTrue(all(a >= b for a, b in zip(trajectory, trajectory[1:])))

    def test_invalid_ids_and_fallback(self):
        distances = torch.zeros(3, 3)
        for labeled in ([0, 0], [-1], [3], [True], [1.5]):
            with self.assertRaises(ValueError):
                adaptive_kernel_sigma(distances, labeled, 1.0)
        for fallback in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                adaptive_kernel_sigma(distances, [], fallback)


class TestUHerding(unittest.TestCase):
    def test_constant_positive_weights_match_maxherding(self):
        generator = torch.Generator().manual_seed(42)
        embeddings = torch.nn.functional.normalize(
            torch.randn(20, 8, generator=generator), dim=1
        )
        kernel = gaussian_kernel(pairwise_distances(embeddings), 0.7)
        for labeled in ([], [1, 5, 9]):
            expected = maxherding(kernel, labeled, 6)
            for value in (0.7, 1.0, 7.0, 1e-20):
                with self.subTest(labeled=labeled, weight=value):
                    weights = torch.full((20,), value)
                    self.assertEqual(uherding(kernel, labeled, 6, weights), expected)

    def test_small_sigma_matches_top_b_with_ties(self):
        kernel = gaussian_kernel(pairwise_distances(torch.eye(6)), 1e-8)
        weights = torch.tensor([0.1, 0.9, 1.0, 0.9, 0.2, 0.7])
        expected = uncertainty_top_b(weights, [0, 1, 3, 4, 5], 4)
        self.assertEqual(uherding(kernel, [2], 4, weights), expected)
        self.assertEqual(expected, [1, 3, 5, 4])

    def test_weights_belong_to_targets_not_candidates(self):
        # Центральна точка має uncertainty=0, але покриває двох невизначених сусідів.
        embeddings = torch.tensor([
            [1.0, 0.0], [0.9, math.sqrt(0.19)], [0.9, -math.sqrt(0.19)],
        ])
        kernel = gaussian_kernel(pairwise_distances(embeddings), 0.8)
        weights = torch.tensor([0.0, 1.0, 1.0])
        self.assertEqual(uherding(kernel, [], 1, weights), [0])
        self.assertEqual(uncertainty_top_b(weights, [0, 1, 2], 1), [1])

    def test_assignment_six_point_example(self):
        embeddings = torch.tensor([[0.0], [1.0], [2.0], [6.0], [7.0], [9.0]])
        kernel = gaussian_kernel(pairwise_distances(embeddings), 2.0)
        weights = torch.tensor([0.2, 0.0, 1.0, 0.2, 1.0, 0.2])
        self.assertEqual(maxherding(kernel, [1], 2), [4, 5])
        self.assertEqual(uherding(kernel, [1], 2, weights), [4, 2])

    def test_matches_independent_weighted_greedy(self):
        generator = torch.Generator().manual_seed(19)
        embeddings = torch.nn.functional.normalize(
            torch.randn(9, 4, generator=generator), dim=1
        )
        weights = torch.rand(9, generator=generator)
        points, scalar_weights = embeddings.tolist(), weights.tolist()
        sigma = 0.9
        similarities = [
            [math.exp(-(math.dist(a, b) / sigma) ** 2) for b in points]
            for a in points
        ]
        kernel = gaussian_kernel(pairwise_distances(embeddings), sigma)
        for labeled in ([], [1], [5, 2]):
            targets = [i for i in range(len(points)) if i not in labeled]
            selected = []
            for _ in range(4):
                gains = {}
                for candidate in targets:
                    if candidate in selected:
                        continue
                    gain = 0.0
                    for target in targets:
                        coverage = max(
                            (similarities[target][j] for j in labeled + selected),
                            default=0.0,
                        )
                        improvement = max(similarities[target][candidate] - coverage, 0.0)
                        gain += scalar_weights[target] * improvement
                    gains[candidate] = gain
                selected.append(min(gains, key=lambda i: (-gains[i], i)))
                self.assertEqual(uherding(kernel, labeled, len(selected), weights), selected)

    def test_labeled_weights_are_irrelevant_and_inputs_unchanged(self):
        kernel = gaussian_kernel(pairwise_distances(torch.eye(5)), 1.0)
        weights = torch.tensor([0.3, 0.5, 0.8, 0.4, 0.2])
        labeled = [1, 3]
        originals = kernel.clone(), weights.clone(), labeled.copy()
        selected = uherding(kernel, labeled, 2, weights)
        torch.testing.assert_close(kernel, originals[0])
        torch.testing.assert_close(weights, originals[1])
        self.assertEqual(labeled, originals[2])
        changed = weights.clone()
        changed[labeled] = 1e20
        self.assertEqual(uherding(kernel, labeled, 2, changed), selected)
        self.assertEqual(uherding(kernel, labeled, 2, weights * 1000), selected)

    def test_zero_weights_and_boundary_budgets(self):
        kernel = torch.eye(5)
        weights = torch.zeros(5)
        self.assertEqual(uherding(kernel, [1, 3], 3, weights), [0, 2, 4])
        self.assertEqual(uherding(kernel, [1], 0, weights), [])
        self.assertEqual(uherding(kernel, list(range(5)), 0, weights), [])
        for budget in (-1, 5, True, 1.5):
            with self.assertRaises(ValueError):
                uherding(kernel, [1], budget, weights)

    def test_invalid_or_missing_weights(self):
        for weights in (
            None, torch.ones(3), torch.ones(4, 1), torch.ones(4, dtype=torch.int64),
            torch.tensor([0.0, -1.0, 0.0, 0.0]),
            torch.tensor([0.0, float("nan"), 0.0, 0.0]),
            torch.tensor([0.0, float("inf"), 0.0, 0.0]),
        ):
            with self.assertRaises(ValueError):
                uherding(torch.eye(4), [0], 1, weights)

    def test_full_pool_adaptive_sigma_and_selection_contract(self):
        sys.path.insert(0, str(ROOT / "src/source"))
        try:
            from alseg.core.selection import (
                AcquisitionMethod, SelectionRequest, validate_selection,
            )
        finally:
            sys.path.pop(0)

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 4))
        try:
            generator = torch.Generator().manual_seed(42)
            embeddings = torch.nn.functional.normalize(
                torch.randn(1464, 384, generator=generator), dim=1
            )
            weights = torch.rand(1464, generator=generator)
            labeled = list(range(336))
            distances = pairwise_distances(embeddings)
            sigma = adaptive_kernel_sigma(distances, labeled, median_kernel_sigma(distances))
            kernel = gaussian_kernel(distances, sigma)
            request = SelectionRequest(
                method=AcquisitionMethod.UHERDING,
                embeddings=embeddings,
                labeled_ids=labeled,
                unlabeled_ids=list(range(336, 1464)),
                n_select=336,
                seed=42,
                uncertainty=weights,
                sigma=sigma,
            )
            selected = uherding(kernel, request.labeled_ids, request.n_select, request.uncertainty)
            validate_selection(request, selected)
            self.assertEqual(len(set(selected)), 336)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
