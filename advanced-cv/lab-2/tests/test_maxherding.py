"""Синтетичні CPU-тести Gaussian kernel та MaxHerding із notebook."""

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
# Лише imports, kernel-функції та MaxHerding, без демонстрацій і training.
for cell_id in ("imports", "b85f0a67", "22f84e14"):
    exec(compile(cells[cell_id], f"normis.ipynb:{cell_id}", "exec"), namespace)

pairwise_distances = namespace["pairwise_distances"]
median_kernel_sigma = namespace["median_kernel_sigma"]
gaussian_kernel = namespace["gaussian_kernel"]
maxherding = namespace["maxherding"]


class TestGaussianKernel(unittest.TestCase):
    def test_exact_formula_without_factor_two(self):
        embeddings = torch.tensor([[0.0, 0.0], [3.0, 4.0]], requires_grad=True)
        distances = pairwise_distances(embeddings)
        torch.testing.assert_close(distances, torch.tensor([[0.0, 5.0], [5.0, 0.0]]))
        kernel = gaussian_kernel(distances, sigma=5.0)
        expected = torch.tensor([[1.0, math.exp(-1)], [math.exp(-1), 1.0]])
        torch.testing.assert_close(kernel, expected)
        self.assertFalse(kernel.requires_grad)
        self.assertEqual(kernel.dtype, torch.float32)

    def test_symmetry_duplicates_and_small_sigma(self):
        generator = torch.Generator().manual_seed(42)
        embeddings = torch.nn.functional.normalize(
            torch.randn(7, 384, generator=generator), dim=1
        )
        embeddings[1] = embeddings[0]
        distances = pairwise_distances(embeddings)
        kernel = gaussian_kernel(distances, sigma=1e-8)
        expected = torch.eye(7)
        expected[0, 1] = expected[1, 0] = 1.0
        torch.testing.assert_close(kernel, expected)
        torch.testing.assert_close(distances, distances.T)
        torch.testing.assert_close(distances.diagonal(), torch.zeros(7))

    def test_median_excludes_diagonal_and_averages_middle_pair(self):
        distances = pairwise_distances(torch.tensor([[0.0], [1.0], [4.0], [10.0]]))
        # Відстані між різними IDs: 1, 3, 4, 6, 9, 10. Median = (4 + 6) / 2.
        self.assertEqual(median_kernel_sigma(distances), 5.0)

    def test_zero_median_floor_and_insufficient_pool(self):
        distances = pairwise_distances(torch.ones(4, 2))
        sigma = median_kernel_sigma(distances)
        self.assertEqual(sigma, 1e-6)
        torch.testing.assert_close(gaussian_kernel(distances, sigma), torch.ones(4, 4))
        with self.assertRaises(ValueError):
            median_kernel_sigma(torch.zeros(1, 1))

    def test_invalid_embeddings_and_sigma(self):
        for embeddings in (
            torch.zeros(2), torch.zeros(0, 2), torch.zeros(2, 0),
            torch.zeros(2, 2, dtype=torch.int64),
            torch.full((2, 2), float("nan")),
            torch.full((2, 2), float("inf")),
        ):
            with self.assertRaises(ValueError):
                pairwise_distances(embeddings)
        for sigma in (0.0, -1.0, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                gaussian_kernel(torch.zeros(2, 2), sigma)


class TestMaxHerding(unittest.TestCase):
    def test_recomputes_gains_instead_of_one_shot_top_b(self):
        embeddings = torch.tensor([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0],
        ])
        distances = pairwise_distances(embeddings)
        kernel = gaussian_kernel(distances, median_kernel_sigma(distances))
        initial_top_two = torch.argsort(
            kernel.sum(dim=0), descending=True, stable=True
        )[:2].tolist()
        self.assertEqual(initial_top_two, [2, 3])
        self.assertEqual(maxherding(kernel, [], 2), [2, 0])
        self.assertEqual(maxherding(kernel, [2], 1), [0])

    def test_assignment_six_point_example(self):
        embeddings = torch.tensor([[0.0], [1.0], [2.0], [6.0], [7.0], [9.0]])
        kernel = gaussian_kernel(pairwise_distances(embeddings), sigma=2.0)
        self.assertEqual(maxherding(kernel, [], 2), [1, 4])
        self.assertEqual(maxherding(kernel, [1], 2), [4, 5])

    def test_matches_independent_scalar_greedy(self):
        generator = torch.Generator().manual_seed(17)
        embeddings = torch.nn.functional.normalize(
            torch.randn(9, 4, generator=generator), dim=1
        )
        points = embeddings.tolist()
        sigma = 0.8
        similarities = [
            [math.exp(-(math.dist(a, b) / sigma) ** 2) for b in points]
            for a in points
        ]
        kernel = gaussian_kernel(pairwise_distances(embeddings), sigma)
        for labeled in ([], [1], [7, 2]):
            targets = [i for i in range(len(points)) if i not in labeled]
            selected = []
            for _ in range(5):
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
                        gain += max(similarities[target][candidate] - coverage, 0.0)
                    gains[candidate] = gain
                chosen = min(gains, key=lambda i: (-gains[i], i))
                selected.append(chosen)
                self.assertEqual(maxherding(kernel, labeled, len(selected)), selected)

    def test_zero_gains_ties_and_no_input_mutation(self):
        kernel = torch.ones(5, 5)
        labeled = [3, 1]
        original_kernel, original_labeled = kernel.clone(), labeled.copy()
        self.assertEqual(maxherding(kernel, labeled, 3), [0, 2, 4])
        torch.testing.assert_close(kernel, original_kernel)
        self.assertEqual(labeled, original_labeled)

    def test_empty_and_full_budgets(self):
        kernel = torch.eye(4)
        self.assertEqual(maxherding(kernel, [], 0), [])
        self.assertEqual(maxherding(kernel, [], 4), [0, 1, 2, 3])
        self.assertEqual(maxherding(kernel, [0, 1, 2, 3], 0), [])
        self.assertEqual(maxherding(torch.ones(1, 1), [], 1), [0])

    def test_invalid_ids_budget_and_shape(self):
        for labeled in ([1, 1], [-1], [4], [True], [1.5]):
            with self.assertRaises(ValueError):
                maxherding(torch.eye(4), labeled, 1)
        for budget in (-1, 4, 1.5, True):
            with self.assertRaises(ValueError):
                maxherding(torch.eye(4), [0], budget)
        with self.assertRaises(ValueError):
            maxherding(torch.ones(3, 2), [], 1)

    def test_full_voc_sized_pool_and_selection_contract(self):
        sys.path.insert(0, str(ROOT / "src/source"))
        try:
            from alseg.core.selection import (
                AcquisitionMethod, SelectionRequest, validate_selection,
            )
        finally:
            sys.path.pop(0)

        # Обмежуємо CPU threads лише на час тесту, не змінюємо training settings.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 4))
        try:
            generator = torch.Generator().manual_seed(42)
            embeddings = torch.nn.functional.normalize(
                torch.randn(1464, 384, generator=generator), dim=1
            )
            distances = pairwise_distances(embeddings)
            sigma = median_kernel_sigma(distances)
            kernel = gaussian_kernel(distances, sigma)
            self.assertEqual(kernel.shape, (1464, 1464))
            self.assertTrue(torch.isfinite(kernel).all().item())
            for n_labeled, n_select in ((0, 21), (336, 336)):
                request = SelectionRequest(
                    method=AcquisitionMethod.MAXHERDING,
                    embeddings=embeddings,
                    labeled_ids=list(range(n_labeled)),
                    unlabeled_ids=list(range(n_labeled, 1464)),
                    n_select=n_select,
                    seed=42,
                    sigma=sigma,
                )
                selected = maxherding(kernel, request.labeled_ids, request.n_select)
                validate_selection(request, selected)
                self.assertEqual(len(set(selected)), n_select)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
