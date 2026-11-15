import json
from dataclasses import replace

import pytest
import torch
from synthetic import make_fixture

from alseg.core.calibration import CalibrationBatch
from alseg.core.pool import PoolManifest
from alseg.core.scores import assemble_scores
from alseg.core.selection import AcquisitionMethod, validate_selection


@pytest.mark.parametrize("dimension", [2, 5])
def test_teammate_handoff(dimension):
    fixture = make_fixture(dimension)
    request = fixture.request
    # Hand-computed ordering: IDs 1 and 3 tie at 0.9, so ID 1 comes first.
    ordered = sorted(
        request.unlabeled_ids, key=lambda i: (-request.uncertainty[i].item(), i)
    )
    assert ordered[: request.n_select] == fixture.expected_top_uncertainty
    validate_selection(request, fixture.expected_top_uncertainty)
    assert set(fixture.calibration.predictions.ids) <= set(request.labeled_ids)
    assert fixture.calibration.targets[0, 1, 1] == 255
    assert fixture.calibration.predictions.valid_pixels[0, 1, 1]


@pytest.mark.parametrize(
    "changes",
    [
        {"labeled_ids": [0, 0]},
        {"labeled_ids": [0, 1]},
        {"unlabeled_ids": [1, 2, 3]},
        {"unlabeled_ids": [5, 3, 2, 1]},
        {"unlabeled_ids": [True, 2, 3, 5]},
        {"n_select": -1},
        {"n_select": 5},
        {"n_select": True},
        {"seed": -1},
        {"seed": 2**63},
        {"uncertainty": None},
        {"uncertainty": torch.ones(4)},
        {"uncertainty": torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0, 0.0])},
        {"uncertainty": torch.full((6,), float("nan"))},
        {"embeddings": torch.zeros(6, 2)},
        {"embeddings": torch.full((6, 2), float("nan"))},
        {"embeddings": torch.ones(6, 2, dtype=torch.float64)},
        {"method": "unknown"},
        {"method": AcquisitionMethod.MAXHERDING},
        {"sigma": 0},
        {"sigma": float("inf")},
        {"sigma": True},
    ],
)
def test_reject_invalid_selection_request(changes):
    with pytest.raises(ValueError):
        replace(make_fixture().request, **changes)


@pytest.mark.parametrize("selected", [[1], [1, 1], [0, 1], [1, 6], [True, 3], (1, 3)])
def test_reject_invalid_selector_output(selected):
    with pytest.raises(ValueError):
        validate_selection(make_fixture().request, selected)


def test_empty_and_exhausted_pool():
    request = replace(make_fixture().request, n_select=0)
    validate_selection(request, [])
    exhausted = replace(request, labeled_ids=list(range(6)), unlabeled_ids=[])
    validate_selection(exhausted, [])
    with pytest.raises(ValueError):
        replace(exhausted, n_select=1)


def test_cold_start_is_explicit_and_full_batch_is_allowed():
    request = replace(
        make_fixture().request,
        method=AcquisitionMethod.RANDOM,
        labeled_ids=[],
        unlabeled_ids=list(range(6)),
        uncertainty=None,
        n_select=6,
    )
    validate_selection(request, list(range(6)))
    with pytest.raises(ValueError):
        replace(request, method=AcquisitionMethod.UHERDING, sigma=1.0)
    replace(request, method=AcquisitionMethod.MAXHERDING, sigma=1.0)


def test_scores_are_aligned_by_id_not_batch_order():
    expected = make_fixture().request.uncertainty
    actual = assemble_scores(
        6, [([3, 1], expected[[3, 1]]), ([5, 0, 4, 2], expected[[5, 0, 4, 2]])]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "batches",
    [
        [([0], torch.ones(1))],
        [([0, 1], torch.ones(2)), ([1], torch.ones(1))],
        [([6], torch.ones(1))],
        [([0, 1], torch.ones(1))],
        [([0], torch.tensor([-1.0]))],
        [([0], torch.tensor([float("nan")]))],
    ],
)
def test_reject_incomplete_or_invalid_scores(batches):
    with pytest.raises(ValueError):
        assemble_scores(6, batches)


def test_logit_batch_rejects_shape_and_geometry_errors():
    batch = make_fixture().predictions
    for changes in (
        {"logits": batch.logits[:, :20]},
        {"ids": [3]},
        {"valid_pixels": torch.ones(2, 3, 2, dtype=torch.bool)},
        {"valid_pixels": torch.zeros(2, 2, 3, dtype=torch.bool)},
        {"logits": torch.full_like(batch.logits, float("inf"))},
    ):
        with pytest.raises(ValueError):
            replace(batch, **changes)
    assert not hasattr(batch, "targets")


@pytest.mark.parametrize("value", [-1, 21])
def test_reject_invalid_calibration_classes(value):
    calibration = make_fixture().calibration
    targets = calibration.targets.clone()
    targets[0, 0, 0] = value
    with pytest.raises(ValueError):
        CalibrationBatch(calibration.predictions, targets)


def test_calibration_padding_must_be_ignored():
    calibration = make_fixture().calibration
    targets = calibration.targets.clone()
    targets[0, 0, -1] = 0
    with pytest.raises(ValueError):
        CalibrationBatch(calibration.predictions, targets)


def test_manifest_preserves_permanent_row_identity():
    manifest = make_fixture().manifest
    assert manifest.pool_ids[4] == "synthetic_train_4"
    for pool, val in ((["a", "a"], ["b"]), (["a"], ["a"]), ([], ["b"])):
        with pytest.raises(ValueError):
            PoolManifest(pool, val)


@pytest.mark.parametrize("method", list(AcquisitionMethod))
def test_method_config_and_result_round_trip(method):
    restored = AcquisitionMethod(json.loads(json.dumps(method)))
    assert restored is method
    assert str(method) == method.value


def test_method_rejects_unknown_config_values_and_unparsed_strings():
    with pytest.raises(ValueError):
        AcquisitionMethod("unknown")
    with pytest.raises(ValueError, match="AcquisitionMethod"):
        replace(make_fixture().request, method="uncertainty")
