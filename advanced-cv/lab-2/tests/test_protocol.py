import pytest

from alseg.core.calibration import split_labeled_ids


@pytest.mark.parametrize(
    "budget,n_calibration", [(1, 0), (2, 1), (21, 4), (42, 8), (672, 134)]
)
def test_shared_holdout_counts_and_canonical_reproducibility(budget, n_calibration):
    ids = list(range(budget))
    fit, calibration = split_labeled_ids(ids, 0.2, 4, 20042)
    assert len(calibration) == n_calibration
    assert fit and set(fit).isdisjoint(calibration)
    assert sorted(fit + calibration) == ids
    assert (fit, calibration) == split_labeled_ids(ids[::-1], 0.2, 4, 20042)


def test_zero_fraction_disables_calibration():
    assert split_labeled_ids([2, 0, 1], 0, 4, 42) == ([0, 1, 2], [])


@pytest.mark.parametrize(
    "ids,fraction,minimum",
    [([], 0.2, 4), ([0, 0], 0.2, 4), ([0], 1, 4), ([0], 0.2, -1)],
)
def test_invalid_splits_are_rejected(ids, fraction, minimum):
    with pytest.raises(ValueError):
        split_labeled_ids(ids, fraction, minimum, 42)
