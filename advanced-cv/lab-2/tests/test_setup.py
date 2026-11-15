import tomllib
from pathlib import Path

import pytest

from alseg.data import find_voc_root, validate_voc_root


def make_voc(root):
    splits = root / "ImageSets" / "Segmentation"
    splits.mkdir(parents=True)
    (root / "JPEGImages").mkdir()
    (root / "SegmentationClass").mkdir()
    for split, image_id in (("train", "example_train"), ("val", "example_val")):
        (splits / f"{split}.txt").write_text(image_id + "\n")
        (root / "JPEGImages" / f"{image_id}.jpg").touch()
        (root / "SegmentationClass" / f"{image_id}.png").touch()
    return root


def test_find_train_val_beside_test_directory(tmp_path):
    (tmp_path / "VOC2012_test" / "VOC2012_test").mkdir(parents=True)
    root = make_voc(tmp_path / "VOC2012_train_val" / "VOC2012_train_val")
    assert find_voc_root(tmp_path) == root
    assert validate_voc_root(root) == {
        "train": ["example_train"],
        "val": ["example_val"],
    }


def test_test_only_mount_is_rejected(tmp_path):
    (tmp_path / "VOC2012_test").mkdir()
    with pytest.raises(ValueError, match="VOC2012_test alone is insufficient"):
        find_voc_root(tmp_path)


def test_missing_mask_is_rejected(tmp_path):
    root = make_voc(tmp_path / "VOC2012")
    (root / "SegmentationClass" / "example_val.png").unlink()
    with pytest.raises(ValueError, match="Missing val file"):
        validate_voc_root(root)


def test_ambiguous_roots_require_explicit_override(tmp_path):
    make_voc(tmp_path / "first")
    make_voc(tmp_path / "second")
    with pytest.raises(ValueError, match="found 2"):
        find_voc_root(tmp_path)


def test_overlapping_splits_are_rejected(tmp_path):
    root = make_voc(tmp_path / "VOC2012")
    (root / "ImageSets" / "Segmentation" / "val.txt").write_text("example_train\n")
    with pytest.raises(ValueError, match="overlap"):
        validate_voc_root(root)


def test_config_uses_cumulative_budgets_and_writable_kaggle_outputs():
    with (Path(__file__).parents[1] / "configs" / "base.toml").open("rb") as file:
        config = tomllib.load(file)
    budgets = config["budgets"]
    assert [budgets[0]] + [b - a for a, b in zip(budgets, budgets[1:])] == [
        21,
        21,
        42,
        84,
        168,
        336,
    ]
    assert all(path.startswith("/kaggle/working/") for path in config["paths"].values())
