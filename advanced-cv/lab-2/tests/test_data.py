import json

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from PIL import Image

from alseg.adapters.voc import VocEvaluation, VocImages, VocOracle
from alseg.data import inspect_voc_samples, load_manifest, load_mask
from setup_project import main


@pytest.fixture
def voc(tmp_path):
    root = tmp_path / "VOC2012"
    splits = root / "ImageSets" / "Segmentation"
    splits.mkdir(parents=True)
    (root / "JPEGImages").mkdir()
    (root / "SegmentationClass").mkdir()
    (splits / "train.txt").write_text("second\nfirst\n")
    (splits / "val.txt").write_text("validation\n")
    for stem in ("second", "first", "validation"):
        Image.new("RGB", (3, 2), (10, 20, 30)).save(root / "JPEGImages" / f"{stem}.jpg")
        # Palette colors intentionally differ from class IDs.
        mask = Image.fromarray(np.array([[0, 1, 255], [20, 3, 0]], dtype=np.uint8))
        mask = mask.convert("P")
        mask.putpalette([value for i in range(256) for value in (i, 255 - i, 42)])
        mask.save(root / "SegmentationClass" / f"{stem}.png")
    return root, load_manifest(root)


def test_manifest_keeps_split_file_order_and_pool_access_has_no_labels(voc):
    root, manifest = voc
    pool = VocImages(root, manifest)
    assert manifest.pool_ids == ["second", "first"]
    assert len(pool) == 2
    image = pool.get_image(0)
    assert image.shape == (3, 2, 3)
    assert image.dtype == torch.uint8 and image.device.type == "cpu"
    assert not hasattr(pool, "get_mask")
    for bad_id in (-1, 2, True):
        with pytest.raises(ValueError):
            pool.get_image(bad_id)


def test_val_and_oracle_preserve_palette_indices_and_void(voc, tmp_path):
    root, manifest = voc
    oracle = VocOracle(root, manifest, tmp_path / "queries.jsonl")
    val = VocEvaluation(root, manifest)
    assert len(val) == 1
    expected = torch.tensor([[0, 1, 255], [20, 3, 0]])
    torch.testing.assert_close(val.get_mask(0), expected)
    with pytest.raises(KeyError):
        oracle.get_mask(1)
    masks = oracle.query([1, 0])
    torch.testing.assert_close(masks[1], expected)
    assert oracle.labeled_ids == [1, 0]
    masks[1].zero_()
    retrieved = oracle.get_mask(1)
    retrieved.zero_()
    torch.testing.assert_close(oracle.get_mask(1), expected)
    assert oracle.labeled_ids == [1, 0]


@pytest.mark.parametrize("ids", [[0, 0], [0, 2], [-1], [True]])
def test_invalid_query_does_not_consume_any_labels(voc, tmp_path, ids):
    root, manifest = voc
    log = tmp_path / "queries.jsonl"
    oracle = VocOracle(root, manifest, log)
    with pytest.raises(ValueError):
        oracle.query(ids)
    assert oracle.labeled_ids == []
    assert json.loads(log.read_text())["status"] == "rejected"


def test_already_labeled_id_rejects_entire_batch_and_is_logged(voc, tmp_path):
    root, manifest = voc
    log = tmp_path / "queries.jsonl"
    oracle = VocOracle(root, manifest, log)
    oracle.query([0])
    with pytest.raises(ValueError):
        oracle.query([1, 0])
    assert oracle.labeled_ids == [0]
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [event["status"] for event in events] == ["accepted", "rejected"]
    assert sum(len(e["ids"]) for e in events if e["status"] == "accepted") == 1
    with pytest.raises(FileExistsError):
        VocOracle(root, manifest, log)


def test_corrupt_mask_does_not_partially_commit_query(voc, tmp_path):
    root, manifest = voc
    (root / "SegmentationClass" / "first.png").write_bytes(b"not a PNG")
    oracle = VocOracle(root, manifest, tmp_path / "queries.jsonl")
    with pytest.raises(OSError):
        oracle.query([0, 1])
    assert oracle.labeled_ids == []


def test_failed_audit_write_does_not_commit_labels(voc, tmp_path, monkeypatch):
    root, manifest = voc
    oracle = VocOracle(root, manifest, tmp_path / "queries.jsonl")

    def fail_log(*args):
        raise OSError("disk full")

    monkeypatch.setattr(oracle, "_log", fail_log)
    with pytest.raises(OSError, match="disk full"):
        oracle.query([0])
    assert oracle.labeled_ids == []


def test_mask_class_shape_and_color_encoding_validation(voc):
    root, _ = voc
    path = root / "SegmentationClass" / "first.png"
    for image, size, message in (
        (Image.new("L", (3, 2), 21), (3, 2), "class IDs"),
        (Image.new("L", (3, 2), 0), (2, 2), "size mismatch"),
        (Image.new("RGB", (3, 2)), (3, 2), "indexed or grayscale"),
    ):
        image.save(path)
        with pytest.raises(ValueError, match=message):
            load_mask(path, size)


def test_sample_inspection_and_cli(voc, tmp_path):
    root, manifest = voc
    samples = inspect_voc_samples(root, manifest)
    assert samples["train"]["stem"] == "second"
    assert samples["val"]["class_ids"] == [0, 1, 3, 20, 255]
    assert samples["val"]["void_pixels"] == 1
    config = tmp_path / "config.toml"
    config.write_text(f'[data]\ndataset_root = "{root.as_posix()}"\nvoc_root = ""\n')
    result = CliRunner().invoke(main, ["--config", str(config), "--inspect-data"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["data_checked"] is True
    assert report["samples"] == samples
