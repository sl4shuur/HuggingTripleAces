"""Filesystem integration for official VOC segmentation splits."""

from pathlib import Path


def validate_voc_root(root: Path) -> dict[str, list[str]]:
    """Require all official train/val images and semantic masks; never use test."""
    splits: dict[str, list[str]] = {}
    for split in ("train", "val"):
        split_file = root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.is_file():
            raise ValueError(f"Missing official segmentation split: {split_file}")
        ids = split_file.read_text().split()
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate IDs in {split_file}")
        for image_id in ids:
            if Path(image_id).name != image_id or image_id in (".", ".."):
                raise ValueError(f"Invalid VOC filename stem: {image_id}")
            for folder, extension in (
                ("JPEGImages", ".jpg"),
                ("SegmentationClass", ".png"),
            ):
                file = root / folder / f"{image_id}{extension}"
                if not file.is_file():
                    raise ValueError(f"Missing {split} file: {file}")
        splits[split] = ids
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("Official segmentation train and val IDs overlap")
    return splits


def find_voc_root(dataset_root: Path) -> Path:
    """Accept only one complete train/val root; ambiguous mounts need an override."""
    matches = []
    for split_file in sorted(dataset_root.rglob("ImageSets/Segmentation/train.txt")):
        candidate = split_file.parents[2]
        try:
            validate_voc_root(candidate)
        except ValueError:
            continue
        matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "Expected one complete VOC segmentation train/val root under "
            f"{dataset_root}; "
            f"found {len(matches)}. Set data.voc_root explicitly to a root containing "
            "ImageSets/Segmentation/{train,val}.txt, JPEGImages and SegmentationClass. "
            "VOC2012_test alone is insufficient."
        )
    return matches[0]
