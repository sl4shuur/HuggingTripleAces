"""Read this competition's metadata and use saved sample folds."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
from .curves import TARGET_COLUMNS, check_ids, validate_curves

DEFAULT_FOLDS_PATH = Path(__file__).with_name("assets") / "sample_folds.csv"


def _name_key(text):
    text = str(text).lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return re.sub(r"[^a-z0-9]", "", text)


def load_metadata(data_dir):
    """Return (labels, photos, submission_template) from the competition input."""
    from .tiles import estimate_ppm
    root = Path(data_dir)
    labels = pd.read_csv(root / "Training_labels_updated.csv")
    template = pd.read_csv(root / "sample_submission.csv")
    cameras = pd.read_csv(root / "ppm_updated.csv")
    for frame in [labels, template]:
        frame["sample_id"] = frame.sample_id.str.strip()
        check_ids(frame)
    validate_curves(labels[list(TARGET_COLUMNS)])
    labels["soil_group"] = np.where(labels["0.063"] >= 50, "fine", np.where(labels["2"] >= 50, "sand", "gravel"))
    phone_names = {_name_key(row.phone): row for row in cameras.itertuples(index=False)}
    camera_names = {_name_key(row.camera): row for row in cameras.itertuples(index=False)}
    test_ids = {_name_key(sample): sample for sample in template.sample_id}
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        is_train = "Training-All_Photos_updated" in path.parts
        if is_train:
            match = re.search(r"_([A-Z]\d{3})_", path.stem)
            sample_id = match.group(1) if match else None
        else:
            name = "HPC_" + path.stem.split("_HPC_", 1)[-1]
            sample_id = test_ids.get(_name_key(re.sub(r"\s*\(\d+\)$", "", name)))
        with Image.open(path) as image:
            width, height = image.size
            model = str(image.getexif().get(272, "")).strip("\x00 ")
            image.verify()
        camera = camera_names.get(_name_key(model))
        if camera is None and is_train and sample_id:
            camera = phone_names.get(_name_key(path.stem.split("_" + sample_id)[0]))
        if camera is None or sample_id is None:
            raise ValueError(f"Cannot match sample/camera for {path.name}.")
        ppm = estimate_ppm(camera.ppm, (width, height), (camera.width, camera.height))
        rows.append({"sample_id": sample_id, "photo": path.name, "path": str(path),
                     "split": "train" if is_train else "test", "device": camera.phone,
                     "width": width, "height": height, "ppm_est": ppm,
                     "ppm_method": "resize_assumed_no_crop", "absolute_scale_verified": False})
    photos = pd.DataFrame(rows)
    if photos.empty:
        raise ValueError("No photos found.")
    if photos.duplicated(["sample_id", "photo"]).any():
        raise ValueError("Photo IDs must be unique within each sample.")
    for split, expected in [("train", labels.sample_id), ("test", template.sample_id)]:
        if set(photos.loc[photos.split.eq(split), "sample_id"]) != set(expected):
            raise ValueError(f"Photo IDs do not match all {split} samples.")
    if set(labels.sample_id) & set(template.sample_id):
        raise ValueError("Train and test sample IDs overlap.")
    return labels, photos, template


def validate_folds(folds, sample_ids, *, expected_folds=(0, 1, 2, 3)):
    check_ids(folds)
    if "fold" not in folds or folds.fold.isna().any():
        raise ValueError("Every sample needs a saved fold.")
    ids = list(sample_ids)
    if len(set(ids)) != len(ids) or set(ids) != set(folds.sample_id):
        raise ValueError("Saved folds must match the unique labelled sample IDs.")
    if len(expected_folds) < 2 or set(folds.fold) != set(expected_folds):
        raise ValueError("Expected all saved validation folds, with no extra folds.")
    return folds


def load_folds(path, sample_ids, *, expected_folds=(0, 1, 2, 3)):
    folds = pd.read_csv(path)
    return validate_folds(folds, sample_ids, expected_folds=expected_folds)


def sample_weights(manifest):
    """Uniform sample -> uniform photo -> uniform tile sampling weights."""
    required = ["sample_id", "photo", "tile_id"]
    if manifest.empty or manifest[required].isna().any().any() or manifest.duplicated(required).any():
        raise ValueError("Use a non-empty manifest with unique tile IDs.")
    n_photos = manifest.groupby("sample_id").photo.transform("nunique").to_numpy()
    n_tiles = manifest.groupby(["sample_id", "photo"]).photo.transform("size").to_numpy()
    weights = 1.0 / (n_photos * n_tiles)
    return weights / weights.sum()


class TileDataset:
    """A DataLoader-compatible dataset. Transform takes PIL RGB and returns CHW tensor."""
    def __init__(self, manifest, cache_root, labels=None, transform=None):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.cache_root = Path(cache_root)
        self.transform = transform
        self.targets = None
        if labels is not None:
            check_ids(labels)
            validate_curves(labels[list(TARGET_COLUMNS)])
            self.targets = labels.set_index("sample_id")[list(TARGET_COLUMNS)]
            if not set(self.manifest.sample_id) <= set(self.targets.index):
                raise ValueError("Some tiles have no target row.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        import torch
        row = self.manifest.iloc[index]
        with Image.open(self.cache_root / row.tile_path) as source:
            image = source.convert("RGB")
        if self.transform is None:
            array = np.asarray(image, dtype=np.float32).copy() / 255
            image = torch.from_numpy(array).permute(2, 0, 1)
        else:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError("Transform must return a CHW torch.Tensor.")
        result = {"image": image, "sample_id": str(row.sample_id), "photo": str(row.photo), "tile_id": str(row.tile_id)}
        if "fold" in row.index and pd.notna(row.fold):
            result["fold"] = int(row.fold)
        if self.targets is not None:
            result["target"] = torch.tensor(self.targets.loc[row.sample_id].to_numpy(float), dtype=torch.float32)
        return result
