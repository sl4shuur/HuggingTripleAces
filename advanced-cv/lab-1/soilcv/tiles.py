"""Fixed-mm crops. PPM remains an estimate unless a real length is known."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .artifacts import file_hash, identity_hash


def estimate_ppm(camera_ppm, source_size, camera_size):
    a, b = np.sort(source_size) / np.sort(camera_size)
    if not np.isfinite([a, b, camera_ppm]).all() or min(a, b, camera_ppm) <= 0:
        raise ValueError("PPM and image sizes must be positive and finite.")
    if abs(a / b - 1) >= 0.01:
        raise ValueError(
            "Aspect ratios differ. A full-image resize estimate is not valid."
        )
    return float(camera_ppm * (a + b) / 2)


def _identity(manifest):
    excluded = {"image_path", "tile_path", "cache_id"}
    records = manifest.drop(columns=list(excluded & set(manifest.columns))).to_dict(
        "records"
    )
    return identity_hash(records)


def make_tile_manifest(
    photos,
    *,
    tile_mm=64,
    output_px=288,
    centres=((0.4, 0.5), (0.5, 0.5), (0.6, 0.5)),
    image_root=None,
):
    if (
        not np.isfinite(tile_mm)
        or tile_mm <= 0
        or (
            output_px is not None and (not isinstance(output_px, int) or output_px <= 0)
        )
    ):
        raise ValueError(
            "Use a positive tile size and integer output pixels, or None for native crops."
        )
    if photos.empty or photos.duplicated(["sample_id", "photo"]).any() or not centres:
        raise ValueError("Use unique photo rows and at least one crop centre.")
    rows = []
    for photo in photos.sort_values(["sample_id", "photo"]).itertuples(index=False):
        path = Path(photo.path)
        ppm = float(photo.ppm_est)
        if not np.isfinite(ppm) or ppm <= 0:
            raise ValueError("PPM must be positive and finite.")
        crop_px = round(tile_mm * ppm)
        with Image.open(path) as image:
            width, height = image.size
            if image.getexif().get(274) in {5, 6, 7, 8}:
                width, height = height, width
        if crop_px < 1 or crop_px > min(width, height):
            raise ValueError(f"Tile does not fit {photo.photo}.")
        source_hash = file_hash(path)
        for tile_id, (cx, cy) in enumerate(centres):
            left, top = (
                round(cx * width - crop_px / 2),
                round(cy * height - crop_px / 2),
            )
            if min(left, top) < 0 or left + crop_px > width or top + crop_px > height:
                raise ValueError(f"Crop centre is outside {photo.photo}.")
            side = crop_px if output_px is None else output_px
            record = {
                "sample_id": str(photo.sample_id),
                "photo": str(photo.photo),
                "tile_id": tile_id,
                "image_path": str(path.relative_to(image_root))
                if image_root
                else str(path.resolve()),
                "source_hash": source_hash,
                "left": left,
                "top": top,
                "crop_px": crop_px,
                "output_px": side,
                "tile_mm_requested": tile_mm,
                "tile_mm_est": crop_px / ppm,
                "ppm_est": ppm,
                "output_ppm_est": side * ppm / crop_px,
                "ppm_method": getattr(photo, "ppm_method", "user_estimate"),
                "absolute_scale_verified": bool(
                    getattr(photo, "absolute_scale_verified", False)
                ),
                "orientation": "exif_transpose",
                "interpolation": "LANCZOS",
                "cache_schema": 1,
            }
            if hasattr(photo, "fold"):
                record["fold"] = int(photo.fold)
            rows.append(record)
    manifest = pd.DataFrame(rows)
    cache_id = _identity(manifest)
    manifest["cache_id"] = cache_id
    manifest["tile_path"] = [f"{cache_id}/{i:06d}.png" for i in range(len(manifest))]
    return manifest


def build_tile_cache(manifest, cache_root, *, image_root=None):
    root = Path(cache_root)
    cache_id = _identity(manifest)
    if not manifest.cache_id.eq(cache_id).all():
        raise ValueError("Manifest settings changed. Make a new manifest first.")
    for path_string, group in manifest.groupby("image_path", sort=False):
        path = (Path(image_root) / path_string) if image_root else Path(path_string)
        if not group.source_hash.eq(file_hash(path)).all():
            raise ValueError("A source image changed. Make a new cache identity.")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            for row in group.itertuples(index=False):
                destination = root / row.tile_path
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    crop = image.crop(
                        (
                            row.left,
                            row.top,
                            row.left + row.crop_px,
                            row.top + row.crop_px,
                        )
                    )
                    if crop.size != (row.output_px, row.output_px):
                        crop = crop.resize(
                            (row.output_px, row.output_px), Image.Resampling.LANCZOS
                        )
                    crop.save(destination)
    path = root / cache_id / "manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict("records"), indent=2, allow_nan=False) + "\n"
    )
    check_cache(manifest, root)
    return manifest


def check_cache(manifest, cache_root):
    root = Path(cache_root)
    cache_id = _identity(manifest)
    if not manifest.cache_id.eq(cache_id).all():
        raise ValueError("Cache identity does not match the requested settings.")
    saved = pd.DataFrame(json.loads((root / cache_id / "manifest.json").read_text()))
    if (
        _identity(saved) != cache_id
        or saved.tile_path.tolist() != manifest.tile_path.tolist()
    ):
        raise ValueError("Saved cache metadata does not match.")
    for row in manifest.itertuples(index=False):
        with Image.open(root / row.tile_path) as image:
            if image.size != (row.output_px, row.output_px):
                raise ValueError("Cached tile dimensions do not match.")
            image.verify()
    return True
