"""Deterministic RGB transforms, independent of image loading and encoders."""

import torch
import torch.nn.functional as F
from torch import Tensor

from alseg.core.helpers import check_tensor
from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def resize_mask(mask: Tensor, image_size: int) -> Tensor:
    """Match full-frame square image geometry without interpolating class IDs."""
    check_tensor(mask, "mask", torch.int64, 2)
    if min(mask.shape) == 0 or type(image_size) is not int or image_size <= 0:
        raise ValueError("Expected a nonempty mask and positive integer image_size")
    if not (((mask >= 0) & (mask < NUM_CLASSES)) | (mask == IGNORE_INDEX)).all():
        raise ValueError("Invalid segmentation class IDs")
    return F.interpolate(
        mask[None, None].float(),
        size=(image_size, image_size),
        mode="nearest-exact",
    )[0, 0].long()


def prepare_images(images: list[Tensor], image_size: int) -> Tensor:
    """Resize the complete frame to a fixed square, then normalize.

    This deliberately avoids classification center-cropping. Aspect ratio is
    changed; task masks must use the same size with nearest-neighbor resizing.
    No random augmentation or hidden masks participate in feature extraction.
    """
    if not images or type(image_size) is not int or image_size <= 0:
        raise ValueError("Expected nonempty images and a positive integer image_size")
    resized = []
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    for image in images:
        check_tensor(image, "image", torch.uint8, 3)
        if image.shape[0] != 3 or min(image.shape[1:]) <= 0:
            raise ValueError("Images must be nonempty RGB [3,H,W]")
        pixels = F.interpolate(
            image.unsqueeze(0).float() / 255,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0].clamp(0, 1)
        resized.append((pixels - mean) / std)
    return torch.stack(resized)
