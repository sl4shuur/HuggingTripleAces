"""Load pinned backbone weights with one loader for the supported encoders."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from alseg.adapters.cached_features import CachedFeatures
from alseg.adapters.dinov2 import DinoV2Encoder
from alseg.adapters.vit16 import ViT16Encoder
from alseg.config import EncoderName
from alseg.interfaces.encoder import FeatureSource

VIT16_WEIGHTS = {
    EncoderName.DINO_VITB16: {
        "url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        "sha256": "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    },
    EncoderName.IBOT_VITB16: {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
        "sha256": "734b9c49f46cd330d740f6c35d6a83517aab770a3efd86d7f1175ced582c9b7f",
    },
}


def load_encoder(config: dict, cache_dir: Path, device: str) -> FeatureSource:
    name = EncoderName(config["name"])
    if config["features_dir"] is not None:
        identity = {
            "name": name,
            "image_size": config["image_size"],
            "image_pooling": "l2_normalized_cls",
            "preprocessing": "full_frame_square_bicubic_antialias_clamp_0_1",
        }
        if name == EncoderName.DINOV2:
            identity.update(model_id=config["model_id"], revision=config["revision"])
        else:
            identity["weights_sha256"] = VIT16_WEIGHTS[name]["sha256"]
        return CachedFeatures(Path(config["features_dir"]), identity)
    if name == EncoderName.DINOV2:
        if config["checkpoint_path"] is not None:
            raise ValueError("checkpoint_path is supported only for DINO/iBOT ViT-B/16")
        return DinoV2Encoder.from_pretrained(
            config["model_id"],
            config["revision"],
            config["image_size"],
            device=device,
            cache_dir=cache_dir / "huggingface",
        )
    import timm

    weights = VIT16_WEIGHTS[name]
    path = (
        Path(config["checkpoint_path"])
        if config["checkpoint_path"]
        else cache_dir / f"{name}.pth"
    )
    if not path.exists() and config["checkpoint_path"] is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        torch.hub.download_url_to_file(
            weights["url"], str(temporary), hash_prefix=weights["sha256"]
        )
        temporary.replace(path)
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != weights["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {name}: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {key: value for key, value in state.items() if not key.startswith("head.")}
    if config["image_size"] != 224:
        # Preserve official DINO/iBOT interpolation rather than changing the features.
        positions = state["pos_embed"]
        grid = config["image_size"] // 16
        patches = positions[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        patches = F.interpolate(
            patches,
            scale_factor=((grid + 0.1) / 14,) * 2,
            mode="bicubic",
            align_corners=False,
        )
        state["pos_embed"] = torch.cat(
            (positions[:, :1], patches.flatten(2).transpose(1, 2)), dim=1
        )
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=config["image_size"],
    )
    model.load_state_dict(state, strict=True)
    return ViT16Encoder(
        model,
        config["image_size"],
        {
            "name": name,
            "model_id": name,
            "weights_url": weights["url"],
            "weights_sha256": digest,
            "pretraining_dataset": "ImageNet-1K",
            "pretraining_method": "DINO" if name == EncoderName.DINO_VITB16 else "iBOT",
            "weights_role": "teacher_backbone_only",
            "voc_finetuned": False,
        },
        device,
    )
