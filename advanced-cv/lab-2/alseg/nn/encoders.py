"""Load pinned backbone weights through the installed Transformers implementation."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from alseg.adapters.dinov2 import DinoV2Encoder
from alseg.adapters.vit16 import ViT16Encoder
from alseg.interfaces.encoder import Encoder

VIT16_WEIGHTS = {
    "dino_vitb16": {
        "url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        "sha256": "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    },
    "ibot_vitb16": {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
        "sha256": "734b9c49f46cd330d740f6c35d6a83517aab770a3efd86d7f1175ced582c9b7f",
    },
}


def convert_vit16_weights(state: dict, image_size: int) -> dict:
    """Map official DINO/iBOT backbone keys to Transformers 5.20.2 ViT keys."""
    if type(image_size) is not int or image_size <= 0 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of patch_size=16")
    state = state.get("state_dict", state)
    converted = {}
    for key, value in state.items():
        # The official iBOT teacher download also contains its SSL projection head.
        if key.startswith("head."):
            continue
        if key == "cls_token":
            target = "embeddings.cls_token"
        elif key == "pos_embed":
            target = "embeddings.position_embeddings"
            if image_size != 224:
                grid = image_size // 16
                patches = value[:, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
                patches = F.interpolate(
                    patches,
                    scale_factor=((grid + 0.1) / 14,) * 2,
                    mode="bicubic",
                    align_corners=False,
                )
                value = torch.cat(
                    (value[:, :1], patches.flatten(2).transpose(1, 2)), dim=1
                )
        elif key.startswith("patch_embed.proj."):
            target = key.replace(
                "patch_embed.proj.", "embeddings.patch_embeddings.projection."
            )
        elif key.startswith("norm."):
            target = key.replace("norm.", "layernorm.", 1)
        elif key.startswith("blocks."):
            target = key.replace("blocks.", "layers.", 1)
            if ".attn.qkv." in target:
                for name, part in zip(
                    ("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0), strict=True
                ):
                    converted[target.replace("attn.qkv", f"attention.{name}")] = part
                continue
            target = target.replace(".attn.proj.", ".attention.o_proj.")
            target = target.replace(".norm1.", ".layernorm_before.")
            target = target.replace(".norm2.", ".layernorm_after.")
        else:
            raise ValueError(f"Unexpected backbone weight: {key}")
        converted[target] = value
    return converted


def load_encoder(config: dict, cache_dir: Path, device: str) -> Encoder:
    name = config["name"]
    if name == "dinov2":
        if config["checkpoint_path"] is not None:
            raise ValueError("checkpoint_path is supported only for DINO/iBOT ViT-B/16")
        return DinoV2Encoder.from_pretrained(
            config["model_id"],
            config["revision"],
            config["image_size"],
            device=device,
            cache_dir=cache_dir / "huggingface",
        )
    if name not in VIT16_WEIGHTS:
        raise ValueError(f"Unknown encoder: {name}")
    from transformers import ViTConfig, ViTModel

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
    model = ViTModel(
        ViTConfig(image_size=config["image_size"], patch_size=16, layer_norm_eps=1e-6),
        add_pooling_layer=False,
    )
    model.load_state_dict(
        convert_vit16_weights(state, config["image_size"]), strict=True
    )
    return ViT16Encoder(
        model,
        config["image_size"],
        {
            "name": name,
            "model_id": name,
            "weights_url": weights["url"],
            "weights_sha256": digest,
            "pretraining_dataset": "ImageNet-1K",
            "pretraining_method": "DINO" if name == "dino_vitb16" else "iBOT",
            "weights_role": "teacher_backbone_only",
            "voc_finetuned": False,
        },
        device,
    )
