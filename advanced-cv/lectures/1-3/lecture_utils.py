import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 95431):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------------------------------------------------
# name: (timm checkpoint or None for random init, pre-training, extra kwargs for timm.create_model)
ENCODERS = {
    "Random ViT-S/16": ("vit_small_patch16_224", "random init (lower bound)", {}),
    "ResNet-50 (sup, IN-1k)": ("resnet50.a1_in1k", "supervised, ImageNet-1k", {}),
    "ViT-S/16 (sup, IN-1k)": ("vit_small_patch16_224.augreg_in1k", "supervised, ImageNet-1k", {}),
    "ViT-S/16 (sup, IN-21k)": ("vit_small_patch16_224.augreg_in21k", "supervised, ImageNet-21k", {}),
    "ViT-S/16 (DINO)": ("vit_small_patch16_224.dino", "self-supervised, ImageNet-1k", {}),
    "ViT-S/14 (DINOv2-reg)": ("vit_small_patch14_reg4_dinov2.lvd142m", "self-supervised, LVD-142M", {"img_size": 224}),
    "ViT-S/16 (DINOv3)": ("vit_small_patch16_dinov3.lvd1689m", "self-supervised, LVD-1689M", {}),
    "ViT-B/16 (MAE)": ("vit_base_patch16_224.mae", "self-supervised (masked), ImageNet-1k", {}),
    "ViT-B/16 (CLIP)": ("vit_base_patch16_clip_224.openai", "image-text, 400M pairs", {}),
}


def load_encoder(name, device, img_size=224):
    """Returns (model in eval mode on `device`, eval transform at `img_size`)."""
    import timm

    ckpt, _, kwargs = ENCODERS[name]
    pretrained = name != "Random ViT-S/16"
    model = timm.create_model(ckpt, pretrained=pretrained, num_classes=0, **kwargs).eval().to(device)
    cfg = timm.data.resolve_data_config({"input_size": (3, img_size, img_size)}, model=model)
    return model, timm.data.create_transform(**cfg)


def num_prefix_tokens(model):
    return getattr(model, "num_prefix_tokens", 0)


def is_vit(model):
    return hasattr(model, "blocks")


# ----------------------------------------------------------------------------------------------------------------------
# Datasets: HuggingFace Hub mirrors with the official splits (the Oxford servers are often very slow)
# ----------------------------------------------------------------------------------------------------------------------
DATASET_NUM_CLASSES = {"pets": 37, "flowers": 102, "dtd": 47}
HF_SOURCES = {
    "pets": ("timm/oxford-iiit-pet", {"trainval": "train", "test": "test"}),
    "flowers": ("dpdl-benchmark/oxford_flowers102", {"train": "train", "val": "validation", "test": "test"}),
    "dtd": ("dpdl-benchmark/dtd", {"train": "train", "val": "validation", "test": "test"}),
}
PETS_ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"   # trimaps, ~19 MB


def load_hf_split(name, split, root):
    from datasets import load_dataset

    repo, mapping = HF_SOURCES[name]
    return load_dataset(repo, split=mapping[split], cache_dir=os.path.join(root, "hf_datasets"))


class HFImageDataset(torch.utils.data.Dataset):
    """(image, label) pairs from a HuggingFace split; `transform` is applied to the RGB PIL image."""

    def __init__(self, hf_split, transform=None):
        self.ds = hf_split
        self.transform = transform
        self.classes = hf_split.features["label"].names
        self._labels = list(hf_split["label"])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        img = row["image"].convert("RGB")
        return (self.transform(img) if self.transform else img), row["label"]


def _stratified_split(targets, val_fraction=0.2, seed=95431):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = rng.permutation(np.flatnonzero(targets == c))
        n_val = int(round(len(idx) * val_fraction))
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    return sorted(train_idx), sorted(val_idx)


def get_targets(dataset):
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return sum((get_targets(d) for d in dataset.datasets), [])
    if isinstance(dataset, torch.utils.data.Subset):
        base = get_targets(dataset.dataset)
        return [base[i] for i in dataset.indices]
    return list(dataset._labels)


def build_dataset(name, split, transform, root):
    """split in {'train', 'val', 'test', 'trainval'}. Oxford-IIIT Pets has no official val: we hold out 20% of trainval."""
    if name == "pets" and split in ("train", "val"):
        full = HFImageDataset(load_hf_split("pets", "trainval", root), transform)
        train_idx, val_idx = _stratified_split(full._labels)
        subset = torch.utils.data.Subset(full, train_idx if split == "train" else val_idx)
        subset.classes = full.classes
        return subset
    if split == "trainval" and name != "pets":
        parts = [build_dataset(name, s, transform, root) for s in ("train", "val")]
        concat = torch.utils.data.ConcatDataset(parts)
        concat.classes = parts[0].classes
        return concat
    return HFImageDataset(load_hf_split(name, split, root), transform)


def ensure_pets_trimaps(root):
    """Downloads and extracts the Oxford-IIIT Pets trimaps (pet / background / border masks) if missing."""
    import tarfile
    import urllib.request

    trimap_dir = os.path.join(root, "oxford-iiit-pet", "annotations", "trimaps")
    if not os.path.isdir(trimap_dir):
        os.makedirs(os.path.join(root, "oxford-iiit-pet"), exist_ok=True)
        archive = os.path.join(root, "oxford-iiit-pet", "annotations.tar.gz")
        urllib.request.urlretrieve(PETS_ANNOTATIONS_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(os.path.join(root, "oxford-iiit-pet"))
        os.remove(archive)
    return trimap_dir


class PetsSegmentation(torch.utils.data.Dataset):
    """Oxford-IIIT Pets with trimaps resized to `size` x `size` (no crop, so image and mask stay aligned).

    Returns (normalized image (3, S, S), mask (S, S) with 1 = pet, 0 = background, -1 = border/ignore).
    """

    def __init__(self, root, split, mean, std, size=224, indices=None):
        self.ds = load_hf_split("pets", split, root)
        self.trimap_dir = ensure_pets_trimaps(root)
        self.indices = list(range(len(self.ds))) if indices is None else [int(i) for i in indices]
        self.mean, self.std, self.size = mean, std, size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import torchvision.transforms.functional as TF
        from PIL import Image

        row = self.ds[self.indices[i]]
        img = row["image"].convert("RGB")
        trimap = Image.open(os.path.join(self.trimap_dir, f"{row['image_id']}.png"))
        img = TF.resize(img, [self.size, self.size], interpolation=TF.InterpolationMode.BICUBIC)
        trimap = TF.resize(trimap, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        x = TF.normalize(TF.to_tensor(img), self.mean, self.std)
        t = torch.from_numpy(np.array(trimap)).long()
        mask = torch.full_like(t, -1)
        mask[t == 1] = 1
        mask[t == 2] = 0
        return x, mask


# ----------------------------------------------------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------------------------------------------------
@torch.inference_mode()
def extract_features(model, dataset, device, batch_size=128, num_workers=4, layers=None, amp=True):
    """Frozen-encoder features for every image of `dataset`.

    Returns a dict of CPU float tensors:
      cls:  (N, D) [CLS] token after the final norm (= global average pool for CNNs)
      gap:  (N, D) mean of the patch tokens after the final norm (= global average pool for CNNs)
      y:    (N,)   labels
      layer_cls / layer_gap: (L, N, D) per-block pooled outputs (only for ViTs and if `layers` is given)
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    captured = {}
    hooks = []
    if layers is not None and is_vit(model):
        for i in layers:
            hooks.append(model.blocks[i].register_forward_hook(
                lambda mod, inp, out, i=i: captured.__setitem__(i, out)))
    npt = num_prefix_tokens(model)
    out = {"cls": [], "gap": [], "y": [], "layer_cls": [], "layer_gap": []}
    try:
        for x, y in loader:
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp and device in ("cuda", "mps")):
                feats = model.forward_features(x.to(device))
            feats = feats.float()
            if feats.ndim == 4:                                  # CNN: (B, C, H, W)
                pooled = feats.mean(dim=(2, 3))
                cls, gap = pooled, pooled
            else:                                                # ViT: (B, prefix + N, D)
                cls, gap = feats[:, 0], feats[:, npt:].mean(dim=1)
            out["cls"].append(cls.cpu())
            out["gap"].append(gap.cpu())
            out["y"].append(torch.as_tensor(y))
            if captured:
                out["layer_cls"].append(torch.stack([captured[i][:, 0].float().cpu() for i in layers]))
                out["layer_gap"].append(torch.stack([captured[i][:, npt:].float().mean(1).cpu() for i in layers]))
    finally:
        for h in hooks:
            h.remove()
    result = {k: torch.cat(v) for k, v in out.items() if k in ("cls", "gap", "y")}
    if out["layer_cls"]:
        result["layer_cls"] = torch.cat(out["layer_cls"], dim=1)
        result["layer_gap"] = torch.cat(out["layer_gap"], dim=1)
    return result


@torch.inference_mode()
def extract_patch_tokens(model, dataset, device, batch_size=64, num_workers=4):
    """Final-layer patch tokens of a ViT on a (image, mask) dataset: returns tokens (N, gh, gw, D) fp16 and masks."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    npt = num_prefix_tokens(model)
    tokens, masks = [], []
    for x, m in loader:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device in ("cuda", "mps")):
            feats = model.forward_features(x.to(device))
        patch = feats[:, npt:].float()
        g = int(round(patch.shape[1] ** 0.5))
        tokens.append(patch.reshape(patch.shape[0], g, g, -1).half().cpu())
        masks.append(m)
    return torch.cat(tokens), torch.cat(masks)


def masks_to_patch_labels(masks, grid):
    """(N, S, S) pixel masks {1, 0, -1} -> (N, grid, grid) patch labels: 1 / 0 by majority of labelled pixels, -1 if
    fewer than half of the patch pixels are labelled (mostly border)."""
    fg = (masks == 1).float().unsqueeze(1)
    labelled = (masks >= 0).float().unsqueeze(1)
    fg_frac = F.adaptive_avg_pool2d(fg, grid).squeeze(1)
    lab_frac = F.adaptive_avg_pool2d(labelled, grid).squeeze(1)
    labels = (fg_frac / lab_frac.clamp_min(1e-6) > 0.5).long()
    labels[lab_frac < 0.5] = -1
    return labels
