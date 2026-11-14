"""Evaluation of pretrained image-text models (CLIP, SigLIP, SigLIP 2) and of DINOv2 as a vision-only reference.

Writes to `results/pretrained/`:
    zero_shot.csv   zero-shot accuracy on STL-10 and the three fine-grained datasets of Lecture 3 (several prompt sets)
    probes.csv      k-NN, linear probe and few-shot linear probes on the same image embeddings
    retrieval.csv   Flickr30k test (1k images, 5k captions) Recall@1/5/10 in both directions
    examples.json   top retrieved images / captions for a few queries
    geometry.npz    cosine-similarity statistics and 2D PCA of image and text embeddings (the modality gap),
                    zero-shot confusion matrices
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lecture_utils import (ImageFiles, STL10_CLASSES, few_shot_probe, get_device, knn_accuracy, linear_probe,
                           load_flickr30k, load_stl10, load_transfer)
from vision_language import TEMPLATES, retrieval_metrics, zero_shot_weights

PRETRAINED = {   # name: (HuggingFace repo, lower-case the text)
    "CLIP ViT-B/16": ("openai/clip-vit-base-patch16", False),
    "SigLIP ViT-B/16": ("google/siglip-base-patch16-224", False),
    "SigLIP 2 ViT-B/16": ("google/siglip2-base-patch16-224", True),   # SigLIP 2 was trained on lower-cased text
}
VISION_ONLY = {"DINOv2 ViT-B/14": "vit_base_patch14_reg4_dinov2.lvd142m"}
PROMPTS = {"class name": ("{}",), "a photo of a {}.": ("a photo of a {}.",), f"ensemble of {len(TEMPLATES)}": TEMPLATES}
# the prompts CLIP uses for these datasets: they tell the text encoder what kind of class name follows
DATASET_PROMPTS = {"stl10": None, "pets": ("a photo of a {}, a type of pet.",),
                   "flowers": ("a photo of a {}, a type of flower.",),
                   "dtd": ("a photo of a {} texture.", "a photo of a {} pattern.", "a photo of a {} thing.",
                           "a photo of a {} object.")}
SHOTS = (1, 2, 4, 8, 16)
FEW_SHOT_SEEDS = 5
QUERY_CAPTIONS = 4   # caption -> image examples
QUERY_IMAGES = 3     # image -> caption examples


class HFDualEncoder:
    """A pretrained CLIP / SigLIP model from the HuggingFace Hub with the two encode functions we need."""

    def __init__(self, repo, lowercase, device):
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(repo, dtype=torch.float16).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(repo)
        self.lowercase, self.device = lowercase, device
        self.siglip = "siglip" in repo

    def image_transform(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def encode_images(self, items, batch_size=256):
        loader = torch.utils.data.DataLoader(ImageFiles(items, self.image_transform), batch_size=batch_size,
                                             num_workers=8)
        out = [self.model.get_image_features(pixel_values=x.to(self.device, torch.float16)).pooler_output
               for x in loader]
        return F.normalize(torch.cat(out).float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts, batch_size=1024):
        texts = [t.lower() for t in texts] if self.lowercase else list(texts)
        # SigLIP was trained with texts padded to 64 tokens and needs the same padding at test time
        kw = dict(padding="max_length", max_length=64) if self.siglip else dict(padding=True)
        out = []
        for i in range(0, len(texts), batch_size):
            t = self.processor(text=texts[i:i + batch_size], truncation=True, return_tensors="pt", **kw).to(self.device)
            out.append(self.model.get_text_features(**t).pooler_output)
        return F.normalize(torch.cat(out).float(), dim=-1)


@torch.no_grad()
def encode_vision_only(ckpt, items, device, batch_size=256):
    import timm

    model = timm.create_model(ckpt, pretrained=True, num_classes=0, img_size=224).eval().to(device)
    cfg = timm.data.resolve_data_config({"input_size": (3, 224, 224)}, model=model)
    tf = timm.data.create_transform(**cfg)
    loader = torch.utils.data.DataLoader(ImageFiles(items, tf), batch_size=batch_size, num_workers=8)
    out = []
    for x in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x.to(device)).float())
    return torch.cat(out)


def probes(name, dataset, train_x, train_y, test_x, test_y):
    c = int(train_y.max()) + 1
    base = {"model": name, "dataset": dataset}
    rows = [{**base, "protocol": "k-NN (all labels)", "shots": -1, "seed": 0,
             "acc": knn_accuracy(train_x, train_y, test_x, test_y, num_classes=c)},
            {**base, "protocol": "linear (all labels)", "shots": -1, "seed": 0,
             "acc": linear_probe(train_x, train_y, test_x, test_y, num_classes=c)["test_acc"]}]
    for k in SHOTS:
        for s in range(FEW_SHOT_SEEDS):
            rows.append({**base, "protocol": "few-shot linear", "shots": k, "seed": s,
                         "acc": few_shot_probe(train_x, train_y, test_x, test_y, k, num_classes=c, seed=s)})
    return rows


def classification_datasets(data_dir):
    """name -> (train images, train labels, test images, test labels, class names)"""
    out = {}
    stl_train_x, stl_train_y = load_stl10(data_dir, "train")
    stl_test_x, stl_test_y = load_stl10(data_dir, "test")
    out["stl10"] = (stl_train_x, stl_train_y, stl_test_x, stl_test_y, STL10_CLASSES)
    for name in ("pets", "flowers", "dtd"):
        out[name] = load_transfer(name, data_dir)
    return out


def geometry(img, txt):
    """Cosine similarities and a shared 2D PCA of image and text embeddings of the same Flickr30k pairs."""
    n = len(img)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(img.device)
    stats = {"matched": (img * txt).sum(1), "random pair": (img * txt[perm]).sum(1),
             "image-image": (img * img[perm]).sum(1), "text-text": (txt * txt[perm]).sum(1)}
    both = torch.cat([img, txt])
    both = both - both.mean(0)
    _, _, v = torch.pca_lowrank(both, q=2, center=False)
    return {k: v_.cpu().numpy() for k, v_ in stats.items()}, (both @ v).cpu().numpy(), \
        (img.mean(0) - txt.mean(0)).norm().item()


def run(data_dir, results_dir):
    device = get_device()
    out = os.path.join(results_dir, "pretrained")
    os.makedirs(out, exist_ok=True)
    datasets = classification_datasets(data_dir)
    paths, captions = load_flickr30k(data_dir, "test")
    flat = [c for caps in captions for c in caps]
    owner = torch.tensor([i for i, caps in enumerate(captions) for _ in caps], device=device)
    rng = np.random.default_rng(0)
    q_caps, q_imgs = rng.choice(len(flat), QUERY_CAPTIONS, replace=False), rng.choice(len(paths), QUERY_IMAGES, replace=False)

    zs_rows, probe_rows, ret_rows, examples, geo = [], [], [], {"query_captions": [int(i) for i in q_caps],
                                                                "query_images": [int(i) for i in q_imgs]}, {}
    for name, (repo, lowercase) in PRETRAINED.items():
        enc = HFDualEncoder(repo, lowercase, device)
        for ds, (xtr, ytr, xte, yte, classes) in datasets.items():
            ytr, yte = torch.as_tensor(ytr, device=device), torch.as_tensor(yte, device=device)
            tr, te = enc.encode_images(xtr), enc.encode_images(xte)
            prompts = dict(PROMPTS) if ds == "stl10" else {"class name": ("{}",), "a photo of a {}.": ("a photo of a {}.",),
                                                          "dataset-specific": DATASET_PROMPTS[ds]}
            for prompt_name, templates in prompts.items():
                w = zero_shot_weights(enc.encode_text, classes, templates)
                pred = (te @ w.T).argmax(1)
                zs_rows.append({"model": name, "dataset": ds, "prompts": prompt_name,
                                "acc": (pred == yte).float().mean().item()})
                if ds == "stl10" and prompt_name.startswith("ensemble"):
                    conf = torch.zeros(10, 10)
                    conf.index_put_((yte.cpu(), pred.cpu()), torch.ones(len(pred)), accumulate=True)
                    geo[f"{name}|confusion"] = conf.numpy()
            probe_rows += probes(name, ds, tr, ytr, te, yte)
            print(f"[pretrained] {name} {ds}: zero-shot {zs_rows[-1]['acc']:.3f}", flush=True)

        img, txt = enc.encode_images(paths), enc.encode_text(flat)
        ret_rows.append({"model": name, **retrieval_metrics(img, txt, owner)})
        sims = img @ txt.T
        examples[name] = {"caption_to_images": sims.T[q_caps].topk(5, dim=1).indices.tolist(),
                          "image_to_captions": sims[q_imgs].topk(5, dim=1).indices.tolist()}
        stats, pca, gap = geometry(img, txt[torch.arange(0, len(txt), 5, device=device)])
        for k, v in stats.items():
            geo[f"{name}|cos|{k}"] = v
        geo[f"{name}|pca"], geo[f"{name}|gap"] = pca, np.array(gap)
        geo[f"{name}|logit_scale"] = np.array(enc.model.logit_scale.exp().item())
        if getattr(enc.model, "logit_bias", None) is not None:
            geo[f"{name}|logit_bias"] = np.array(enc.model.logit_bias.item())
        print(f"[pretrained] {name}: {ret_rows[-1]['mean_recall']:.3f} retrieval", flush=True)
        del enc
        torch.cuda.empty_cache()

    for name, ckpt in VISION_ONLY.items():
        for ds, (xtr, ytr, xte, yte, _) in datasets.items():
            tr, te = encode_vision_only(ckpt, xtr, device), encode_vision_only(ckpt, xte, device)
            probe_rows += probes(name, ds, tr, torch.as_tensor(ytr, device=device), te,
                                 torch.as_tensor(yte, device=device))

    pd.DataFrame(zs_rows).to_csv(os.path.join(out, "zero_shot.csv"), index=False)
    pd.DataFrame(probe_rows).to_csv(os.path.join(out, "probes.csv"), index=False)
    pd.DataFrame(ret_rows).to_csv(os.path.join(out, "retrieval.csv"), index=False)
    with open(os.path.join(out, "examples.json"), "w") as f:
        json.dump(examples, f)
    np.savez_compressed(os.path.join(out, "geometry.npz"), **{k: np.asarray(v, dtype=np.float32) for k, v in geo.items()})
