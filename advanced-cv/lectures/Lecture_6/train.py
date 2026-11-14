"""Feature extraction and training of the dual encoders of Lecture 6 (LiT setting).

Image towers are frozen, so their features are computed once and cached in `features/<tower>/`:
    flickr_train.pt   (29000, V, D)  V random-resized-crop views of every training image
    flickr_val.pt     (1014, D), flickr_test.pt (1000, D)   center crop
    stl10_test.pt     (8000, D)   STL-10 test images (96 px upsampled to the input size of the tower)
A run trains the image projection and the text tower with the CLIP or the SigLIP loss and writes
`results/runs/<name>.csv` (validation retrieval during training), `<name>_steps.csv` (loss, temperature, bias)
and `<name>.json` (final test retrieval on Flickr30k and zero-shot accuracy on STL-10).
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from lecture_utils import ImageFiles, STL10_CLASSES, get_device, load_flickr30k, load_stl10, set_seed
from vision_language import (DualEncoder, PretrainedText, TextTransformer, retrieval_metrics, zero_shot_accuracy,
                             zero_shot_weights)

LECTURE_5_DINO = os.path.join("..", "Lecture_5", "logdirs", "dino", "encoder.pt")

# name: (timm checkpoint or "lecture5", pretrained, input size)
IMAGE_TOWERS = {
    "dinov2_vits14": ("vit_small_patch14_reg4_dinov2.lvd142m", True, 224),
    "sup_vits16_in21k": ("vit_small_patch16_224.augreg_in21k", True, 224),
    "random_vits16": ("vit_small_patch16_224", False, 224),
    "lecture5_dino_vitti8": ("lecture5", True, 96),
}
TEXT_TOWERS = {
    "scratch": ("openai/clip-vit-base-patch16", None),                                  # CLIP's BPE tokenizer
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
}

DEFAULT_CONFIG = dict(
    name="siglip_bs1024",
    image_tower="dinov2_vits14",
    text_tower="scratch",
    loss="siglip",
    batch_size=1024,
    epochs=50,               # one epoch = every training image once, with one of its 5 captions
    train_images=None,       # use only the first n training images (None: all 29k)
    lr=5e-4,
    text_lr=None,            # learning rate of a pretrained text tower (None: same as lr)
    weight_decay=0.1,
    warmup_fraction=0.05,
    embed_dim=256,
    max_len=32,
    eval_every=5,
    seed=95431,
)


# ----------------------------------------------------------------------------------------------------------------------
# Frozen image towers
# ----------------------------------------------------------------------------------------------------------------------
def build_image_tower(name, device):
    """Returns (model mapping normalized images to (B, D) features, transforms(train, eval), input size)."""
    import timm
    import torchvision.transforms as T

    ckpt, pretrained, size = IMAGE_TOWERS[name]
    if ckpt == "lecture5":
        from vision_transformer import VisionTransformer
        from lecture_utils import STL10_MEAN, STL10_STD

        model = VisionTransformer(img_size=96, patch_size=8, num_classes=0, embed_dim=192, depth=12, num_heads=3,
                                  pos_embed_type="sincos")
        model.load_state_dict(torch.load(LECTURE_5_DINO, map_location="cpu"))
        mean, std = STL10_MEAN, STL10_STD
    else:
        kwargs = {"img_size": 224} if "dinov2" in ckpt else {}
        model = timm.create_model(ckpt, pretrained=pretrained, num_classes=0, **kwargs)
        cfg = model.pretrained_cfg
        mean, std = cfg["mean"], cfg["std"]
    norm = T.Normalize(mean, std)
    train_tf = T.Compose([T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                          T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size),
                         T.ToTensor(), norm])
    return model.eval().to(device), (train_tf, eval_tf), size


@torch.no_grad()
def encode_images(model, dataset, device, batch_size=256, workers=8):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    out = []
    for x in loader:
        shape = x.shape
        x = x.to(device, non_blocking=True).flatten(0, -4)                 # views are folded into the batch
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f = model(x)
        out.append(f.float().reshape(*shape[:-3], -1).half().cpu())
    return torch.cat(out)


def extract_features(tower, data_dir, features_dir, device, train_views=5):
    """Computes and caches the features of a frozen image tower; returns the cache folder."""
    folder = os.path.join(features_dir, tower)
    files = ["flickr_train.pt", "flickr_val.pt", "flickr_test.pt", "stl10_test.pt"]
    if all(os.path.exists(os.path.join(folder, f)) for f in files):
        return folder
    os.makedirs(folder, exist_ok=True)
    set_seed(0)
    model, (train_tf, eval_tf), _ = build_image_tower(tower, device)
    t0 = time.time()
    for split in ("train", "val", "test"):
        paths, _ = load_flickr30k(data_dir, split)
        ds = ImageFiles(paths, train_tf, views=train_views) if split == "train" else ImageFiles(paths, eval_tf)
        torch.save(encode_images(model, ds, device), os.path.join(folder, f"flickr_{split}.pt"))
    x, _ = load_stl10(data_dir, "test")
    torch.save(encode_images(model, ImageFiles(x, eval_tf), device), os.path.join(folder, "stl10_test.pt"))
    print(f"[features] {tower}: {time.time() - t0:.0f} s", flush=True)
    return folder


# ----------------------------------------------------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------------------------------------------------
def get_tokenizer(text_tower):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TEXT_TOWERS[text_tower][0])


def tokenize(tokenizer, texts, max_len):
    """-> (N, max_len) int64 ids, lower-cased, padded with the pad id of the tokenizer."""
    return tokenizer([t.lower() for t in texts], padding="max_length", truncation=True, max_length=max_len,
                     return_tensors="pt")["input_ids"]


def build_text_tower(text_tower, tokenizer, max_len):
    if TEXT_TOWERS[text_tower][1] is None:
        # CLIP's tokenizer pads with its end-of-text token, so the end token is masked like the padding
        return TextTransformer(len(tokenizer), max_len=max_len, pad_id=tokenizer.pad_token_id)
    return PretrainedText(TEXT_TOWERS[text_tower][1])


# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------
def make_encode_text(model, tokenizer, max_len, device, batch_size=2048):
    @torch.no_grad()
    def encode_text(texts):
        ids = tokenize(tokenizer, texts, max_len).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            return torch.cat([model.encode_text(ids[i:i + batch_size]).float() for i in range(0, len(ids), batch_size)])
    return encode_text


@torch.no_grad()
def evaluate_retrieval(model, image_feats, captions, encode_text, device):
    img = model.encode_image(image_feats.to(device))
    flat = [c for caps in captions for c in caps]
    owner = torch.tensor([i for i, caps in enumerate(captions) for _ in caps], device=device)
    return retrieval_metrics(img, encode_text(flat), owner)


def lr_lambda(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train(config, data_dir, results_dir, features_dir):
    cfg = {**DEFAULT_CONFIG, **config}
    name = cfg["name"]
    device = get_device()
    set_seed(cfg["seed"])
    feats = extract_features(cfg["image_tower"], data_dir, features_dir, device)

    train_x = torch.load(os.path.join(feats, "flickr_train.pt")).to(device)          # (N, V, D)
    val_x = torch.load(os.path.join(feats, "flickr_val.pt"))
    test_x = torch.load(os.path.join(feats, "flickr_test.pt"))
    _, train_caps = load_flickr30k(data_dir, "train")
    _, val_caps = load_flickr30k(data_dir, "val")
    _, test_caps = load_flickr30k(data_dir, "test")

    tokenizer = get_tokenizer(cfg["text_tower"])
    text_tower = build_text_tower(cfg["text_tower"], tokenizer, cfg["max_len"])
    train_tokens = tokenize(tokenizer, [c for caps in train_caps for c in caps], cfg["max_len"])
    train_tokens = train_tokens.view(len(train_caps), 5, -1).to(device)             # (N, 5, L)
    if cfg["train_images"]:
        keep = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(0))[:cfg["train_images"]]
        train_x, train_tokens = train_x[keep.to(device)], train_tokens[keep.to(device)]

    model = DualEncoder(train_x.shape[-1], text_tower, cfg["embed_dim"], cfg["loss"]).to(device)
    pretrained_params = list(text_tower.parameters()) if cfg["text_lr"] else []
    pretrained_ids = {id(p) for p in pretrained_params}
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if id(p) in pretrained_ids:
            continue
        (no_decay if p.ndim < 2 or "logit" in n or "embed" in n else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    if pretrained_params:
        groups.append({"params": pretrained_params, "weight_decay": cfg["weight_decay"], "lr": cfg["text_lr"]})
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-6)

    n, bs = len(train_x), cfg["batch_size"]
    steps_per_epoch = n // bs
    # the number of optimizer steps is fixed by the full dataset, so that runs on fewer images train as long
    total = (29000 // bs) * cfg["epochs"]
    epochs = math.ceil(total / steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total, int(cfg["warmup_fraction"] * total)))
    encode_text = make_encode_text(model, tokenizer, cfg["max_len"], device)
    print(f"[{name}] {cfg['image_tower']} + {cfg['text_tower']} text, {cfg['loss']} loss, batch {bs}, "
          f"{steps_per_epoch} steps/epoch, {total} steps", flush=True)

    epoch_rows, step_rows, step = [], [], 0
    t0 = time.time()

    def log_eval(epoch):
        model.eval()
        r = evaluate_retrieval(model, val_x, val_caps, encode_text, device)
        model.train()
        epoch_rows.append({"epoch": epoch, "step": step, "minutes": (time.time() - t0) / 60, **r})
        print(f"[{name}] epoch {epoch:3d} | val mean recall {r['mean_recall']:.3f} | t2i R@1 {r['t2i_R@1']:.3f} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    log_eval(0)
    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        for b in range(min(steps_per_epoch, total - step)):
            idx = perm[b * bs:(b + 1) * bs]
            view = torch.randint(train_x.shape[1], (bs,), device=device)
            cap = torch.randint(5, (bs,), device=device)
            loss = model(train_x[idx, view], train_tokens[idx, cap])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % max(1, total // 400) == 0:
                row = {"step": step, "epoch": step / steps_per_epoch, "loss": loss.item(),
                       "temperature": 1 / model.logit_scale.exp().item()}
                if model.logit_bias is not None:
                    row["bias"] = model.logit_bias.item()
                step_rows.append(row)
        if epoch % cfg["eval_every"] == 0 or epoch == epochs:
            log_eval(epoch)

    model.eval()
    minutes = (time.time() - t0) / 60
    test = evaluate_retrieval(model, test_x, test_caps, encode_text, device)
    stl_x = torch.load(os.path.join(feats, "stl10_test.pt"))
    _, stl_y = load_stl10(data_dir, "test")
    with torch.no_grad():
        stl_emb = model.encode_image(stl_x.to(device))
    zs = zero_shot_accuracy(stl_emb, torch.as_tensor(stl_y, device=device),
                            zero_shot_weights(encode_text, STL10_CLASSES))

    os.makedirs(os.path.join(results_dir, "runs"), exist_ok=True)
    pd.DataFrame(epoch_rows).to_csv(os.path.join(results_dir, "runs", f"{name}.csv"), index=False)
    pd.DataFrame(step_rows).to_csv(os.path.join(results_dir, "runs", f"{name}_steps.csv"), index=False)
    summary = {"name": name, **{k: cfg[k] for k in ("image_tower", "text_tower", "loss", "batch_size", "epochs",
                                                     "train_images", "seed")},
               **{f"test_{k}": v for k, v in test.items()}, "stl10_zero_shot": zs,
               "temperature": 1 / model.logit_scale.exp().item(),
               "bias": model.logit_bias.item() if model.logit_bias is not None else np.nan,
               "trainable_params": sum(p.numel() for p in model.parameters()), "minutes": minutes, "config": cfg}
    with open(os.path.join(results_dir, "runs", f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[{name}] done in {minutes:.1f} min | test mean recall {test['mean_recall']:.3f} | "
          f"i2t R@1 {test['i2t_R@1']:.3f} | t2i R@1 {test['t2i_R@1']:.3f} | STL-10 zero-shot {zs:.3f}", flush=True)
    return summary
