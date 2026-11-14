"""Vision-language representation learning: everything the notebook of Lecture 6 implements from scratch.

* the two contrastive image-text losses: CLIP (softmax) and SigLIP (sigmoid);
* a dual encoder trained in the LiT setting (frozen image tower, text tower trained from scratch);
* zero-shot classification and image-text retrieval;
* the vision path of nanoVLM: SigLIP weights in the ViT of Lecture 1 and the pixel-shuffle projector.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_transformer import VisionTransformer


# ----------------------------------------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------------------------------------
def clip_loss(image_emb, text_emb, logit_scale):
    """Symmetric InfoNCE. image_emb, text_emb: (B, D) L2-normalized, row i of both is a matching pair.

    Every image is classified among the B texts of the batch and every text among the B images.
    """
    logits = logit_scale.exp() * image_emb @ text_emb.T                    # (B, B) scaled cosine similarities
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def siglip_loss(image_emb, text_emb, logit_scale, logit_bias):
    """Sigmoid loss: every (image, text) pair of the batch is an independent binary problem, match or not.

    No softmax over the batch: there is no normalization across pairs, so the loss does not need the full
    similarity matrix at once (it can be computed in chunks across devices).
    """
    logits = logit_scale.exp() * image_emb @ text_emb.T + logit_bias       # (B, B)
    labels = 2 * torch.eye(len(logits), device=logits.device) - 1           # +1 on the diagonal, -1 elsewhere
    return -F.logsigmoid(labels * logits).sum() / len(logits)


# ----------------------------------------------------------------------------------------------------------------------
# Towers
# ----------------------------------------------------------------------------------------------------------------------
class TextTransformer(nn.Module):
    """A small text encoder trained from scratch: token + position embeddings, pre-norm transformer blocks,
    mean over the non-padding tokens."""

    def __init__(self, vocab_size, max_len=32, width=256, depth=4, heads=4, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.width = width
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward(self, tokens):                                              # (B, L) token ids
        pad = tokens == self.pad_id
        x = self.token_embed(tokens) + self.pos_embed[:, :tokens.shape[1]]
        x = self.norm(self.blocks(x, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).float()
        return (x * keep).sum(1) / keep.sum(1).clamp_min(1)                # (B, width)


class PretrainedText(nn.Module):
    """A pretrained sentence encoder from the HuggingFace Hub (e.g. MiniLM), mean-pooled, fine-tuned end to end."""

    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name)
        self.width = self.model.config.hidden_size
        self.pad_id = self.model.config.pad_token_id or 0

    def forward(self, tokens):
        mask = (tokens != self.pad_id).long()
        x = self.model(input_ids=tokens, attention_mask=mask).last_hidden_state
        keep = mask.unsqueeze(-1).float()
        return (x * keep).sum(1) / keep.sum(1).clamp_min(1)


class DualEncoder(nn.Module):
    """Image and text towers, each followed by a linear projection into a shared embedding space.

    In the LiT setting ("locked image, unlocked text") the image tower is a frozen pretrained network, so the
    image side receives its pre-computed features and only the projection is trained.
    """

    def __init__(self, image_dim, text_tower, embed_dim=256, loss="siglip"):
        super().__init__()
        assert loss in ("clip", "siglip")
        self.loss_type = loss
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, embed_dim))
        self.text_tower = text_tower
        self.text_proj = nn.Linear(text_tower.width, embed_dim)
        if loss == "clip":   # temperature 0.07 as in CLIP
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            self.logit_bias = None
        else:                # SigLIP: t = 10, b = -10, so that at the start almost all pairs are predicted "no match"
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def encode_image(self, features):
        return F.normalize(self.image_proj(features.float()), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.text_proj(self.text_tower(tokens).float()), dim=-1)

    def forward(self, features, tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=features.is_cuda):
            img, txt = self.encode_image(features), self.encode_text(tokens)
        img, txt = img.float(), txt.float()
        self.logit_scale.data.clamp_(max=math.log(100))                     # as in CLIP
        if self.loss_type == "clip":
            return clip_loss(img, txt, self.logit_scale)
        return siglip_loss(img, txt, self.logit_scale, self.logit_bias)


# ----------------------------------------------------------------------------------------------------------------------
# Zero-shot classification and retrieval
# ----------------------------------------------------------------------------------------------------------------------
TEMPLATES = [   # a subset of the 80 ImageNet prompts of CLIP
    "a photo of a {}.", "a bad photo of a {}.", "a photo of many {}.", "a photo of the large {}.",
    "a photo of the small {}.", "a close-up photo of a {}.", "a bright photo of a {}.", "a dark photo of a {}.",
    "a blurry photo of a {}.", "a cropped photo of a {}.", "a low resolution photo of a {}.", "a good photo of a {}.",
    "a photo of the {}.", "a photo of my {}.", "art of the {}.", "a drawing of a {}.",
]


@torch.no_grad()
def zero_shot_weights(encode_text, classnames, templates=("a photo of a {}.",)):
    """One "classifier weight" per class: the mean of the normalized text embeddings of its prompts.

    encode_text: list of strings -> (N, D) L2-normalized embeddings.
    """
    weights = []
    for name in classnames:
        emb = encode_text([t.format(name) for t in templates])
        weights.append(F.normalize(emb.float().mean(0), dim=-1))
    return torch.stack(weights)                                             # (C, D)


@torch.no_grad()
def zero_shot_accuracy(image_emb, labels, class_weights):
    return ((image_emb.float() @ class_weights.T).argmax(1) == labels).float().mean().item()


@torch.no_grad()
def retrieval_metrics(image_emb, text_emb, text_to_image, ks=(1, 5, 10)):
    """image_emb (Ni, D), text_emb (Nt, D) normalized; text_to_image (Nt,) index of the image of each caption.

    Image -> text: an image is a hit at k if any of its captions is among the k most similar captions.
    Text -> image: a caption is a hit at k if its image is among the k most similar images.
    """
    sims = image_emb.float() @ text_emb.float().T                           # (Ni, Nt)
    correct = text_to_image[None, :] == torch.arange(len(image_emb), device=sims.device)[:, None]
    # rank of the best correct caption of every image = number of captions scored above it
    best = sims.masked_fill(~correct, -float("inf")).max(1).values
    i2t_rank = (sims > best[:, None]).sum(1)
    # rank of the correct image of every caption
    t2i_rank = (sims.T > sims.T.gather(1, text_to_image[:, None])).sum(1)
    out = {}
    for k in ks:
        out[f"i2t_R@{k}"] = (i2t_rank < k).float().mean().item()
        out[f"t2i_R@{k}"] = (t2i_rank < k).float().mean().item()
    out["mean_recall"] = sum(out.values()) / len(out)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# The vision path of a VLM (nanoVLM)
# ----------------------------------------------------------------------------------------------------------------------
def load_siglip_vision(name="google/siglip2-base-patch16-512"):
    """Loads the image tower of a SigLIP / SigLIP 2 checkpoint into the ViT of Lecture 1.

    The same mapping is done by nanoVLM (models/vision_transformer.py). SigLIP has no [CLS] token: the ViT returns
    one token per patch, and these tokens are what a VLM passes to the language model.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(name).vision_config
    vit = VisionTransformer(img_size=cfg.image_size, patch_size=cfg.patch_size, num_classes=0,
                            embed_dim=cfg.hidden_size, depth=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
                            mlp_ratio=cfg.intermediate_size / cfg.hidden_size, global_pool="avg", gelu_approx="tanh")
    sd = {}
    with safe_open(hf_hub_download(name, "model.safetensors"), framework="pt") as f:
        get = lambda k: f.get_tensor("vision_model." + k)
        sd["patch_embed.proj.weight"] = get("embeddings.patch_embedding.weight")
        sd["patch_embed.proj.bias"] = get("embeddings.patch_embedding.bias")
        sd["pos_embed"] = get("embeddings.position_embedding.weight")[None]
        sd["norm.weight"], sd["norm.bias"] = get("post_layernorm.weight"), get("post_layernorm.bias")
        for i in range(cfg.num_hidden_layers):
            src, dst = f"encoder.layers.{i}.", f"blocks.{i}."
            for a, b in [("layer_norm1", "norm1"), ("layer_norm2", "norm2"), ("mlp.fc1", "mlp.fc1"),
                         ("mlp.fc2", "mlp.fc2"), ("self_attn.out_proj", "attn.proj")]:
                for p in ("weight", "bias"):
                    sd[f"{dst}{b}.{p}"] = get(f"{src}{a}.{p}")
            for p in ("weight", "bias"):   # q, k, v are separate in HF and one matrix in our Attention
                sd[f"{dst}attn.qkv.{p}"] = torch.cat([get(f"{src}self_attn.{x}_proj.{p}") for x in "qkv"])
    vit.load_state_dict(sd)
    return vit.eval()


def pixel_shuffle(x, factor=4):
    """nanoVLM's modality projector, first half: (B, N, D) patch tokens on a square grid -> (B, N / f^2, D * f^2).

    Every f x f group of neighbouring tokens is concatenated into one token, so the language model receives
    f^2 times fewer (and wider) image tokens. A linear layer then maps them to the width of the language model.
    """
    B, N, D = x.shape
    h = int(math.isqrt(N))
    assert h * h == N and h % factor == 0
    x = x.reshape(B, h // factor, factor, h // factor, factor, D)           # (B, H', f, W', f, D)
    x = x.permute(0, 1, 3, 2, 4, 5)                                         # (B, H', W', f, f, D)
    return x.reshape(B, (h // factor) ** 2, factor * factor * D)
