"""Small, fold-local DINO-style continuation; not full DINOv3 pretraining."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from soilcv.artifacts import ComputeMeter, model_state_hash, write_json
from soilcv.nn import make_param_groups, set_trainable

ADAPTATION = {
    "epochs": 10,
    "batch_size": 32,
    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "layer_decay": 0.75,
    "weight_decay": 0.04,
    "student_temperature": 0.1,
    "teacher_temperature": 0.04,
    "center_momentum": 0.9,
    "teacher_momentum": 0.996,
    "prototypes": 256,
    "views": "two full 64mm tiles; dihedral + mild color; no zoom/resize crops",
    "selection": "fixed final EMA teacher; no validation labels",
}


class DINOHead(nn.Module):
    def __init__(self, size, prototypes=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(size, 512), nn.GELU(), nn.Linear(512, 128))
        self.prototypes = nn.Linear(128, prototypes, bias=False)
        nn.init.normal_(self.prototypes.weight, std=0.02)

    def forward(self, features):
        return F.linear(
            F.normalize(self.mlp(features), dim=-1),
            F.normalize(self.prototypes.weight, dim=-1),
        )


def dino_loss(
    student, teacher, center, *, student_temperature=0.1, teacher_temperature=0.04
):
    """Two crossed views; teacher probabilities are explicitly detached."""
    if len(student) != 2 or len(teacher) != 2:
        raise ValueError("Use two student views and two teacher views.")
    probability = [
        ((value.detach().float() - center) / teacher_temperature).softmax(-1)
        for value in teacher
    ]
    log_probability = [
        (value.float() / student_temperature).log_softmax(-1) for value in student
    ]
    loss = (
        -sum((probability[i] * log_probability[1 - i]).sum(-1).mean() for i in range(2))
        / 2
    )
    return loss, probability


@torch.no_grad()
def ema_update(teacher, student, momentum):
    for target, source in zip(teacher.parameters(), student.parameters(), strict=True):
        target.lerp_(source, 1 - momentum)
    for target, source in zip(teacher.buffers(), student.buffers(), strict=True):
        target.copy_(source)


def adapt_encoder(
    encoder, loader, folder, *, fold, device="cuda:0", run=None, settings=None
):
    settings = dict(ADAPTATION if settings is None else settings)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = loader.dataset.manifest
    if manifest.fold.isna().any() or manifest.fold.eq(fold).any():
        raise ValueError("Adaptation includes a held-out or test photo.")
    write_json(
        folder / "config.json",
        {
            **settings,
            "fold": fold,
            "train_ids": sorted(manifest.sample_id.unique()),
            "photos": len(manifest[["sample_id", "photo"]].drop_duplicates()),
            "tiles": len(manifest),
            "initial_state_hash": model_state_hash(encoder),
        },
    )
    encoder.to(device).train()
    head = DINOHead(encoder.output_dim, settings["prototypes"]).to(device)
    student = nn.ModuleList([encoder, head])
    teacher = set_trainable(copy.deepcopy(student), False).eval()
    groups = make_param_groups(
        encoder,
        lr=settings["backbone_lr"],
        weight_decay=settings["weight_decay"],
        layer_ids=encoder.layer_ids(),
        layer_decay=settings["layer_decay"],
        no_weight_decay=encoder.no_weight_decay(),
    )
    groups += make_param_groups(
        head, lr=settings["head_lr"], weight_decay=settings["weight_decay"]
    )
    optimizer = torch.optim.AdamW(groups)
    center = torch.zeros(1, settings["prototypes"], device=device)
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, step = [], 0
    if run is not None:
        run.define_metric("adaptation/*", step_metric="epoch")
    total_steps = settings["epochs"] * len(loader)
    with ComputeMeter(device, output_path=folder / "compute.json"):
        for epoch in range(settings["epochs"]):
            totals = {
                key: 0.0
                for key in [
                    "loss",
                    "teacher_entropy",
                    "marginal_entropy",
                    "feature_std",
                ]
            }
            seen, skipped = 0, 0
            for first, second in loader:
                views = [first.to(device), second.to(device)]
                progress = step / max(1, total_steps - 1)
                rate = min(1.0, (step + 1) / max(1, len(loader))) * (
                    0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
                )
                for group, initial in zip(optimizer.param_groups, groups, strict=True):
                    group.setdefault("initial_lr", initial["lr"])
                    group["lr"] = group["initial_lr"] * rate
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=torch.device(device).type, enabled=use_amp
                ):
                    features = [encoder(view) for view in views]
                    outputs = [head(value) for value in features]
                    with torch.no_grad():
                        targets = [teacher[1](teacher[0](view)) for view in views]
                    loss, probability = dino_loss(
                        outputs,
                        targets,
                        center,
                        student_temperature=settings["student_temperature"],
                        teacher_temperature=settings["teacher_temperature"],
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite adaptation loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    if scaler.get_scale() >= previous:
                        momentum = (
                            1
                            - (1 - settings["teacher_momentum"])
                            * (1 + math.cos(math.pi * progress))
                            / 2
                        )
                        ema_update(teacher, student, momentum)
                        center.lerp_(
                            torch.cat(targets).float().mean(0, keepdim=True),
                            1 - settings["center_momentum"],
                        )
                    else:
                        skipped += 1
                    p = torch.cat(probability).clamp_min(1e-8)
                    mean = p.mean(0)
                    batch = len(first)
                    totals["loss"] += loss.item() * batch
                    totals["teacher_entropy"] += (
                        -(p * p.log()).sum(-1).mean()
                    ).item() * batch
                    totals["marginal_entropy"] += (
                        -(mean * mean.log()).sum()
                    ).item() * batch
                    totals["feature_std"] += (
                        features[0].float().std(0, unbiased=False).mean().item() * batch
                    )
                    seen += batch
                    step += 1
            record = {
                "epoch": epoch,
                **{key: value / seen for key, value in totals.items()},
                "skipped_steps": skipped,
            }
            history.append(record)
            write_json(folder / "history.json", history)
            print("adaptation", fold, record, flush=True)
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        **{
                            f"adaptation/{key}": value
                            for key, value in record.items()
                            if key != "epoch"
                        },
                    }
                )
    encoder.load_state_dict(teacher[0].state_dict())
    set_trainable(encoder, True)
    torch.save(encoder.state_dict(), folder / "encoder.pt")
    write_json(
        folder / "state.json",
        {"state_hash": model_state_hash(encoder), "epoch": settings["epochs"] - 1},
    )
    return encoder.eval(), history
