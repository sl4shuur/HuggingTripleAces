"""Single-device Lightning training of the head; official val never drives fitting."""

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from alseg.core.predictions import IGNORE_INDEX, NUM_CLASSES
from alseg.metrics import confusion_matrix, segmentation_metrics
from alseg.nn.heads import LinearSegmentationHead, segmentation_loss


class SegmentationTraining(L.LightningModule):
    def __init__(self, feature_dim: int, learning_rate: float, weight_decay: float):
        super().__init__()
        self.head = LinearSegmentationHead(feature_dim)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.history: list[dict] = []
        self.register_buffer(
            "train_confusion",
            torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "loss_sum", torch.zeros((), dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "valid_count", torch.zeros((), dtype=torch.int64), persistent=False
        )

    def on_train_epoch_start(self) -> None:
        self.train_confusion.zero_()
        self.loss_sum.zero_()
        self.valid_count.zero_()

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor | None:
        features, targets = batch
        count = (targets != IGNORE_INDEX).sum()
        if count == 0:
            return None
        logits = self.head(features, targets.shape[-2:])
        loss = segmentation_loss(logits, targets)
        self.loss_sum += loss.detach().double() * count
        self.valid_count += count
        self.train_confusion += confusion_matrix(logits.detach().argmax(dim=1), targets)
        self.log(
            "train/loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            batch_size=len(features),
        )
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = segmentation_metrics(self.train_confusion)
        record = {
            "epoch": self.current_epoch,
            "train/loss": (self.loss_sum / self.valid_count).item(),
            "train/miou": metrics["miou"],
            "train/pixel_accuracy": metrics["pixel_accuracy"],
            "train/lr": self.learning_rate,
            "train/valid_pixels": int(self.valid_count),
        }
        self.history.append(record)
        self.log_dict(
            {name: value for name, value in record.items() if name != "epoch"}
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def fit_head(
    features: Tensor,
    targets: Tensor,
    config: dict,
    seed: int,
    device: str,
    logger=False,
) -> SegmentationTraining:
    """Every call resets the head and optimizer; features remain outside the trainer."""
    if len(features) == 0 or len(features) != len(targets):
        raise ValueError("Training requires aligned nonempty features and masks")
    if not (targets != IGNORE_INDEX).any():
        raise ValueError("Training set contains no labeled pixels")
    if type(config["epochs"]) is not int or config["epochs"] <= 0:
        raise ValueError("epochs must be a positive integer")
    L.seed_everything(seed, workers=True)
    task = SegmentationTraining(
        features.shape[1], config["learning_rate"], config["weight_decay"]
    )
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision="32-true",
        deterministic=True,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(task, train_dataloaders=loader)
    task.head.cpu().eval()
    return task
