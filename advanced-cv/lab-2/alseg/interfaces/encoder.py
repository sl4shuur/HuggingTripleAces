"""Encoder contracts shared with concrete implementations."""

from abc import ABC, abstractmethod

from torch import Tensor

from alseg.core.features import EncoderFeatures


class Encoder(ABC):
    """Frozen feature extraction, independent of image storage and labels."""

    @abstractmethod
    def encode(self, images: list[Tensor]) -> EncoderFeatures:
        """Encode raw RGB uint8 CPU [3,H,W] images, preserving batch order."""
        ...

    @abstractmethod
    def describe(self) -> dict:
        """Return checkpoint, preprocessing and feature identity for artifacts."""
        ...
