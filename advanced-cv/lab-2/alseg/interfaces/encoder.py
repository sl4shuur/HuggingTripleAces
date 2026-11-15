"""Encoder contracts shared with concrete implementations."""

from abc import ABC, abstractmethod

from torch import Tensor

from alseg.core.features import EncoderFeatures
from alseg.interfaces.data import ImageSource


class FeatureSource(ABC):
    """ID-aligned frozen features, computed from images or read from a cache."""

    @abstractmethod
    def get_features(self, source: ImageSource, ids: list[int]) -> EncoderFeatures: ...

    @abstractmethod
    def describe(self) -> dict: ...


class Encoder(FeatureSource):
    """Frozen feature extraction, independent of image storage and labels."""

    def get_features(self, source: ImageSource, ids: list[int]) -> EncoderFeatures:
        return self.encode([source.get_image(i) for i in ids])

    @abstractmethod
    def encode(self, images: list[Tensor]) -> EncoderFeatures:
        """Encode raw RGB uint8 CPU [3,H,W] images, preserving batch order."""
        ...

    @abstractmethod
    def describe(self) -> dict:
        """Return checkpoint, preprocessing and feature identity for artifacts."""
        ...
