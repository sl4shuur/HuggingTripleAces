"""Implementation contracts shared by the runner and concrete adapters."""

from abc import ABC, abstractmethod

from torch import Tensor

from alseg.core.selection import SelectionRequest


class Selector(ABC):
    """Acquisition boundary; implementations must not mutate the request."""

    @abstractmethod
    def select(self, request: SelectionRequest) -> list[int]:
        """Return exactly n_select new pool IDs, ordered by acquisition priority."""
        ...


class ImageSource(ABC):
    """Images indexed by permanent split-local IDs; no mask access."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_image(self, image_id: int) -> Tensor:
        """Return an unnormalized RGB uint8 CPU tensor [3,H,W]."""
        ...


class EvaluationSource(ImageSource):
    """Official val access, available to evaluation code only."""

    @abstractmethod
    def get_mask(self, image_id: int) -> Tensor:
        """Return an int64 CPU mask [H,W], preserving ignore label 255."""
        ...


class Oracle(ABC):
    """The runner's annotation capability; never pass it to selectors."""

    @property
    @abstractmethod
    def labeled_ids(self) -> list[int]: ...

    @abstractmethod
    def query(self, ids: list[int]) -> dict[int, Tensor]:
        """Atomically annotate new pool IDs; reject repeated/invalid queries."""
        ...

    @abstractmethod
    def get_mask(self, image_id: int) -> Tensor:
        """Read an already-queried mask without consuming annotation budget."""
        ...
