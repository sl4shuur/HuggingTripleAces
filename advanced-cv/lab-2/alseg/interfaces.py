"""Implementation contracts shared by the runner and concrete adapters."""

from abc import ABC, abstractmethod

from alseg.core.selection import SelectionRequest


class Selector(ABC):
    """Acquisition boundary; implementations must not mutate the request."""

    @abstractmethod
    def select(self, request: SelectionRequest) -> list[int]:
        """Return exactly n_select new pool IDs, ordered by acquisition priority."""
        ...
