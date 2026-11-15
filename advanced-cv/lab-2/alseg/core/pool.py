"""Identity of the official segmentation pool and evaluation split."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PoolManifest:
    """Ordered VOC filename stems, stored separately for official train and val."""

    pool_ids: list[str]
    val_ids: list[str]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()

    def __post_init__(self) -> None:
        for name, ids in (("pool_ids", self.pool_ids), ("val_ids", self.val_ids)):
            if not ids or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError(f"{name} must contain nonempty filename stems")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} contains duplicates")
        if set(self.pool_ids) & set(self.val_ids):
            raise ValueError("Official train and val IDs must be disjoint")
