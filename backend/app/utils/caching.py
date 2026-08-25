import hashlib
import time
from dataclasses import dataclass


@dataclass
class CacheEntry:
    data: bytes
    created_at: float
    metadata: dict


class ResultCache:
    def __init__(self, ttl: int = 3600, max_entries: int = 100):
        self._ttl = ttl
        self._max = max_entries
        self._store: dict[str, CacheEntry] = {}

    @staticmethod
    def make_key(image_bytes: bytes, quality: str, model_id: str) -> str:
        h = hashlib.sha256()
        h.update(image_bytes)
        h.update(quality.encode())
        h.update(model_id.encode())
        return h.hexdigest()

    def get(self, key: str) -> CacheEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() - entry.created_at > self._ttl:
            del self._store[key]
            return None
        return entry

    def set(self, key: str, data: bytes, metadata: dict) -> None:
        # Evict oldest entries when at capacity
        if len(self._store) >= self._max:
            oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest_key]
        self._store[key] = CacheEntry(
            data=data, created_at=time.time(), metadata=metadata
        )

    def __len__(self) -> int:
        return len(self._store)
