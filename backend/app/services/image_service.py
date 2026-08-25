"""
Image service — validates uploads and wraps birefnet inference with
result caching and session history tracking.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from app.config import settings
from app.services.birefnet_service import birefnet
from app.utils.caching import ResultCache

logger = logging.getLogger(__name__)

MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

result_cache = ResultCache(
    ttl=settings.RESULT_CACHE_TTL,
    max_entries=200,
)


@dataclass
class HistoryEntry:
    id: str
    filename: str
    width: int
    height: int
    file_size: int
    mode: str
    processing_time_ms: float
    result_url: Optional[str]
    created_at: str


_session_history: list[HistoryEntry] = []
_MAX_HISTORY = 50


def validate_image(data: bytes) -> tuple[int, int, str]:
    """
    Returns (width, height, format) or raises ValueError.
    """
    if len(data) > MAX_BYTES:
        raise ValueError(
            f"File too large ({len(data) // (1024 * 1024):.1f} MB). "
            f"Maximum is {settings.MAX_UPLOAD_MB} MB."
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        fmt = img.format or "UNKNOWN"
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(
                f"Unsupported format '{fmt}'. Supported: {', '.join(ALLOWED_FORMATS)}"
            )
        return img.width, img.height, fmt
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Cannot decode image: {exc}") from exc


def process_image(
    image_bytes: bytes,
    filename: str,
    quality: str = "fast",
) -> tuple[bytes, dict]:
    """
    Remove background, applying cache when enabled.
    Returns (result_png_bytes, metadata).
    """
    model_id = birefnet._model_id or settings.BIREFNET_MODEL

    cache_hit = False
    if settings.ENABLE_RESULT_CACHE:
        cache_key = ResultCache.make_key(image_bytes, quality, model_id)
        entry = result_cache.get(cache_key)
        if entry is not None:
            logger.info("Cache hit for %s (quality=%s)", filename, quality)
            cache_hit = True
            result_bytes = entry.data
            metadata = dict(entry.metadata)
            metadata["cache_hit"] = True
        else:
            result_bytes, metadata = birefnet.remove_background(image_bytes, quality)
            metadata["cache_hit"] = False
            result_cache.set(cache_key, result_bytes, metadata)
    else:
        result_bytes, metadata = birefnet.remove_background(image_bytes, quality)
        metadata["cache_hit"] = False

    # Record in session history
    entry_id = str(uuid.uuid4())
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    history_entry = HistoryEntry(
        id=entry_id,
        filename=filename,
        width=metadata["input_width"],
        height=metadata["input_height"],
        file_size=len(image_bytes),
        mode=quality,
        processing_time_ms=metadata["total_time_ms"],
        result_url=f"/api/results/{entry_id}",
        created_at=created_at,
    )

    _session_history.insert(0, history_entry)
    if len(_session_history) > _MAX_HISTORY:
        _session_history.pop()

    # Store result for later retrieval (short TTL)
    _result_store[entry_id] = result_bytes
    while len(_result_store) > _MAX_RESULTS:
        oldest_id = next(iter(_result_store))
        if oldest_id == entry_id and len(_result_store) > 1:
            oldest_id = next(iter(list(_result_store.keys())[:-1]))
        _result_store.pop(oldest_id, None)

    metadata["history_id"] = entry_id
    return result_bytes, metadata


def get_history() -> list[HistoryEntry]:
    return list(_session_history)


# Temporary result store for /api/results/{id} endpoint (in-memory)
_result_store: dict[str, bytes] = {}
_MAX_RESULTS = 50


def get_result(result_id: str) -> bytes | None:
    return _result_store.get(result_id)
