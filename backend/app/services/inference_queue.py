"""
Inference queue / semaphore to prevent GPU OOM from concurrent requests.

CPU preprocessing can run concurrently; GPU inference is serialised by
the semaphore whose concurrency is controlled by MAX_GPU_CONCURRENCY.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.MAX_GPU_CONCURRENCY)
    return _semaphore


async def run_in_inference_queue(coro_fn, *args, **kwargs):
    """
    Await coro_fn(*args, **kwargs) under the GPU semaphore.

    Usage:
        result = await run_in_inference_queue(my_async_fn, arg1, arg2)
    """
    sem = get_semaphore()
    queue_depth = settings.MAX_GPU_CONCURRENCY - sem._value  # type: ignore[attr-defined]
    if queue_depth >= settings.MAX_GPU_CONCURRENCY:
        logger.info("GPU semaphore at capacity — request queued")
    async with sem:
        return await coro_fn(*args, **kwargs)
