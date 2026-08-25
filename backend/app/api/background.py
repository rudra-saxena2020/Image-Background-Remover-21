"""
Background removal API endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.services.image_service import (
    get_history,
    get_result,
    process_image,
    validate_image,
)
from app.services.inference_queue import get_semaphore
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    quality: str = Form(default="fast"),
):
    if quality not in ("fast", "high"):
        raise HTTPException(status_code=422, detail="quality must be 'fast' or 'high'")

    # Read in bounded chunks so a malicious Content-Length cannot force an
    # unbounded allocation before image validation runs.
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await image.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB upload limit",
            )
        chunks.append(chunk)
    raw = b"".join(chunks)
    filename = image.filename or "image.png"

    # Validate
    try:
        width, height, fmt = validate_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Run inference under semaphore to prevent GPU OOM
    sem = get_semaphore()
    try:
        async with sem:
            loop = asyncio.get_event_loop()
            result_bytes, metadata = await loop.run_in_executor(
                None,
                process_image,
                raw,
                filename,
                quality,
            )
    except RuntimeError as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})
    except Exception as exc:
        logger.error("Unexpected error during inference: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "Internal processing error"})

    headers = {
        "X-Inference-Time": f"{metadata.get('inference_time_ms', 0):.0f}",
        "X-Total-Time": f"{metadata.get('total_time_ms', 0):.0f}",
        "X-Input-Width": str(metadata.get("input_width", width)),
        "X-Input-Height": str(metadata.get("input_height", height)),
        "X-Inference-Width": str(metadata.get("inference_width", 0)),
        "X-Inference-Height": str(metadata.get("inference_height", 0)),
        "X-Device": str(metadata.get("device", "unknown")),
        "X-Mode": quality,
        "X-Cache-Hit": "true" if metadata.get("cache_hit") else "false",
        "X-History-Id": str(metadata.get("history_id", "")),
        "Content-Disposition": f'attachment; filename="{filename.rsplit(".", 1)[0]}_nobg.png"',
    }

    return Response(
        content=result_bytes,
        media_type="image/png",
        headers=headers,
    )


@router.get("/history")
async def history():
    entries = get_history()
    return [asdict(e) for e in entries]


@router.get("/results/{result_id}")
async def download_result(result_id: str):
    data = get_result(result_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Result not found or expired")
    return Response(
        content=data,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{result_id}.png"'},
    )


# ── Batch endpoints ────────────────────────────────────────────────────────────
# In-memory batch registry
_batches: dict[str, dict] = {}


@router.post("/remove-background/batch")
async def batch_remove_background():
    """
    Batch processing is currently frontend-driven (sequential single-image calls).
    This stub exists for future server-side batch queue support.
    """
    batch_id = str(uuid.uuid4())
    _batches[batch_id] = {
        "batch_id": batch_id,
        "total": 0,
        "completed": 0,
        "failed": 0,
        "status": "pending",
        "items": [],
    }
    return JSONResponse(content=_batches[batch_id], status_code=202)


@router.get("/remove-background/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    batch = _batches.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch
