"""Server-side proxy to the RunPod FLUX image generation server.

The RUNPOD_URL secret never leaves the backend. External callers POST a
source image and prompt here; this service forwards both to RunPod and
streams the binary image back to the caller.

Route: POST /api/generate
"""

from __future__ import annotations

import logging
import mimetypes
import secrets
import urllib.error
import urllib.request

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageFilter
from io import BytesIO
import numpy as np
from app.config import settings
from app.services.runpod_qwen_image_edit_service import (
    RunPodQwenError,
    generate_frame as generate_qwen_frame,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_MIN_PROMPT = 1


def _looks_like_multi_view_collage(source: bytes) -> bool:
    """Reject panoramic reference boards before an image-edit model can copy them.

    A single product photo can be wide, so this intentionally only catches the
    strong montage signal: a panoramic canvas wider than a normal 16:9 frame.
    The UI still allows normal portrait, square, and moderately landscape
    source photos.
    """
    try:
        with Image.open(BytesIO(source)) as image:
            width, height = image.size
    except Exception:
        return False
    return width / max(height, 1) >= 1.85


def _validate_single_product_output(image_bytes: bytes) -> None:
    """Reject obvious multi-product outputs before they reach the gallery.

    This is intentionally a conservative screen, not a replacement for
    identity review. It looks for two or more large disconnected foreground
    regions against the neutral studio background used by this workflow.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            width, height = opened.size
            if width / max(height, 1) >= 1.85:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Generated output was rejected because it looks like a "
                        "multi-view collage or contact sheet. No image was added "
                        "to the gallery."
                    ),
                )
            if width < 256 or height < 256:
                return
            image = np.asarray(opened.convert("RGB").resize((256, 256)), dtype=np.int16)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Generated output is not a readable image.") from exc

    # Compare against a locally blurred background rather than a single border
    # color. Studio floors often have a natural gradient that otherwise joins
    # every product and shadow into one giant connected component.
    blurred = np.asarray(
        Image.fromarray(image.astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
        dtype=np.int16,
    )
    foreground = np.linalg.norm(image - blurred, axis=2) > 28
    visited = np.zeros((256, 256), dtype=bool)
    components: list[int] = []
    for row in range(256):
        for column in range(256):
            if not foreground[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            area = 0
            while stack:
                current_row, current_column = stack.pop()
                area += 1
                for next_row in range(max(0, current_row - 1), min(256, current_row + 2)):
                    for next_column in range(max(0, current_column - 1), min(256, current_column + 2)):
                        if (
                            foreground[next_row, next_column]
                            and not visited[next_row, next_column]
                        ):
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))
            if area >= 256:
                components.append(area)

    if len(components) >= 2:
        raise HTTPException(
            status_code=502,
            detail=(
                "Generated output was rejected because more than one substantial "
                "product-like object was detected. No image was added to the gallery."
            ),
        )


def _multipart_body(prompt: str, filename: str, content_type: str, image: bytes) -> tuple[bytes, str]:
    boundary = f"----AtelierRunPod{secrets.token_hex(12)}"
    chunks = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n{prompt}\r\n".encode(),
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
        ).encode(),
        image,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _runpod_generate(prompt: str, filename: str, content_type: str, image: bytes) -> tuple[bytes, str]:
    """Call the RunPod FLUX image-to-image endpoint."""
    if not settings.RUNPOD_GENERATE_ENABLED or not settings.RUNPOD_URL:
        raise HTTPException(status_code=503, detail="Image generator is not configured.")

    body, multipart_type = _multipart_body(prompt, filename, content_type, image)

    request = urllib.request.Request(
        settings.RUNPOD_URL,
        data=body,
        method="POST",
        headers={
            "Accept": "image/png, image/jpeg, image/*",
            "Content-Type": multipart_type,
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.RUNPOD_GENERATE_REQUEST_TIMEOUT_SECONDS
        ) as response:
            content_type: str = response.headers.get("Content-Type", "image/png")
            image_bytes: bytes = response.read(settings.RUNPOD_GENERATE_MAX_RESULT_BYTES + 1)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("RunPod returned HTTP %s: %s", exc.code, body[:300])
        raise HTTPException(
            status_code=502,
            detail=f"Image server returned HTTP {exc.code}.",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error("RunPod unreachable: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="Image server could not be reached.",
        ) from exc

    if len(image_bytes) > settings.RUNPOD_GENERATE_MAX_RESULT_BYTES:
        raise HTTPException(status_code=502, detail="Image server returned an oversized response.")

    if not image_bytes:
        raise HTTPException(status_code=502, detail="Image server returned an empty response.")

    if not content_type.startswith("image/"):
        logger.error("RunPod returned unexpected content-type: %s", content_type)
        raise HTTPException(status_code=502, detail="Image server returned an unexpected content type.")

    return image_bytes, content_type


@router.post("/generate")
async def generate_image(
    prompt: str = Form(..., min_length=_MIN_PROMPT),
    image: list[UploadFile] = File(...),
) -> Response:
    """
    Transform a source image using a prompt via the RunPod FLUX server.

    Returns binary image data (image/png or image/jpeg).
    Errors are returned as JSON with an `error` key.
    """
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")
    if not image or len(image) > 3:
        raise HTTPException(status_code=400, detail="Provide between 1 and 3 source images.")
    references: list[tuple[bytes, str, str]] = []
    for upload in image:
        declared_content_type = upload.content_type or ""
        guessed_content_type = mimetypes.guess_type(upload.filename or "")[0] or ""
        content_type = declared_content_type if declared_content_type.startswith("image/") else guessed_content_type
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Every source file must be a valid image.")
        source = await upload.read(settings.RUNPOD_GENERATE_MAX_RESULT_BYTES + 1)
        if not source:
            raise HTTPException(status_code=400, detail="A source image is empty.")
        if len(source) > settings.RUNPOD_GENERATE_MAX_RESULT_BYTES:
            raise HTTPException(status_code=400, detail="A source image is too large.")
        if _looks_like_multi_view_collage(source):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Product identity lock requires one original product photo. "
                    "The uploaded reference looks like a multi-view collage or "
                    "panoramic product board. Upload a single unobstructed product "
                    "image so shape, color, hardware, and construction can remain exact."
                ),
            )
        references.append((source, upload.filename or "reference-image", content_type))

    logger.info("Generating image for prompt (%d chars)", len(prompt))
    try:
        if settings.QWEN_RUNPOD_ENABLED and settings.RUNPOD_API_KEY:
            # The private FLUX pod is text-to-image only (GET /generate).
            # Use Qwen as the primary image-to-image provider whenever it is
            # configured; it accepts the source image as a data URL.
            try:
                qwen_result = await generate_qwen_frame(
                    references=references,
                    prompt=prompt,
                    seed=secrets.randbelow(2_147_483_647),
                )
                image_bytes, content_type = qwen_result.image_bytes, "image/png"
                logger.info("Image generated through RunPod Qwen image-edit")
            except RunPodQwenError as qwen_error:
                raise HTTPException(status_code=502, detail=str(qwen_error)) from qwen_error
        else:
            image_bytes, content_type = _runpod_generate(
                prompt,
                references[0][1],
                references[0][2],
                references[0][0],
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during image generation")
        raise HTTPException(status_code=502, detail="Image generation failed.") from exc

    _validate_single_product_output(image_bytes)
    logger.info("Image generation complete — %d bytes, %s", len(image_bytes), content_type)
    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/generate/status")
async def generate_status() -> dict[str, object]:
    """Report whether the RunPod image generation endpoint is configured."""
    return {
        "configured": settings.RUNPOD_GENERATE_ENABLED,
        "ready": settings.RUNPOD_GENERATE_ENABLED,
        "reason": (
            (
                "RunPod Qwen image-to-image is available."
                if settings.QWEN_RUNPOD_ENABLED and settings.RUNPOD_API_KEY
                else "Private endpoint is configured, but it only supports text-to-image."
            )
            if settings.RUNPOD_GENERATE_ENABLED
            else "RUNPOD_URL secret is not set."
        ),
    }
