"""RunPod public endpoint bridge for Qwen Image Edit 2511.

The RunPod credential and provider request remain server-side. The provider
returns a short-lived image URL, so this service downloads the image before
passing it into Atelier's existing validation pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


DEFAULT_ENDPOINT = "https://api.runpod.ai/v2/qwen-image-edit-2511/runsync"
PRICING_SOURCE = "https://docs.runpod.io/public-endpoints/models/qwen-image-edit-2511"
SUPPORTED_SIZES = {
    "1024*1024",
    "1024*1280",
    "1280*1024",
    "1280*1280",
    "1280*1536",
    "1536*1080",
}


class RunPodQwenError(RuntimeError):
    """A safe, provider-specific error that can be shown in shoot status."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RunPodGeneratedFrame:
    image_bytes: bytes
    request_id: str
    result_url: str
    cost: float


def _endpoint() -> str:
    return settings.QWEN_RUNPOD_ENDPOINT.rstrip("/") or DEFAULT_ENDPOINT


def _configured() -> bool:
    return settings.QWEN_RUNPOD_ENABLED and bool(settings.RUNPOD_API_KEY)


def status() -> dict[str, object]:
    configured = _configured()
    return {
        "id": "qwen-runpod",
        "name": "Qwen Image Edit 2511 · RunPod",
        "mode": "hosted-paid",
        "provider": "RunPod",
        "model": "qwen-image-edit-2511",
        "endpoint": _endpoint(),
        "configured": configured,
        "runtime_ready": configured,
        "ready": configured,
        "supports_source_images": True,
        "supports_multi_reference": True,
        "max_references": 3,
        "human_product_verified": False,
        "verification_state": "unverified",
        "provider_billing": "RunPod public endpoint usage-based API",
        "pricing": {
            "unit": "image",
            "rate_usd": settings.QWEN_RUNPOD_PRICE_USD_PER_IMAGE,
            "estimated_cost_usd": settings.QWEN_RUNPOD_PRICE_USD_PER_IMAGE,
            "source": PRICING_SOURCE,
        },
        "reason": (
            "Available through the configured RunPod API key. Each output is "
            "downloaded and validated before delivery."
            if configured
            else "RunPod Qwen Image Edit requires the RUNPOD_API_KEY secret."
        ),
    }


def _data_url_references(
    references: list[tuple[bytes, str, str]],
) -> list[str]:
    return [
        f"data:{content_type or 'image/png'};base64,{base64.b64encode(content).decode('ascii')}"
        for content, _filename, content_type in references
    ]


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    if not _configured():
        raise RunPodQwenError("RUNPOD_UNAVAILABLE", str(status()["reason"]))
    request = urllib.request.Request(
        _endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.QWEN_RUNPOD_REQUEST_TIMEOUT_SECONDS
        ) as response:
            raw = response.read()
            result = json.loads(raw.decode("utf-8"))
            output = result.get("output") if isinstance(result, dict) else None
            logger.info(
                "RunPod Qwen response shape: top_level=%s output_type=%s output_keys=%s",
                sorted(result.keys()) if isinstance(result, dict) else type(result).__name__,
                type(output).__name__,
                sorted(output.keys()) if isinstance(output, dict) else None,
            )
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RunPodQwenError(
                "RUNPOD_AUTH_FAILED",
                "RunPod rejected the configured API key.",
            ) from exc
        raise RunPodQwenError(
            "RUNPOD_REQUEST_FAILED",
            f"RunPod returned HTTP {exc.code}.",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RunPodQwenError(
            "RUNPOD_UNAVAILABLE",
            "The RunPod Qwen endpoint could not be reached.",
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RunPodQwenError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned invalid JSON.",
        ) from exc
    if not isinstance(result, dict):
        raise RunPodQwenError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned an invalid response object.",
        )
    return result


def _download_result(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    allowed_hosts = {
        "image.runpod.ai",
        "d2h7xmz5gqybh9.cloudfront.net",
    }
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        logger.error("RunPod returned unsupported output host: %s", parsed.hostname or "missing")
        raise RunPodQwenError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned an unsafe output image URL.",
        )
    request = urllib.request.Request(url, headers={"Accept": "image/*"})
    try:
        with urllib.request.urlopen(
            request, timeout=settings.QWEN_RUNPOD_REQUEST_TIMEOUT_SECONDS
        ) as response:
            chunks: list[bytes] = []
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > settings.QWEN_RUNPOD_MAX_RESULT_BYTES:
                    raise RunPodQwenError(
                        "RUNPOD_RESULT_TOO_LARGE",
                        "RunPod returned an output larger than Atelier's safety limit.",
                    )
                chunks.append(chunk)
            output = b"".join(chunks)
    except RunPodQwenError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RunPodQwenError(
            "RUNPOD_RESULT_DOWNLOAD_FAILED",
            "The RunPod output image could not be downloaded for validation.",
        ) from exc
    if not output:
        raise RunPodQwenError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned an empty output image.",
        )
    return output


def _decode_inline_image(value: object) -> bytes | None:
    if not isinstance(value, str):
        return None
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None


def _extract_image_output(value: object) -> tuple[str | None, bytes | None]:
    """Handle the output.image, output.images, and nested URL shapes."""
    if isinstance(value, str):
        if value.startswith("https://"):
            return value, None
        return None, _decode_inline_image(value)
    if isinstance(value, list):
        for item in value:
            image_url, image_bytes = _extract_image_output(item)
            if image_url or image_bytes:
                return image_url, image_bytes
    if isinstance(value, dict):
        for key in ("image_url", "image", "url", "images", "result", "output"):
            image_url, image_bytes = _extract_image_output(value.get(key))
            if image_url or image_bytes:
                return image_url, image_bytes
    return None, None


async def generate_frame(
    *,
    references: list[tuple[bytes, str, str]],
    prompt: str,
    seed: int,
) -> RunPodGeneratedFrame:
    if not 1 <= len(references) <= 3:
        raise RunPodQwenError(
            "RUNPOD_INVALID_INPUT",
            "RunPod Qwen Image Edit accepts between 1 and 3 reference images.",
        )
    size = settings.QWEN_RUNPOD_SIZE
    if size not in SUPPORTED_SIZES:
        raise RunPodQwenError(
            "RUNPOD_INVALID_INPUT",
            f"Unsupported RunPod Qwen output size: {size}.",
        )
    payload = {
        "input": {
            "prompt": prompt,
            "images": _data_url_references(references),
            "size": size,
            "seed": seed,
            "output_format": "png",
        }
    }
    result = await asyncio.to_thread(_request, payload)
    state = str(result.get("status") or "").upper()
    if state == "FAILED":
        error = result.get("error") or result.get("message") or "RunPod marked the request as failed."
        raise RunPodQwenError("RUNPOD_JOB_FAILED", str(error))
    if state and state != "COMPLETED":
        raise RunPodQwenError(
            "RUNPOD_INVALID_RESPONSE",
            f"RunPod returned unexpected request status {state}.",
        )
    output = result.get("output")
    if not isinstance(output, dict):
        raise RunPodQwenError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod completed without an output object.",
        )
    image_url, image_bytes = _extract_image_output(output)
    if isinstance(image_url, str) and image_url:
        image_bytes = await asyncio.to_thread(_download_result, image_url)
    if not image_bytes:
        raise RunPodQwenError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod completed without an output image.",
        )
    request_id = result.get("id")
    if not isinstance(request_id, str) or not request_id:
        request_id = "runpod-qwen"
    cost = output.get("cost")
    if not isinstance(cost, (int, float)):
        cost = settings.QWEN_RUNPOD_PRICE_USD_PER_IMAGE
    return RunPodGeneratedFrame(
        image_bytes=image_bytes,
        request_id=request_id,
        result_url=image_url if isinstance(image_url, str) else "",
        cost=float(cost),
    )