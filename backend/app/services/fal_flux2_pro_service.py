"""Authenticated FLUX.2 Pro image-editing provider through the Replit fal.ai connector.

This module intentionally has no API-key support. Every request crosses the
configured connector bridge, and every provider output is returned to the
normal Atelier validation pipeline before it can become a ready frame.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings

MODEL_ENDPOINT = "fal-ai/flux-2-pro/edit"
QUEUE_BASE_URL = "https://queue.fal.run"
_TERMINAL_FAILURES = {"FAILED", "CANCELED", "CANCELLED"}
_ACTIVE_STATES = {"IN_QUEUE", "IN_PROGRESS", "QUEUED", "PROCESSING"}
_NON_RETRYABLE_CODES = {"FAL_BALANCE_EXHAUSTED", "FAL_AUTH_FAILED"}
_runtime_block_reason: str | None = None


class FalFlux2ProError(RuntimeError):
    """A safe, provider-specific error that can be shown in shoot status."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.code not in _NON_RETRYABLE_CODES


@dataclass(frozen=True)
class FalGeneratedFrame:
    image_bytes: bytes
    request_id: str
    seed: int | None
    result_url: str


def _bridge_path() -> Path:
    return Path(settings.FAL_FLUX2_PRO_BRIDGE).expanduser()


def _configured() -> bool:
    return (
        settings.FAL_FLUX2_PRO_ENABLED
        and _bridge_path().is_file()
        and _runtime_block_reason is None
    )


def status() -> dict[str, object]:
    """Return availability, never a fabricated inference verification state."""
    bridge_exists = _bridge_path().is_file()
    configured = _configured()
    return {
        "id": "flux2-pro",
        "name": "FLUX.2 Pro · fal.ai",
        "mode": "hosted-paid",
        "provider": "fal.ai",
        "model": MODEL_ENDPOINT,
        "configured": configured,
        "runtime_ready": configured,
        "ready": configured,
        "supports_source_images": True,
        "supports_multi_reference": True,
        "human_product_verified": False,
        "verification_state": "unverified",
        "provider_billing": "fal.ai usage-based API",
        "reason": (
            _runtime_block_reason
            if _runtime_block_reason
            else "Available through the connected fal.ai account. Each output is validated "
            "for product identity, human interaction, anatomy, and composition before delivery."
            if configured
            else (
                "FLUX.2 Pro is disabled. Enable FAL_FLUX2_PRO_ENABLED after attaching the fal.ai connector."
                if not settings.FAL_FLUX2_PRO_ENABLED
                else "The FLUX.2 Pro connector bridge is missing."
            )
        ),
    }


def _data_url_references(references: list[tuple[bytes, str, str]]) -> list[str]:
    if not references:
        raise FalFlux2ProError("FAL_INVALID_INPUT", "FLUX.2 Pro requires at least one product reference.")
    return [
        f"data:{content_type or 'image/png'};base64,{base64.b64encode(content).decode('ascii')}"
        for content, _filename, content_type in references
    ]


def _queue_url(path: str) -> str:
    return f"{QUEUE_BASE_URL}/{path.lstrip('/')}"


def _is_safe_queue_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme == "https" and parsed.netloc == "queue.fal.run"


def _request_queue(
    url: str, method: str = "GET", payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not _configured():
        raise FalFlux2ProError(
            "FAL_UNAVAILABLE",
            str(status()["reason"]),
        )
    if not _is_safe_queue_url(url):
        raise FalFlux2ProError("FAL_INVALID_RESPONSE", "fal.ai returned an invalid queue URL.")
    request_payload = json.dumps(
        {
            "connectorName": settings.FAL_FLUX2_PRO_CONNECTOR_NAME,
            "url": url,
            "method": method,
            "payload": payload,
        }
    )
    try:
        completed = subprocess.run(
            [settings.FAL_FLUX2_PRO_NODE, str(_bridge_path())],
            input=request_payload,
            capture_output=True,
            text=True,
            timeout=settings.FAL_FLUX2_PRO_REQUEST_TIMEOUT_SECONDS,
            check=False,
        )
        envelope = json.loads(completed.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise FalFlux2ProError(
            "FAL_CONNECTOR_UNAVAILABLE",
            "The authenticated fal.ai connector could not be reached.",
        ) from exc
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        raise FalFlux2ProError(
            "FAL_CONNECTOR_UNAVAILABLE",
            "The authenticated fal.ai connector request failed.",
        )
    response_status = envelope.get("status")
    try:
        response_status = int(response_status)
    except (TypeError, ValueError):
        raise FalFlux2ProError("FAL_INVALID_RESPONSE", "fal.ai returned no HTTP status.") from None
    raw_body = envelope.get("body")
    try:
        body = json.loads(raw_body) if isinstance(raw_body, str) and raw_body else {}
    except json.JSONDecodeError:
        body = {}
    provider_message = (
        body.get("detail") or body.get("message") or body.get("error")
        if isinstance(body, dict)
        else None
    )
    if response_status in (401, 403):
        detail = provider_message if isinstance(provider_message, str) else ""
        if "exhausted balance" in detail.lower() or "balance" in detail.lower() and "top up" in detail.lower():
            global _runtime_block_reason
            _runtime_block_reason = (
                "FLUX.2 Pro is unavailable because the connected fal.ai account has exhausted "
                "its balance. Top up the account at fal.ai/dashboard/billing, then try again."
            )
            raise FalFlux2ProError("FAL_BALANCE_EXHAUSTED", _runtime_block_reason)
        raise FalFlux2ProError(
            "FAL_AUTH_FAILED",
            "The connected fal.ai account rejected the FLUX.2 Pro request."
            + (f" {detail}" if detail else ""),
        )
    if not 200 <= response_status < 300:
        raise FalFlux2ProError(
            "FAL_REQUEST_FAILED",
            f"fal.ai returned HTTP {response_status}"
            + (f": {provider_message}" if isinstance(provider_message, str) else "."),
        )
    if not isinstance(body, dict):
        raise FalFlux2ProError("FAL_INVALID_RESPONSE", "fal.ai returned an invalid JSON response.")
    return body


def _response_url(submitted: dict[str, Any], request_id: str, suffix: str) -> str:
    supplied = submitted.get(f"{suffix}_url")
    if isinstance(supplied, str) and _is_safe_queue_url(supplied):
        return supplied
    return _queue_url(f"{MODEL_ENDPOINT}/requests/{request_id}/{suffix}")


def _download_result(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise FalFlux2ProError("FAL_INVALID_RESPONSE", "fal.ai returned an unsafe output URL.")
    request = urllib.request.Request(url, headers={"Accept": "image/*"})
    try:
        with urllib.request.urlopen(request, timeout=settings.FAL_FLUX2_PRO_REQUEST_TIMEOUT_SECONDS) as response:
            if response.status < 200 or response.status >= 300:
                raise FalFlux2ProError(
                    "FAL_RESULT_DOWNLOAD_FAILED",
                    f"fal.ai output download returned HTTP {response.status}.",
                )
            chunks: list[bytes] = []
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > settings.FAL_FLUX2_PRO_MAX_RESULT_BYTES:
                    raise FalFlux2ProError(
                        "FAL_RESULT_TOO_LARGE",
                        "fal.ai returned an output larger than Atelier's safety limit.",
                    )
                chunks.append(chunk)
            output = b"".join(chunks)
    except FalFlux2ProError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FalFlux2ProError(
            "FAL_RESULT_DOWNLOAD_FAILED",
            "fal.ai output could not be downloaded for validation.",
        ) from exc
    if not output:
        raise FalFlux2ProError("FAL_INVALID_RESPONSE", "fal.ai returned an empty output image.")
    return output


def _image_size_for(shot_kind: str) -> str:
    return "landscape_4_3" if shot_kind in {"editorial", "lifestyle", "hero"} else "square_hd"


async def generate_frame(
    *,
    references: list[tuple[bytes, str, str]],
    prompt: str,
    shot_kind: str,
    seed: int,
) -> FalGeneratedFrame:
    """Submit one multi-reference edit and return the provider image for validation."""
    payload = {
        "prompt": prompt,
        "image_urls": _data_url_references(references),
        "image_size": _image_size_for(shot_kind),
        "output_format": "png",
        "safety_tolerance": "2",
        "seed": seed,
    }
    submitted = await asyncio.to_thread(
        _request_queue, _queue_url(MODEL_ENDPOINT), "POST", payload
    )
    request_id = submitted.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise FalFlux2ProError("FAL_INVALID_RESPONSE", "fal.ai did not return a queue request id.")
    status_url = _response_url(submitted, request_id, "status")
    result_url = _response_url(submitted, request_id, "response")
    deadline = asyncio.get_running_loop().time() + settings.FAL_FLUX2_PRO_JOB_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        job = await asyncio.to_thread(_request_queue, status_url)
        state = str(job.get("status") or "").upper()
        if state == "COMPLETED":
            result_url = str(job.get("response_url") or result_url)
            break
        if state in _TERMINAL_FAILURES:
            reason = job.get("error") or job.get("detail") or "fal.ai marked the request as failed."
            raise FalFlux2ProError("FAL_JOB_FAILED", str(reason))
        if state and state not in _ACTIVE_STATES:
            raise FalFlux2ProError("FAL_INVALID_RESPONSE", f"fal.ai returned unknown queue status {state}.")
        await asyncio.sleep(settings.FAL_FLUX2_PRO_POLL_SECONDS)
    else:
        raise FalFlux2ProError(
            "FAL_TIMEOUT",
            "FLUX.2 Pro did not finish before Atelier's generation timeout.",
        )
    result = await asyncio.to_thread(_request_queue, result_url)
    images = result.get("images")
    first_image = images[0] if isinstance(images, list) and images else None
    image_url = first_image.get("url") if isinstance(first_image, dict) else None
    if not isinstance(image_url, str) or not image_url:
        raise FalFlux2ProError("FAL_INVALID_RESPONSE", "fal.ai completed without an output image.")
    image_bytes = await asyncio.to_thread(_download_result, image_url)
    result_seed = result.get("seed")
    return FalGeneratedFrame(
        image_bytes=image_bytes,
        request_id=request_id,
        seed=result_seed if isinstance(result_seed, int) else None,
        result_url=image_url,
    )