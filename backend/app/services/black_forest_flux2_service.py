"""Server-routed Black Forest Labs FLUX.2 image-editing provider.

This service never exposes provider keys, BFL request URLs, or temporary
reference URLs to a browser. Product references arrive as multipart bytes at
Atelier, are forwarded as binary data URIs over the provider TLS connection,
and the resulting image is downloaded server-side before validation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import settings

MODEL_ENDPOINT = "flux-2-pro"
_TERMINAL_FAILURES = {"FAILED", "ERROR", "CANCELLED", "CANCELED", "MODERATED"}
_ACTIVE_STATES = {"PENDING", "QUEUED", "PROCESSING", "IN_PROGRESS", "RUNNING"}
_NON_RETRYABLE_CODES = {
    "BFL_AUTH_FAILED",
    "BFL_BALANCE_EXHAUSTED",
    "BFL_INPUT_MODERATED",
    "BFL_UNAVAILABLE",
}
_runtime_block_reason: str | None = None


class BlackForestFlux2Error(RuntimeError):
    """A safe provider error which may be surfaced in a shoot state."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.code not in _NON_RETRYABLE_CODES


@dataclass(frozen=True)
class BlackForestGeneratedFrame:
    image_bytes: bytes
    request_id: str
    seed: int | None
    result_url: str


def _configured() -> bool:
    return (
        settings.BFL_FLUX2_ENABLED
        and bool(settings.BFL_API_KEY)
        and bool(settings.BFL_API_BASE_URL)
        and bool(settings.BFL_FLUX2_MODEL_ENDPOINT)
        and _runtime_block_reason is None
    )


def status() -> dict[str, object]:
    """Report truthful configuration state without consuming provider credits."""
    configured = _configured()
    if _runtime_block_reason:
        reason = _runtime_block_reason
    elif not settings.BFL_FLUX2_ENABLED:
        reason = "Black Forest FLUX.2 is disabled. Enable BFL_FLUX2_ENABLED after adding the server-side BFL_API_KEY."
    elif not settings.BFL_API_KEY:
        reason = "Black Forest FLUX.2 needs a server-side BFL_API_KEY. The chat MCP connection cannot be used by the app."
    else:
        reason = (
            "Available through the server-side Black Forest Labs API. References and output "
            "remain server-routed and every frame is validated before delivery."
        )
    return {
        "id": "bfl-flux2",
        "name": "FLUX.2 Pro · Black Forest",
        "mode": "hosted-paid",
        "provider": "Black Forest Labs",
        "model": settings.BFL_FLUX2_MODEL_ENDPOINT or MODEL_ENDPOINT,
        "configured": configured,
        "runtime_ready": configured,
        "ready": configured,
        "supports_source_images": True,
        "supports_multi_reference": True,
        "human_product_verified": False,
        "verification_state": "unverified",
        "provider_billing": "Black Forest Labs usage-based API",
        "reason": reason,
    }


def _data_uri_references(references: list[tuple[bytes, str, str]]) -> list[str]:
    if not references:
        raise BlackForestFlux2Error(
            "BFL_INVALID_INPUT", "Black Forest FLUX.2 requires at least one product reference."
        )
    return [
        f"data:{content_type or 'image/png'};base64,{base64.b64encode(content).decode('ascii')}"
        for content, _filename, content_type in references
    ]


def _endpoint(path: str) -> str:
    return f"{settings.BFL_API_BASE_URL}/{path.lstrip('/')}"


def _is_safe_bfl_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    expected = urllib.parse.urlparse(settings.BFL_API_BASE_URL)
    return (
        parsed.scheme == "https"
        and parsed.hostname == expected.hostname
        and parsed.port == expected.port
    )


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call only BFL's configured HTTPS API and normalize provider failures."""
    if not _configured():
        raise BlackForestFlux2Error("BFL_UNAVAILABLE", str(status()["reason"]))
    if not _is_safe_bfl_url(url):
        raise BlackForestFlux2Error("BFL_INVALID_RESPONSE", "Black Forest returned an invalid API URL.")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "x-key": settings.BFL_API_KEY,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.BFL_FLUX2_REQUEST_TIMEOUT_SECONDS
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        lowered = detail.lower()
        global _runtime_block_reason
        if "credit" in lowered or "balance" in lowered:
            _runtime_block_reason = "Black Forest FLUX.2 is unavailable because the provider account has no remaining credits."
            raise BlackForestFlux2Error("BFL_BALANCE_EXHAUSTED", _runtime_block_reason) from exc
        if exc.code in {401, 403}:
            _runtime_block_reason = (
                "Black Forest FLUX.2 rejected the server connection. Check the BFL_API_KEY "
                "and provider account permissions."
            )
            raise BlackForestFlux2Error("BFL_AUTH_FAILED", _runtime_block_reason) from exc
        if "moderat" in lowered or "safety" in lowered:
            raise BlackForestFlux2Error("BFL_INPUT_MODERATED", "Black Forest rejected the requested input for safety.")
        raise BlackForestFlux2Error(
            "BFL_REQUEST_FAILED", f"Black Forest returned HTTP {exc.code}."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BlackForestFlux2Error(
            "BFL_CONNECTOR_UNAVAILABLE", "Black Forest could not be reached from the server."
        ) from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlackForestFlux2Error(
            "BFL_INVALID_RESPONSE", "Black Forest returned invalid JSON."
        ) from exc
    if not isinstance(decoded, dict):
        raise BlackForestFlux2Error(
            "BFL_INVALID_RESPONSE", "Black Forest returned an invalid response."
        )
    return decoded


def _download_result(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise BlackForestFlux2Error("BFL_INVALID_RESPONSE", "Black Forest returned an unsafe output URL.")
    request = urllib.request.Request(url, headers={"Accept": "image/*"})
    try:
        with urllib.request.urlopen(
            request, timeout=settings.BFL_FLUX2_REQUEST_TIMEOUT_SECONDS
        ) as response:
            chunks: list[bytes] = []
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > settings.BFL_FLUX2_MAX_RESULT_BYTES:
                    raise BlackForestFlux2Error(
                        "BFL_RESULT_TOO_LARGE",
                        "Black Forest returned an output larger than Atelier's safety limit.",
                    )
                chunks.append(chunk)
    except BlackForestFlux2Error:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BlackForestFlux2Error(
            "BFL_RESULT_DOWNLOAD_FAILED",
            "Black Forest output could not be downloaded for validation.",
        ) from exc
    output = b"".join(chunks)
    if not output:
        raise BlackForestFlux2Error("BFL_INVALID_RESPONSE", "Black Forest returned an empty output image.")
    return output


def _image_size_for(shot_kind: str) -> tuple[int, int]:
    return (1536, 1152) if shot_kind in {"editorial", "lifestyle", "hero"} else (1440, 1440)


def _job_state(job: dict[str, Any]) -> str:
    return str(job.get("status") or job.get("state") or "").upper()


def _result_url(job: dict[str, Any]) -> str | None:
    result = job.get("result")
    if isinstance(result, dict):
        for key in ("sample", "url", "image_url"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("sample", "image_url", "result_url"):
        value = job.get(key)
        if isinstance(value, str) and value:
            return value
    return None


async def generate_frame(
    *,
    references: list[tuple[bytes, str, str]],
    prompt: str,
    shot_kind: str,
    seed: int,
) -> BlackForestGeneratedFrame:
    """Create, poll, then server-download one provider frame for Atelier validation."""
    width, height = _image_size_for(shot_kind)
    payload = {
        "prompt": prompt,
        "input_images": _data_uri_references(references),
        "width": width,
        "height": height,
        "seed": seed,
        "output_format": "png",
        "safety_tolerance": 2,
    }
    submitted = await asyncio.to_thread(
        _request_json,
        _endpoint(settings.BFL_FLUX2_MODEL_ENDPOINT),
        method="POST",
        payload=payload,
    )
    request_id = submitted.get("id") or submitted.get("request_id")
    polling_url = submitted.get("polling_url") or submitted.get("status_url")
    if not isinstance(request_id, str) or not request_id:
        raise BlackForestFlux2Error("BFL_INVALID_RESPONSE", "Black Forest did not return a request id.")
    if not isinstance(polling_url, str) or not _is_safe_bfl_url(polling_url):
        polling_url = _endpoint(f"requests/{request_id}")
    deadline = asyncio.get_running_loop().time() + settings.BFL_FLUX2_JOB_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        job = await asyncio.to_thread(_request_json, polling_url)
        state = _job_state(job)
        if state in {"READY", "COMPLETED", "SUCCESS"}:
            image_url = _result_url(job)
            if not image_url:
                raise BlackForestFlux2Error(
                    "BFL_INVALID_RESPONSE", "Black Forest completed without an output image."
                )
            image_bytes = await asyncio.to_thread(_download_result, image_url)
            result = job.get("result")
            result_seed = result.get("seed") if isinstance(result, dict) else job.get("seed")
            return BlackForestGeneratedFrame(
                image_bytes=image_bytes,
                request_id=request_id,
                seed=result_seed if isinstance(result_seed, int) else None,
                result_url=image_url,
            )
        if state in _TERMINAL_FAILURES:
            reason = job.get("error") or job.get("detail") or "Black Forest marked the request as failed."
            raise BlackForestFlux2Error("BFL_JOB_FAILED", str(reason))
        if state and state not in _ACTIVE_STATES:
            raise BlackForestFlux2Error(
                "BFL_INVALID_RESPONSE", f"Black Forest returned unknown request status {state}."
            )
        await asyncio.sleep(settings.BFL_FLUX2_POLL_SECONDS)
    raise BlackForestFlux2Error(
        "BFL_TIMEOUT", "Black Forest FLUX.2 did not finish before Atelier's generation timeout."
    )


async def cancel_frame(request_id: str) -> None:
    """Best-effort provider cancellation; local cancellation always remains authoritative."""
    if not request_id or not _configured():
        return
    try:
        await asyncio.to_thread(
            _request_json,
            _endpoint(f"requests/{urllib.parse.quote(request_id, safe='')}/cancel"),
            method="POST",
            payload={},
        )
    except BlackForestFlux2Error:
        # BFL cancellation is advisory. Never turn an already-cancelled Atelier shoot
        # back into a failure just because the remote request finished simultaneously.
        return