"""Server-side Gemini native image generation bridge."""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import settings


class GeminiImageError(RuntimeError):
    """A safe provider-specific error for the shoot status."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.code not in {"GEMINI_UNAVAILABLE", "GEMINI_AUTH_FAILED", "GEMINI_INVALID_INPUT"}


@dataclass(frozen=True)
class GeminiImageFrame:
    image_bytes: bytes
    request_id: str


def _configured() -> bool:
    return bool(
        settings.GEMINI_IMAGE_ENABLED
        and settings.GEMINI_IMAGE_BASE_URL
        and settings.GEMINI_IMAGE_API_KEY
    )


def status() -> dict[str, object]:
    configured = _configured()
    return {
        "id": "gemini-image",
        "name": "Gemini image generation",
        "mode": "hosted-managed",
        "provider": "Google Gemini",
        "model": settings.GEMINI_IMAGE_MODEL,
        "configured": configured,
        "runtime_ready": configured,
        "ready": configured,
        "supports_source_images": True,
        "supports_multi_reference": True,
        "human_product_verified": False,
        "verification_state": "unverified",
        "reason": (
            "Available through the server-side Gemini image integration."
            if configured
            else "Gemini image generation requires the managed Gemini integration."
        ),
    }


def _endpoint(base_url: str | None = None) -> str:
    base = (base_url or settings.GEMINI_IMAGE_BASE_URL).rstrip("/")
    if base.endswith("/v1beta"):
        return f"{base}/models/{settings.GEMINI_IMAGE_MODEL}:generateContent"
    return f"{base}/v1beta/models/{settings.GEMINI_IMAGE_MODEL}:generateContent"


def _request(
    payload: dict[str, Any],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    if not _configured():
        raise GeminiImageError("GEMINI_UNAVAILABLE", str(status()["reason"]))
    request = urllib.request.Request(
        _endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "x-goog-api-key": api_key or settings.GEMINI_IMAGE_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.GEMINI_IMAGE_REQUEST_TIMEOUT_SECONDS
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").lower()
        if exc.code in (401, 403):
            raise GeminiImageError(
                "GEMINI_AUTH_FAILED", "Gemini rejected the configured integration."
            ) from exc
        if exc.code == 429 or "quota" in detail or "rate" in detail:
            raise GeminiImageError(
                "GEMINI_RATE_LIMITED",
                "Gemini image generation quota is exhausted or temporarily rate limited.",
            ) from exc
        raise GeminiImageError(
            "GEMINI_REQUEST_FAILED", f"Gemini returned HTTP {exc.code}."
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeminiImageError(
            "GEMINI_UNAVAILABLE", "The Gemini image endpoint could not be reached."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GeminiImageError(
            "GEMINI_INVALID_RESPONSE", "Gemini returned invalid JSON."
        ) from exc
    if not isinstance(result, dict):
        raise GeminiImageError("GEMINI_INVALID_RESPONSE", "Gemini returned an invalid response object.")
    return result


def _request_with_provider_fallback(payload: dict[str, Any]) -> dict[str, Any]:
    """Try the selected provider, then the managed proxy if a personal key is limited."""
    candidates = [
        (settings.GEMINI_IMAGE_BASE_URL, settings.GEMINI_IMAGE_API_KEY),
    ]
    managed = (
        getattr(settings, "_GEMINI_MANAGED_BASE_URL", ""),
        getattr(settings, "_GEMINI_MANAGED_KEY", ""),
    )
    if managed[0] and managed[1] and managed not in candidates:
        candidates.append(managed)

    last_error: GeminiImageError | None = None
    for base_url, api_key in candidates:
        try:
            return _request(payload, base_url=base_url, api_key=api_key)
        except GeminiImageError as exc:
            last_error = exc
            if exc.code != "GEMINI_RATE_LIMITED":
                raise
    raise last_error or GeminiImageError(
        "GEMINI_UNAVAILABLE", "No Gemini image provider is configured."
    )


def _inline_image(value: object) -> bytes | None:
    if not isinstance(value, str):
        return None
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None
    return decoded or None


def _extract_image(result: dict[str, Any]) -> bytes | None:
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict):
                image = _inline_image(inline.get("data"))
                if image:
                    return image
    return None


async def generate_frame(
    *,
    references: list[tuple[bytes, str, str]],
    prompt: str,
    seed: int,
) -> GeminiImageFrame:
    if not references:
        raise GeminiImageError("GEMINI_INVALID_INPUT", "Gemini requires at least one product reference.")
    if len(references) > 3:
        raise GeminiImageError("GEMINI_INVALID_INPUT", "Gemini accepts up to three product references.")
    parts: list[dict[str, Any]] = [{"text": f"{prompt}\nDeterministic variation seed: {seed}."}]
    for content, _filename, content_type in references:
        parts.append(
            {
                "inlineData": {
                    "mimeType": content_type or "image/png",
                    "data": base64.b64encode(content).decode("ascii"),
                }
            }
        )
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    result = await asyncio.to_thread(_request_with_provider_fallback, payload)
    image_bytes = _extract_image(result)
    if not image_bytes:
        raise GeminiImageError(
            "GEMINI_INVALID_RESPONSE", "Gemini completed without an output image."
        )
    if len(image_bytes) > settings.GEMINI_IMAGE_MAX_RESULT_BYTES:
        raise GeminiImageError(
            "GEMINI_RESULT_TOO_LARGE", "Gemini returned an output larger than Atelier's safety limit."
        )
    return GeminiImageFrame(
        image_bytes=image_bytes,
        request_id=str(result.get("responseId") or result.get("response_id") or "gemini-image"),
    )