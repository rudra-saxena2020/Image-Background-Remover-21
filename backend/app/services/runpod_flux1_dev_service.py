"""RunPod public endpoint bridge for Black Forest Labs FLUX.1 Dev."""

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


DEFAULT_ENDPOINT = (
    "https://api.runpod.ai/v2/black-forest-labs-flux-1-dev/runsync"
)
PRICING_SOURCE = "https://docs.runpod.io/public-endpoints/models/flux-dev"


class RunPodFlux1DevError(RuntimeError):
    """A safe, provider-specific error that can be shown in shoot status."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    @property
    def retryable(self) -> bool:
        return self.code not in {
            "RUNPOD_AUTH_FAILED",
            "RUNPOD_BALANCE_EXHAUSTED",
            "RUNPOD_UNAVAILABLE",
            "RUNPOD_INVALID_INPUT",
        }


@dataclass(frozen=True)
class RunPodFlux1DevFrame:
    image_bytes: bytes
    request_id: str
    result_url: str
    cost: float


def _endpoint() -> str:
    return (
        settings.RUNPOD_FLUX1_DEV_ENDPOINT.rstrip("/")
        or DEFAULT_ENDPOINT
    )


def _configured() -> bool:
    return settings.RUNPOD_FLUX1_DEV_ENABLED and bool(settings.RUNPOD_API_KEY)


def _dimensions() -> tuple[int, int]:
    return settings.RUNPOD_FLUX1_DEV_WIDTH, settings.RUNPOD_FLUX1_DEV_HEIGHT


def estimate_cost() -> float:
    width, height = _dimensions()
    return round(
        (width * height / 1_000_000)
        * settings.RUNPOD_FLUX1_DEV_PRICE_USD_PER_MEGAPIXEL,
        8,
    )


def status() -> dict[str, object]:
    configured = _configured()
    width, height = _dimensions()
    rate = settings.RUNPOD_FLUX1_DEV_PRICE_USD_PER_MEGAPIXEL
    return {
        "id": "flux1-runpod",
        "name": "FLUX.1 Dev · RunPod",
        "mode": "hosted-paid",
        "provider": "RunPod",
        "model": "black-forest-labs-flux-1-dev",
        "endpoint": _endpoint(),
        "configured": configured,
        "runtime_ready": configured,
        "ready": configured,
        "supports_source_images": False,
        "supports_multi_reference": False,
        "human_product_verified": False,
        "verification_state": "unverified",
        "provider_billing": (
            f"${rate:.2f} per megapixel · approximately "
            f"${estimate_cost():.3f} per {width}×{height} image"
        ),
        "pricing": {
            "unit": "megapixel",
            "rate_usd": rate,
            "width": width,
            "height": height,
            "megapixels": round(width * height / 1_000_000, 6),
            "estimated_cost_usd": estimate_cost(),
            "source": PRICING_SOURCE,
        },
        "reason": (
            "Available through the configured RunPod API key. FLUX.1 Dev is a "
            "text-to-image endpoint; references remain local context and are not "
            "uploaded to this provider."
            if configured
            else "RunPod FLUX.1 Dev requires the RUNPOD_API_KEY secret."
        ),
    }


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    if not _configured():
        raise RunPodFlux1DevError("RUNPOD_UNAVAILABLE", str(status()["reason"]))
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
            request, timeout=settings.RUNPOD_FLUX1_DEV_REQUEST_TIMEOUT_SECONDS
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").lower()
        if "credit" in detail or "balance" in detail:
            raise RunPodFlux1DevError(
                "RUNPOD_BALANCE_EXHAUSTED",
                "RunPod has no remaining credits for FLUX.1 Dev.",
            ) from exc
        if exc.code in (401, 403):
            raise RunPodFlux1DevError(
                "RUNPOD_AUTH_FAILED",
                "RunPod rejected the configured API key.",
            ) from exc
        raise RunPodFlux1DevError(
            "RUNPOD_REQUEST_FAILED",
            f"RunPod returned HTTP {exc.code}.",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RunPodFlux1DevError(
            "RUNPOD_UNAVAILABLE",
            "The RunPod FLUX.1 Dev endpoint could not be reached.",
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned invalid JSON.",
        ) from exc
    if not isinstance(result, dict):
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned an invalid response object.",
        )
    return result


def _download_result(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    allowed_host = hostname == "image.runpod.ai" or hostname.endswith(".runpod.ai")
    if parsed.scheme != "https" or not allowed_host:
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned an unsafe output image URL.",
        )
    request = urllib.request.Request(url, headers={"Accept": "image/*"})
    try:
        with urllib.request.urlopen(
            request, timeout=settings.RUNPOD_FLUX1_DEV_REQUEST_TIMEOUT_SECONDS
        ) as response:
            chunks: list[bytes] = []
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > settings.RUNPOD_FLUX1_DEV_MAX_RESULT_BYTES:
                    raise RunPodFlux1DevError(
                        "RUNPOD_RESULT_TOO_LARGE",
                        "RunPod returned an output larger than Atelier's safety limit.",
                    )
                chunks.append(chunk)
    except RunPodFlux1DevError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RunPodFlux1DevError(
            "RUNPOD_RESULT_DOWNLOAD_FAILED",
            "The RunPod FLUX.1 Dev output image could not be downloaded.",
        ) from exc
    output = b"".join(chunks)
    if not output:
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod returned an empty output image.",
        )
    return output


def _decode_inline_image(value: object) -> bytes | None:
    if not isinstance(value, str):
        return None
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    return decoded or None


def _extract_image_output(value: object) -> tuple[str | None, bytes | None]:
    """Handle the output.image and output.image_url shapes used by RunPod workers."""
    if isinstance(value, str):
        if value.startswith("https://"):
            return value, None
        return None, _decode_inline_image(value)
    if isinstance(value, list):
        for item in value:
            image_url, image_bytes = _extract_image_output(item)
            if image_url or image_bytes:
                return image_url, image_bytes
        return None, None
    if isinstance(value, dict):
        for key in ("image_url", "image", "url", "images", "output"):
            image_url, image_bytes = _extract_image_output(value.get(key))
            if image_url or image_bytes:
                return image_url, image_bytes
    return None, None


async def generate_frame(
    *,
    prompt: str,
    seed: int,
) -> RunPodFlux1DevFrame:
    width, height = _dimensions()
    if (
        width < 256
        or width > 1536
        or height < 256
        or height > 1536
        or width % 64
        or height % 64
    ):
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_INPUT",
            "RunPod FLUX.1 Dev width and height must be divisible by 64 and between 256 and 1536.",
        )
    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": (
                "blurry, low quality, distorted product, extra products, "
                "text, watermark, deformed anatomy"
            ),
            "width": width,
            "height": height,
            "num_inference_steps": settings.RUNPOD_FLUX1_DEV_STEPS,
            "guidance": settings.RUNPOD_FLUX1_DEV_GUIDANCE,
            "seed": seed,
            "image_format": "png",
        }
    }
    result = await asyncio.to_thread(_request, payload)
    state = str(result.get("status") or "").upper()
    if state == "FAILED":
        error = result.get("error") or result.get("message") or "RunPod marked the request as failed."
        raise RunPodFlux1DevError("RUNPOD_JOB_FAILED", str(error))
    if state and state != "COMPLETED":
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_RESPONSE",
            f"RunPod returned unexpected request status {state}.",
        )
    output = result.get("output")
    if not isinstance(output, dict):
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod completed without an output object.",
        )
    image_url, image_bytes = _extract_image_output(output)
    if image_url:
        image_bytes = await asyncio.to_thread(_download_result, image_url)
    if not image_bytes:
        raise RunPodFlux1DevError(
            "RUNPOD_INVALID_RESPONSE",
            "RunPod completed without an output image.",
        )
    cost = output.get("cost")
    if not isinstance(cost, (int, float)):
        cost = estimate_cost()
    return RunPodFlux1DevFrame(
        image_bytes=image_bytes,
        request_id=str(result.get("id") or "runpod-flux1-dev"),
        result_url=image_url or "",
        cost=float(cost),
    )