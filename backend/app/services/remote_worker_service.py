"""Authenticated bridge to the optional local Colab GPU worker.

The worker is treated as an untrusted, ephemeral accelerator. Reachability,
model presence, and successful generation never imply verification.
"""

from __future__ import annotations

import asyncio
import base64
import json
import urllib.error
import urllib.request
from typing import Any

from app.config import settings


class RemoteWorkerError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _enabled() -> bool:
    return bool(settings.WORKER_URL and settings.WORKER_TOKEN)


def _request(path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _enabled():
        raise RemoteWorkerError("WORKER_UNAVAILABLE", "Colab worker URL and token are not configured.")
    url = settings.WORKER_URL.rstrip("/") + "/" + path.lstrip("/")
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {settings.WORKER_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.WORKER_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RemoteWorkerError("WORKER_AUTH_FAILED", "Colab worker authentication failed.") from exc
        if exc.code == 404:
            raise RemoteWorkerError(
                "WORKER_ENDPOINT_NOT_FOUND",
                "The configured Colab URL is reachable, but the Atelier worker routes are missing. "
                "Restart the current Colab worker and update COLAB_WORKER_URL to its current network URL.",
            ) from exc
        raise RemoteWorkerError("WORKER_REQUEST_FAILED", f"Colab worker returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RemoteWorkerError("WORKER_UNAVAILABLE", "Colab worker could not be reached.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RemoteWorkerError("WORKER_INVALID_RESPONSE", "Colab worker returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise RemoteWorkerError("WORKER_INVALID_RESPONSE", "Colab worker response was not an object.")
    return result


def status() -> dict[str, Any]:
    if not _enabled():
        return {
            "id": "colab-worker",
            "name": "Authenticated Colab GPU worker",
            "configured": False,
            "ready": False,
            "runtime_ready": False,
            "verification_state": "unavailable",
            "reason": "WORKER_UNAVAILABLE: Colab worker URL and token are not configured.",
        }
    try:
        health = _request("/health")
        if health.get("worker") != "atelier-colab" or health.get("service") != "atelier-colab-worker":
            return {
                "id": "colab-worker",
                "name": "Incompatible external worker",
                "configured": True,
                "ready": False,
                "runtime_ready": False,
                "verification_state": "unavailable",
                "reason": "INCOMPATIBLE_WORKER: endpoint is not the shipped atelier-colab worker.",
            }
        verified = health.get("verification_state") == "verified" and health.get("human_product_verified") is True
        gpu_payload = health.get("gpu")
        gpu_available = (
            bool(gpu_payload.get("cuda_available"))
            if isinstance(gpu_payload, dict)
            else bool(gpu_payload)
        )
        raw_models = health.get("models")
        provider_name = str(health.get("provider") or "").lower()
        provider_status = health.get("provider_status")
        generation_models = [
            model for model in (raw_models if isinstance(raw_models, list) else [])
            if not (
                isinstance(model, str)
                and model.lower() in {"birefnet", "background-removal"}
            )
            and not (
                isinstance(model, dict)
                and str(model.get("id") or "").lower() in {"birefnet", "background-removal"}
            )
        ]
        if not provider_name and generation_models:
            first_model = generation_models[0]
            provider_name = (
                str(first_model.get("id") or "").lower()
                if isinstance(first_model, dict)
                else str(first_model).lower()
            )
        generation_ready = bool(health.get("generation_ready") is True and generation_models)
        provider_configured = bool(
            isinstance(provider_status, dict)
            and provider_status.get("provider") is not None
            and provider_status.get("error_code") != "NO_PROVIDER_CONFIGURED"
        ) or bool(provider_name and generation_models)
        provider_loaded = bool(
            isinstance(provider_status, dict)
            and provider_status.get("model_loaded") is True
        ) or generation_ready
        inference_passed = bool(
            health.get("inference_passed") is True
            or generation_ready
            or (
                isinstance(provider_status, dict)
                and provider_status.get("inference_passed") is True
            )
        )
        if isinstance(raw_models, list):
            provider_loaded = provider_loaded or any(
                (
                    isinstance(model, dict)
                    and str(model.get("id") or "").lower() == provider_name
                    and model.get("loaded") is True
                    and inference_passed
                )
                or (
                    isinstance(model, str)
                    and model.lower() == provider_name
                    and model.lower() != "birefnet"
                )
                for model in raw_models
            )
        return {
            "id": "colab-worker",
            "name": health.get("provider") or "Authenticated Colab GPU worker",
            "configured": True,
            "ready": gpu_available and provider_configured and provider_loaded and inference_passed and verified,
            "runtime_ready": bool(health.get("runtime_ready") or generation_ready),
            "verification_state": "verified" if verified else health.get("verification_state", "unverified"),
            "human_product_verified": bool(health.get("human_product_verified")),
            "gpu": health.get("gpu_info") or health.get("gpu"),
            "models": raw_models,
            "model_loaded": provider_loaded,
            "inference_available": inference_passed,
            "inference_passed": inference_passed,
            "generation_ready": generation_ready,
            "provider_configured": provider_configured,
            "worker_state": health.get("worker_state") or (
                "verified" if verified else "inference-ready-unverified" if inference_passed else "reachable-empty"
            ),
            "next_action": health.get("next_action") or (
                "Ready for verified human-with-product campaign jobs." if verified
                else "POST real authenticated product references to /verify after every Colab restart."
            ),
            "provider": provider_name or None,
            "gpu_available": gpu_available,
            "provider_status": provider_status,
            "verification": health.get("verification"),
            "reason": (
                None
                if verified
                else health.get("reason")
                or (
                    provider_status.get("reason")
                    if isinstance(provider_status, dict) and not inference_passed
                    else (
                        "Worker has no fresh human/product verification."
                        if inference_passed
                        else "Worker has no loaded generation provider."
                    )
                )
            ),
        }
    except RemoteWorkerError as exc:
        return {
            "id": "colab-worker",
            "name": "Authenticated Colab GPU worker",
            "configured": True,
            "ready": False,
            "runtime_ready": False,
            "verification_state": "unavailable",
            "reason": f"{exc.code}: {exc}",
        }


def _encode_references(references: list[tuple[bytes, str, str]]) -> list[dict[str, str]]:
    return [
        {
            "filename": filename,
            "content_type": content_type,
            "data_base64": base64.b64encode(content).decode("ascii"),
        }
        for content, filename, content_type in references
    ]


def _data_url_references(references: list[tuple[bytes, str, str]]) -> list[str]:
    return [
        f"data:{content_type or 'image/png'};base64,{base64.b64encode(content).decode('ascii')}"
        for content, _filename, content_type in references
    ]


def _legacy_contract() -> bool:
    """Detect the older deployed worker contract without trusting it as verified."""
    try:
        document = _request("/openapi.json")
        schema = (
            document.get("components", {})
            .get("schemas", {})
            .get("VerifyRequest", {})
            .get("properties", {})
        )
        return "source_images" in schema and "references" not in schema
    except RemoteWorkerError:
        return False


async def generate_frame(
    *,
    references: list[tuple[bytes, str, str]],
    identity_profile: dict[str, Any],
    prompt: str,
    shot_kind: str,
    seed: int,
    model: str = "auto",
) -> bytes:
    if _legacy_contract():
        payload = {
            "product_id": str(identity_profile.get("product_id") or "uploaded-product"),
            "source_images": _data_url_references(references),
            "category": str(identity_profile.get("category") or "handbag"),
            "frames": 1,
            "model": model,
            "quality": "high",
            "transparent": False,
            "prompt": prompt,
            "seed": seed,
        }
    else:
        payload = {
            "references": _encode_references(references),
            "identity_profile": identity_profile,
            "prompt": prompt,
            "shot_kind": shot_kind,
            "seed": seed,
            "model": model,
            "upscale": True,
        }
    submitted = await asyncio.to_thread(_request, "/generate", "POST", payload)
    job_id = submitted.get("job_id")
    if not job_id:
        raise RemoteWorkerError(str(submitted.get("error_code") or "WORKER_INVALID_RESPONSE"), "Worker did not return a job id.")
    deadline = asyncio.get_running_loop().time() + settings.WORKER_JOB_TIMEOUT_SECONDS
    while asyncio.get_running_loop().time() < deadline:
        job = await asyncio.to_thread(_request, f"/job/{job_id}")
        state = job.get("state") or job.get("status")
        if state == "completed":
            outputs = job.get("outputs") or job.get("images") or []
            if not outputs:
                raise RemoteWorkerError("WORKER_INVALID_RESPONSE", "Completed worker job contained no image.")
            try:
                first = outputs[0]
                encoded = first.get("data_base64") or first.get("b64_json") if isinstance(first, dict) else first
                if isinstance(encoded, str) and encoded.startswith("data:"):
                    encoded = encoded.split(",", 1)[1]
                if not encoded:
                    raise ValueError("missing image payload")
                return base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise RemoteWorkerError("WORKER_INVALID_RESPONSE", "Worker returned invalid image data.") from exc
        if state == "failed":
            raise RemoteWorkerError(str(job.get("error_code") or "WORKER_JOB_FAILED"), str(job.get("error") or "Worker job failed."))
        await asyncio.sleep(settings.WORKER_POLL_SECONDS)
    await asyncio.to_thread(_request, f"/cancel/{job_id}", "POST")
    raise RemoteWorkerError("WORKER_TIMEOUT", "Colab worker job timed out.")


async def verify(
    *,
    references: list[tuple[bytes, str, str]],
    identity_profile: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    if _legacy_contract():
        payload = {
            "product_id": str(identity_profile.get("product_id") or "uploaded-product"),
            "source_images": _data_url_references(references),
            "category": str(identity_profile.get("category") or "handbag"),
            "model": "auto",
        }
    else:
        payload = {
            "references": _encode_references(references),
            "identity_profile": identity_profile,
            "prompt": prompt,
        }
    return await asyncio.to_thread(_request, "/verify", "POST", payload)