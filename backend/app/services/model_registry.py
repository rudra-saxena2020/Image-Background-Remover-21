"""Capability-aware model readiness and audit status.

This module only normalizes facts reported by a local provider or the Colab
worker. It never promotes an engine to verified based on configuration alone.
"""

from __future__ import annotations

from typing import Any


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "qwen": {
        "name": "Qwen Image Edit 2511",
        "capabilities": ["image_edit", "reference_edit", "human_generation", "reference_guidance"],
        "requires_human_product_test": True,
    },
    "flux-schnell": {
        "name": "FLUX.1 Schnell",
        "capabilities": ["text_to_image", "image_edit", "human_generation", "reference_guidance"],
        "requires_human_product_test": True,
    },
    "fooocus": {
        "name": "Fooocus",
        "capabilities": ["text_to_image", "image_edit", "reference_guidance"],
        "requires_human_product_test": False,
    },
    "hidream": {
        "name": "HiDream-I1 Image",
        "capabilities": ["text_to_image", "image_edit", "human_generation", "reference_guidance"],
        "requires_human_product_test": True,
    },
    "flux2": {
        "name": "FLUX.2 Dev",
        "capabilities": ["text_to_image", "image_edit", "human_generation", "reference_guidance"],
        "requires_human_product_test": True,
    },
    "flux2-klein": {
        "name": "FLUX.2 Klein 4B",
        "capabilities": ["text_to_image", "image_edit", "human_generation", "reference_guidance"],
        "requires_human_product_test": True,
    },
    "sdxl": {
        "name": "Stable Diffusion XL",
        "capabilities": ["text_to_image", "image_edit", "human_generation", "reference_guidance"],
        "requires_human_product_test": True,
    },
    "colab": {
        "name": "Authenticated Colab GPU worker",
        "capabilities": ["text_to_image", "image_edit", "reference_edit", "human_generation", "reference_guidance"],
        "requires_human_product_test": True,
    },
    "reference-preview": {
        "name": "Reference-locked preview",
        "capabilities": ["source_preserved_preview", "product_composite"],
        "requires_human_product_test": False,
    },
}


def _verification_values(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = raw.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    validations = verification.get("validations")
    if not isinstance(validations, dict):
        validations = {}
    return verification, validations


def _is_remote_loaded(raw: dict[str, Any]) -> bool:
    models = raw.get("models")
    return bool(
        raw.get("model_loaded") is True
        and raw.get("inference_passed") is True
    )


def _status_for(
    *,
    installed: bool,
    runtime_reachable: bool,
    model_loaded: bool,
    inference_passed: bool,
    product_validation_passed: bool,
    human_model_passed: bool,
    verified: bool,
    raw: dict[str, Any],
    verification: dict[str, Any],
) -> str:
    if verified:
        return "verified"
    state = str(raw.get("verification_state") or "")
    reason = str(raw.get("reason") or "").lower()
    if "larger gpu" in reason or "insufficient_vram" in reason or "nvfp4" in reason:
        return "requires_larger_gpu"
    if state == "failed" or verification.get("status") == "failed":
        return "failed"
    if not installed or not runtime_reachable or not model_loaded:
        return "unavailable"
    if inference_passed or human_model_passed or product_validation_passed:
        return "online"
    return "unverified"


def _next_action(status: str, raw: dict[str, Any]) -> str:
    if status == "verified":
        return "Ready for the capabilities listed above."
    if status == "requires_larger_gpu":
        return "Move this engine to a compatible, larger GPU worker with enough VRAM."
    if status == "failed":
        return "Review the last audit error, fix the runtime, and rerun the audit."
    if status == "online":
        return "Run the capability-specific product audit before using it for campaigns."
    if not raw.get("model_present", True) or not raw.get("model_loaded", True):
        return "Install or load the model weights on the configured GPU worker."
    return "Run a fresh capability-specific audit."


def decorate(engine: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Return a stable diagnostic record while preserving provider details."""
    definition = MODEL_REGISTRY.get(engine, {
        "name": raw.get("name") or engine,
        "capabilities": [],
        "requires_human_product_test": True,
    })
    verification, validations = _verification_values(raw)
    remote = engine == "colab"
    installed = bool(
        raw.get("installed") is True
        or raw.get("model_present") is True
        or raw.get("repository_present") is True and raw.get("configured") is True
        or remote and raw.get("configured") is True
    )
    runtime_reachable = bool(
        raw.get("runtime_reachable") is True
        or raw.get("runtime_ready") is True
        or remote and raw.get("gpu") is not None
    )
    model_loaded = bool(
        raw.get("model_loaded") is True
        or raw.get("model_present") is True and raw.get("runtime_ready") is True
        or remote and _is_remote_loaded(raw)
    )
    inference_passed = bool(
        raw.get("inference_passed") is True
        or verification.get("status") == "passed"
        or validations.get("inference_passed") is True
    )
    product_validation_passed = bool(
        raw.get("product_validation_passed") is True
        or validations.get("product_identity") is True
        or float(validations.get("identity_score", 0) or 0) >= 0.9
    )
    human_model_passed = bool(
        raw.get("human_model_passed") is True
        or validations.get("human_present") is True
        or validations.get("person_detected") is True
    )
    verified = bool(raw.get("ready") is True and raw.get("verification_state") == "verified")
    status = _status_for(
        installed=installed,
        runtime_reachable=runtime_reachable,
        model_loaded=model_loaded,
        inference_passed=inference_passed,
        product_validation_passed=product_validation_passed,
        human_model_passed=human_model_passed,
        verified=verified,
        raw=raw,
        verification=verification,
    )
    checked_at = verification.get("checked_at") or raw.get("verified_at")
    return {
        **raw,
        "registry_id": engine,
        "capabilities": list(definition["capabilities"]),
        "requires_human_product_test": bool(definition["requires_human_product_test"]),
        "installed": installed,
        "runtime_reachable": runtime_reachable,
        "model_loaded": model_loaded,
        "inference_passed": inference_passed,
        "product_validation_passed": product_validation_passed,
        "human_model_passed": human_model_passed,
        "verified": verified,
        "last_test_time": checked_at,
        "last_error": None if status not in {"failed", "unavailable", "requires_larger_gpu"} else raw.get("reason"),
        "registry_status": status,
        "next_action": _next_action(status, raw),
    }


def reference_preview_status() -> dict[str, Any]:
    return decorate("reference-preview", {
        "id": "reference-preview",
        "name": "Reference-locked preview",
        "mode": "local-composite",
        "configured": True,
        "ready": True,
        "runtime_ready": True,
        "verification_state": "verified",
        "reason": "Source-preserved preview; not AI human generation.",
    })