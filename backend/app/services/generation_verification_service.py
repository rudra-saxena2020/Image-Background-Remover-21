"""Persistent, fail-closed verification state for local generation engines.

An engine is runtime-ready when its files, Python runtime and device are
available. It is generation-ready only after a current human-with-product smoke
test has passed against the same runtime fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.model_registry import decorate

REPORT_VERSION = 1
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_DYNAMIC_STATUS_FIELDS = {
    "active_processes",
    "ready",
    "reason",
    "verification",
    "verification_state",
    "human_product_verified",
    "runtime_ready",
    "worker_ready",
}
_runtime_probe_cache: dict[str, dict[str, object]] = {}


def _iso_timestamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content_hash(path: Path) -> str | None:
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _file_signature(path: Path, root: Path | None = None) -> dict[str, object]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    signature: dict[str, object] = {
        "path": str(path.relative_to(root)) if root else str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    content_hash = _content_hash(path)
    if content_hash:
        signature["sha256"] = content_hash
    return signature


def _path_signature(value: object) -> dict[str, object] | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    signature: dict[str, object] = {
        "path": str(path),
        "exists": True,
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.is_file():
        signature.update(_file_signature(path))
        return signature
    files: list[dict[str, object]] = []
    try:
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            files.append(_file_signature(child, path))
    except OSError:
        signature["scan_error"] = True
    signature["files"] = files
    return signature


def _engine_dependencies(engine: str, base_status: dict[str, object]) -> list[object]:
    scripts = _BACKEND_ROOT / "scripts"
    dependencies: dict[str, list[object]] = {
        "qwen": [
            settings.QWEN_PYTHON,
            scripts / "qwen_image_edit_runner.py",
        ],
        "flux-schnell": [
            settings.FLUX_SCHNELL_PYTHON,
            scripts / "flux_schnell_runner.py",
        ],
        "fooocus": [
            settings.FOOOCUS_PYTHON,
            Path(settings.FOOOCUS_ROOT) / "entry_with_update.py",
        ],
        "hidream": [
            settings.HIDREAM_PYTHON,
            Path(settings.HIDREAM_REPO) / "inference.py",
        ],
        "flux2": [
            settings.FLUX2_PYTHON,
            settings.FLUX2_AE_MODEL_PATH,
            scripts / "flux2_runner.py",
            Path(settings.FLUX2_REPO) / "src" / "flux2" / "util.py",
        ],
        "flux2-klein": [settings.FLUX2_KLEIN_PYTHON],
        "sdxl": [
            settings.SDXL_PYTHON,
            scripts / "sdxl_runner.py",
        ],
    }
    return [base_status.get("model_path"), *dependencies.get(engine, [])]


def _engine_configuration(engine: str) -> dict[str, object]:
    names: dict[str, tuple[str, ...]] = {
        "qwen": ("QWEN_WIDTH", "QWEN_HEIGHT", "QWEN_STEPS", "QWEN_GUIDANCE"),
        "flux-schnell": (
            "FLUX_SCHNELL_WIDTH",
            "FLUX_SCHNELL_HEIGHT",
            "FLUX_SCHNELL_STEPS",
            "FLUX_SCHNELL_STRENGTH",
        ),
        "fooocus": (),
        "hidream": (
            "HIDREAM_MODEL_TYPE",
            "HIDREAM_WIDTH",
            "HIDREAM_HEIGHT",
            "HIDREAM_SHIFT",
        ),
        "flux2": ("FLUX2_WIDTH", "FLUX2_HEIGHT", "FLUX2_STEPS", "FLUX2_GUIDANCE"),
        "flux2-klein": (),
        "sdxl": (
            "SDXL_WIDTH",
            "SDXL_HEIGHT",
            "SDXL_STEPS",
            "SDXL_GUIDANCE",
            "SDXL_HUMAN_STRENGTH",
            "SDXL_FAST_HUMAN_WIDTH",
            "SDXL_FAST_HUMAN_HEIGHT",
            "SDXL_FAST_HUMAN_STEPS",
            "SDXL_FAST_HUMAN_STRENGTH",
        ),
    }
    return {name: getattr(settings, name) for name in names.get(engine, ())}


def _engine_python_path(engine: str) -> str | None:
    paths = {
        "qwen": settings.QWEN_PYTHON,
        "flux-schnell": settings.FLUX_SCHNELL_PYTHON,
        "fooocus": settings.FOOOCUS_PYTHON,
        "hidream": settings.HIDREAM_PYTHON,
        "flux2": settings.FLUX2_PYTHON,
        "flux2-klein": settings.FLUX2_KLEIN_PYTHON,
        "sdxl": settings.SDXL_PYTHON,
    }
    return paths.get(engine)


def _python_environment_signature(value: str | None) -> dict[str, object] | None:
    if not value:
        return None
    python = Path(value).expanduser()
    if not python.is_file():
        return {"python": str(python), "exists": False}
    environment_root = python.parent.parent
    metadata_files: list[Path] = []
    for site_packages in sorted(
        environment_root.glob("lib/python*/site-packages")
    ):
        metadata_files.extend(sorted(site_packages.glob("*.dist-info/METADATA")))
        metadata_files.extend(sorted(site_packages.glob("*.dist-info/direct_url.json")))
        metadata_files.extend(sorted(site_packages.glob("*.pth")))
        metadata_files.extend(sorted(site_packages.glob("torch/version.py")))
        metadata_files.extend(sorted(site_packages.glob("torch/lib/*")))
    marker_payload = {
        "python": _path_signature(str(python)),
        "pyvenv": _path_signature(str(environment_root / "pyvenv.cfg")),
        "packages": [
            _file_signature(path, environment_root) for path in metadata_files
        ],
        "nvidia_driver": _path_signature("/proc/driver/nvidia/version"),
    }
    marker = hashlib.sha256(
        json.dumps(marker_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    cached = _runtime_probe_cache.get(marker)
    if cached is not None:
        return cached
    probe = (
        "import importlib.metadata as m,json,platform,sys;"
        "versions={d.metadata.get('Name',str(d._path)):d.version for d in m.distributions()};"
        "data={'python':sys.version,'implementation':platform.python_implementation(),"
        "'packages':dict(sorted(versions.items()))};"
        "\nprint(json.dumps(data,sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        runtime: object = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        runtime = {"probe_error": type(exc).__name__}
    signature = {
        "marker": marker,
        "environment": marker_payload,
        "runtime": runtime,
    }
    if not isinstance(runtime, dict) or "probe_error" not in runtime:
        _runtime_probe_cache[marker] = signature
    return signature


def runtime_fingerprint(engine: str, base_status: dict[str, object]) -> str:
    stable = {
        key: value
        for key, value in base_status.items()
        if key not in _DYNAMIC_STATUS_FIELDS
    }
    stable["engine"] = engine
    stable["dependencies"] = [
        _path_signature(str(path)) if isinstance(path, Path) else _path_signature(path)
        for path in _engine_dependencies(engine, base_status)
    ]
    stable["generation_configuration"] = _engine_configuration(engine)
    stable["python_environment"] = _python_environment_signature(
        _engine_python_path(engine)
    )
    stable["validator"] = {
        "checkpoint": _path_signature(settings.HUMAN_PRODUCT_VALIDATOR_MODEL_PATH),
        "person_score": settings.HUMAN_PRODUCT_PERSON_SCORE,
        "bag_score": settings.HUMAN_PRODUCT_BAG_SCORE,
        "identity_score": settings.HUMAN_PRODUCT_IDENTITY_SCORE,
    }
    payload = json.dumps(stable, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_report() -> dict[str, Any]:
    path = Path(settings.GENERATION_VERIFICATION_REPORT_PATH).expanduser()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": REPORT_VERSION, "engines": {}}
    if payload.get("version") != REPORT_VERSION or not isinstance(payload.get("engines"), dict):
        return {"version": REPORT_VERSION, "engines": {}}
    return payload


def write_verification(
    engine: str,
    base_status: dict[str, object],
    *,
    status: str,
    reason: str,
    latency_ms: int | None = None,
    output_path: str | None = None,
    validations: dict[str, object] | None = None,
) -> dict[str, object]:
    if status not in {"passed", "failed", "blocked"}:
        raise ValueError(f"Unsupported verification status: {status}")
    report = _read_report()
    now = time.time()
    record: dict[str, object] = {
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "checked_at": _iso_timestamp(now),
        "checked_at_epoch": now,
        "expires_at": _iso_timestamp(
            now + (max(settings.GENERATION_VERIFICATION_TTL_HOURS, 1) * 3600)
        ),
        "runtime_fingerprint": runtime_fingerprint(engine, base_status),
        "latency_ms": latency_ms,
        "output_path": output_path,
        "validations": validations or {},
    }
    report["generated_at"] = _iso_timestamp(now)
    report["engines"][engine] = record
    path = Path(settings.GENERATION_VERIFICATION_REPORT_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        json.dump(report, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    return record


def verification_for(
    engine: str, base_status: dict[str, object]
) -> dict[str, object]:
    report = _read_report()
    record = report["engines"].get(engine)
    if not isinstance(record, dict):
        return {
            "status": "not-run",
            "passed": False,
            "current": False,
            "reason": "Human-with-product smoke test has not been run for this runtime.",
            "checked_at": None,
            "expires_at": None,
            "latency_ms": None,
            "output_path": None,
            "validations": {},
        }
    fingerprint_matches = record.get("runtime_fingerprint") == runtime_fingerprint(
        engine, base_status
    )
    checked_at_epoch = record.get("checked_at_epoch")
    try:
        age_seconds = max(0.0, time.time() - float(checked_at_epoch))
    except (TypeError, ValueError):
        age_seconds = float("inf")
    current = bool(
        fingerprint_matches
        and age_seconds
        <= max(settings.GENERATION_VERIFICATION_TTL_HOURS, 1) * 3600
    )
    passed = bool(record.get("status") == "passed" and current)
    reason = str(record.get("reason") or "Smoke test did not provide a reason.")
    if not fingerprint_matches:
        reason = "Runtime or model files changed after the last smoke test."
    elif not current:
        reason = "Human-with-product smoke test expired and must be run again."
    return {
        "status": record.get("status", "failed") if current else "stale",
        "passed": passed,
        "current": current,
        "reason": reason,
        "checked_at": record.get("checked_at"),
        "expires_at": record.get("expires_at"),
        "latency_ms": record.get("latency_ms"),
        "output_path": record.get("output_path"),
        "validations": record.get("validations") or {},
    }


def verified_status(
    engine: str, base_status: dict[str, object]
) -> dict[str, object]:
    decorated = dict(base_status)
    runtime_ready = bool(base_status.get("ready") is True)
    verification = verification_for(engine, base_status)
    verified = bool(runtime_ready and verification["passed"] is True)
    if verified:
        state = "verified"
        reason = None
    elif not runtime_ready:
        state = "unavailable"
        reason = str(
            base_status.get("reason")
            or "Local model files, runtime, runner, or required device are unavailable."
        )
    elif verification["status"] == "failed":
        state = "failed"
        reason = str(verification["reason"])
    else:
        state = "unverified"
        reason = str(verification["reason"])
    decorated.update(
        {
            "runtime_ready": runtime_ready,
            "ready": verified,
            "human_product_verified": verified,
            "verification_state": state,
            "verification": verification,
            "reason": reason,
        }
    )
    return decorate(engine, decorated)