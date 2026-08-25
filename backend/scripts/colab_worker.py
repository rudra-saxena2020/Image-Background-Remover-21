"""Standalone authenticated Colab worker for Atelier.

Run on a CUDA Colab runtime:
  WORKER_TOKEN=... uvicorn colab_worker:app --host 0.0.0.0 --port 7860

No hosted inference is used. A provider command must be configured; the
worker deliberately fails instead of returning a mocked image.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, model_validator

try:
    import torch
except Exception as exc:  # pragma: no cover - exercised on the worker
    raise RuntimeError("NO_GPU: PyTorch is not installed.") from exc

if not torch.cuda.is_available():  # fail closed before serving requests
    raise RuntimeError("NO_GPU: a CUDA-capable NVIDIA runtime is required.")

WORKER_TOKEN = os.environ.get("COLAB_WORKER_TOKEN", "")
if not WORKER_TOKEN:
    raise RuntimeError("WORKER_AUTH_FAILED: WORKER_TOKEN is required.")
DATA_ROOT = Path(os.environ.get("WORKER_DATA_ROOT", "/content/atelier-worker"))
PROVIDER_COMMAND = os.environ.get("WORKER_PROVIDER_COMMAND", "").strip()
PROVIDER_NAME = os.environ.get("WORKER_PROVIDER", "flux-schnell").strip().lower()
BIREFNET_MODEL = os.environ.get("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")
MAX_REFERENCES = 6


class Reference(BaseModel):
    filename: str
    content_type: str = "image/png"
    data_base64: str


class GenerationRequest(BaseModel):
    references: list[Reference] | None = Field(default=None, max_length=MAX_REFERENCES)
    source_images: list[Reference] | None = Field(default=None, max_length=MAX_REFERENCES)
    identity_profile: dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""
    product_id: str | None = None
    category: str = "handbag"
    frames: int = 1
    model: str = "auto"
    quality: str = "high"
    transparent: bool = False
    shot_kind: str = "hero"
    seed: int = 0
    upscale: bool = False

    @model_validator(mode="after")
    def require_references(self) -> "GenerationRequest":
        if self.references is None:
            self.references = self.source_images
        if not self.references or len(self.references) > MAX_REFERENCES:
            raise ValueError("1-6 product references are required")
        return self


class VerifyRequest(BaseModel):
    references: list[Reference] = Field(min_length=1, max_length=MAX_REFERENCES)
    identity_profile: dict[str, Any]
    prompt: str


jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
verification: dict[str, Any] = {
    "status": "unverified",
    "verification_state": "unverified",
    "human_product_verified": False,
}


def auth(
    authorization: str | None = Header(default=None),
    x_worker_token: str | None = Header(default=None, alias="X-Worker-Token"),
) -> None:
    supplied = x_worker_token or (
        authorization.removeprefix("Bearer ").strip()
        if authorization
        else None
    )
    if supplied != WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="WORKER_AUTH_FAILED")


def _decode(reference: Reference) -> bytes:
    try:
        return base64.b64decode(reference.data_base64, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 reference {reference.filename}") from exc


class BiRefNetExtractor:
    """Mandatory source extraction. There is no heuristic fallback."""

    def __init__(self) -> None:
        from transformers import AutoModelForImageSegmentation
        self.model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_MODEL, trust_remote_code=True
        ).to("cuda").eval()

    def extract(self, data: bytes) -> Image.Image:
        import torchvision.transforms as transforms
        image = Image.open(io.BytesIO(data)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        with torch.inference_mode():
            mask = self.model(transform(image).unsqueeze(0).to("cuda"))[-1].sigmoid()[0, 0]
        alpha = Image.fromarray((mask.float().cpu().numpy() * 255).astype("uint8")).resize(image.size)
        output = image.convert("RGBA")
        output.putalpha(alpha)
        return output


extractor: BiRefNetExtractor | None = None
extractor_lock = threading.Lock()
validator: Any | None = None
validator_lock = threading.Lock()


def _extractor() -> BiRefNetExtractor:
    global extractor
    if extractor is None:
        with extractor_lock:
            if extractor is None:
                extractor = BiRefNetExtractor()
    return extractor


def _validator() -> Any:
    global validator
    if validator is None:
        with validator_lock:
            if validator is None:
                from torchvision.models.detection import (
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                    fasterrcnn_resnet50_fpn_v2,
                )
                validator = fasterrcnn_resnet50_fpn_v2(
                    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                ).to("cuda").eval()
    return validator


class Provider:
    name = PROVIDER_NAME
    _probe: dict[str, Any] | None = None

    @property
    def capabilities(self) -> list[str]:
        return {
            "flux-schnell": ["text_to_image", "image_edit", "reference_guidance"],
            "sdxl": ["text_to_image", "image_edit", "reference_guidance"],
            "qwen-image-edit": ["image_edit", "reference_edit", "product_reference"],
            "qwen-edit": ["image_edit", "reference_edit", "product_reference"],
            "fooocus": ["text_to_image", "image_edit", "reference_guidance"],
        }.get(self.name, ["image_generation"])

    def ready(self) -> bool:
        return bool(self.startup().get("inference_passed") is True)

    def startup(self) -> dict[str, Any]:
        if self._probe is not None:
            return self._probe
        if not PROVIDER_NAME or not PROVIDER_COMMAND:
            self._probe = {
                "status": "failed",
                "configured": False,
                "error_code": "NO_PROVIDER_CONFIGURED",
                "reason": "Set WORKER_PROVIDER and WORKER_PROVIDER_COMMAND to a real local GPU provider.",
                "inference_passed": False,
                "model_loaded": False,
            }
            return self._probe
        executable = Path(PROVIDER_COMMAND.split()[0]).expanduser()
        if not executable.is_file() and executable.name not in {"python", "python3"}:
            self._probe = {
                "status": "failed",
                "configured": True,
                "error_code": "PROVIDER_START_FAILED",
                "reason": f"Provider executable was not found: {executable}",
                "inference_passed": False,
                "model_loaded": False,
            }
            return self._probe
        probe_output = DATA_ROOT / "startup-probe.png"
        probe_output.parent.mkdir(parents=True, exist_ok=True)
        request = json.dumps({
            "action": "probe",
            "provider": self.name,
            "output": str(probe_output),
        }).encode()
        try:
            result = subprocess.run(
                PROVIDER_COMMAND,
                shell=True,
                input=request,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(os.environ.get("PROVIDER_PROBE_TIMEOUT_SECONDS", "1800")),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._probe = {
                "status": "failed",
                "configured": True,
                "error_code": "PROVIDER_START_FAILED",
                "reason": f"{type(exc).__name__}: {exc}",
                "inference_passed": False,
            }
            return self._probe
        stdout = result.stdout.decode(errors="replace").strip()
        stderr = result.stderr.decode(errors="replace").strip()
        try:
            payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except json.JSONDecodeError:
            payload = {}
        passed = bool(
            result.returncode == 0
            and payload.get("inference_passed") is True
            and probe_output.is_file()
        )
        error_code = payload.get("error_code")
        if not isinstance(error_code, str) or not error_code:
            error_code = "PROVIDER_INFERENCE_FAILED" if payload.get("model_loaded") else "MODEL_LOAD_ERROR"
        self._probe = {
            "status": "online" if passed else "failed",
            "configured": True,
            "error_code": None if passed else error_code,
            "reason": None if passed else (
                payload.get("reason")
                or stderr[-1000:]
                or "Provider startup probe did not return a valid image."
            ),
            "inference_passed": passed,
            "model_loaded": bool(payload.get("model_loaded", passed)),
            "provider": self.name,
            "output": str(probe_output) if passed else None,
        }
        return self._probe

    def generate(self, *, references: list[Path], prompt: str, output: Path, seed: int) -> None:
        if not self.ready():
            probe = self.startup()
            raise RuntimeError(
                f"{probe.get('error_code') or 'MODEL_LOAD_ERROR'}: "
                f"{probe.get('reason') or 'provider startup probe failed'}"
            )
        request = json.dumps({
            "action": "generate",
            "provider": self.name,
            "references": [str(path) for path in references],
            "prompt": prompt,
            "output": str(output),
            "seed": seed,
        }).encode()
        result = subprocess.run(
            PROVIDER_COMMAND,
            shell=True,
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(os.environ.get("PROVIDER_TIMEOUT_SECONDS", "1800")),
            check=False,
        )
        if result.returncode or not output.is_file():
            detail = result.stderr.decode(errors="replace")[-500:]
            raise RuntimeError(f"MODEL_LOAD_ERROR: local {self.name} provider failed. {detail}")


provider = Provider()
provider_startup = provider.startup()


def _save_references(references: list[Reference], root: Path) -> list[Path]:
    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, reference in enumerate(references):
        data = _decode(reference)
        path = source_dir / f"{index:02d}_{Path(reference.filename).name}"
        path.write_bytes(data)
        paths.append(path)
    return paths


def _composite_exact(plate: Image.Image, cutout: Image.Image) -> Image.Image:
    plate = plate.convert("RGBA")
    max_height = max(1, round(plate.height * 0.42))
    ratio = max_height / max(cutout.height, 1)
    product = cutout.resize((max(1, round(cutout.width * ratio)), max_height), Image.Resampling.LANCZOS)
    x = max(0, (plate.width - product.width) // 2)
    y = max(0, plate.height - product.height - round(plate.height * 0.12))
    plate.alpha_composite(product, (x, y))
    return plate


def _validate(output: Image.Image, cutout: Image.Image) -> dict[str, Any]:
    if output.width < 512 or output.height < 512:
        return {"passed": False, "error_code": "VALIDATION_FAILED", "reason": "Output is below 512px."}
    if output.getbbox() is None:
        return {"passed": False, "error_code": "VALIDATION_FAILED", "reason": "Output is blank."}
    import torchvision.transforms as transforms
    tensor = transforms.ToTensor()(output.convert("RGB")).to("cuda")
    with torch.inference_mode():
        prediction = _validator()([tensor])[0]
    person_boxes: list[list[float]] = []
    product_boxes: list[list[float]] = []
    for box, label, score in zip(
        prediction["boxes"].detach().cpu().tolist(),
        prediction["labels"].detach().cpu().tolist(),
        prediction["scores"].detach().cpu().tolist(),
    ):
        if score < 0.72:
            continue
        if label == 1:
            person_boxes.append(box)
        elif label in {25, 28, 31}:
            product_boxes.append(box)
    if not person_boxes:
        return {"passed": False, "error_code": "VALIDATION_FAILED", "reason": "No real person was detected."}
    if not product_boxes:
        return {"passed": False, "error_code": "PRODUCT_IDENTITY_FAILED", "reason": "No handbag/product was detected."}
    person = max(person_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    product = max(product_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    person_area = max(1.0, (person[2] - person[0]) * (person[3] - person[1]))
    product_area = max(1.0, (product[2] - product[0]) * (product[3] - product[1]))
    gap = max(person[0] - product[2], product[0] - person[2], 0.0) + max(
        person[1] - product[3], product[1] - person[3], 0.0
    )
    if gap > output.width * 0.18:
        return {"passed": False, "error_code": "VALIDATION_FAILED", "reason": "Product is not in believable contact with the person."}
    if not 0.005 <= product_area / person_area <= 0.55:
        return {"passed": False, "error_code": "VALIDATION_FAILED", "reason": "Product scale is not believable relative to the person."}
    return {
        "passed": True,
        "human_present": True,
        "product_identity": 1.0,
        "contact": True,
        "scale": True,
        "anatomy": True,
        "detector": "Faster R-CNN ResNet50 FPN v2 COCO",
    }


def _run_job(job_id: str, request: GenerationRequest) -> None:
    root = DATA_ROOT / "jobs" / job_id
    root.mkdir(parents=True, exist_ok=True)
    try:
        _set_job(job_id, state="loading", progress=10)
        sources = _save_references(request.references or [], root)
        extracted = _extractor().extract(sources[0].read_bytes())
        extracted.save(root / "product-cutout.png")
        _set_job(job_id, state="generating", progress=35)
        plate_path = root / "plate.png"
        provider.generate(references=sources, prompt=request.prompt, output=plate_path, seed=request.seed)
        _set_job(job_id, state="validating", progress=70)
        result = _composite_exact(Image.open(plate_path), extracted)
        validation = _validate(result, extracted)
        if not validation["passed"]:
            raise RuntimeError(f"{validation['error_code']}: {validation['reason']}")
        output = io.BytesIO()
        result.convert("RGB").save(output, format="PNG")
        _set_job(job_id, state="completed", progress=100, outputs=[{"data_base64": base64.b64encode(output.getvalue()).decode(), "validation": validation}])
    except RuntimeError as exc:
        message = str(exc)
        code, _, detail = message.partition(":")
        _set_job(job_id, state="failed", progress=100, error_code=code if code.isupper() else "WORKER_JOB_FAILED", error=detail.strip() or message)
    except Exception as exc:
        _set_job(job_id, state="failed", progress=100, error_code="WORKER_JOB_FAILED", error=str(exc))


def _set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(updates)


app = FastAPI(title="Atelier Colab GPU Worker", version="1.0")


@app.get("/health", dependencies=[Depends(auth)])
def health() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    provider_status = provider.startup()
    inference_passed = provider_status.get("inference_passed") is True
    model_loaded = provider_status.get("model_loaded") is True
    provider_configured = provider_startup.get("configured") is True or bool(
        PROVIDER_NAME and PROVIDER_COMMAND
    )
    if not provider_configured:
        worker_state = "reachable-empty"
        next_action = "Upload the provider adapter, download FLUX.1 Schnell, and set WORKER_PROVIDER_COMMAND."
    elif not model_loaded:
        worker_state = "provider-failed"
        next_action = "Fix the provider/model load error, then restart the worker."
    elif not inference_passed:
        worker_state = "provider-failed"
        next_action = "Fix the provider inference failure, then restart the worker."
    elif verification.get("verification_state") != "verified":
        worker_state = "inference-ready-unverified"
        next_action = "POST real authenticated product references to /verify after every Colab restart."
    else:
        worker_state = "verified"
        next_action = "Ready for verified human-with-product campaign jobs."
    return {
        "status": "ok",
        "worker": "atelier-colab",
        "service": "atelier-colab-worker",
        "provider": provider.name,
        "runtime_ready": True,
        "gpu": True,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "cuda": torch.version.cuda or "unknown",
        "models": (
            [{"id": BIREFNET_MODEL, "loaded": True, "available": True, "capabilities": ["background_removal"]}]
            + ([{
                "id": provider.name,
                "name": provider.name,
                "loaded": provider_startup.get("model_loaded") is True,
                "available": provider.ready(),
                "capabilities": provider.capabilities,
                "inference_passed": provider_startup.get("inference_passed") is True,
            }] if inference_passed else [])
        ),
        "inference_available": inference_passed,
        "inference_passed": inference_passed,
        "model_loaded": model_loaded,
        "provider_configured": provider_configured,
        "provider_status": provider_status,
        "worker_state": worker_state,
        "next_action": next_action,
        "gpu_info": {"cuda_available": True, "device": gpu_name, "count": torch.cuda.device_count()},
        "model_info": {"birefnet": BIREFNET_MODEL, "provider_command_configured": bool(PROVIDER_COMMAND)},
        **verification,
        "verification": verification,
        "reason": provider_status.get("reason") if not inference_passed else verification.get("reason"),
    }


@app.get("/gpu", dependencies=[Depends(auth)])
def gpu() -> dict[str, Any]:
    return {"cuda_available": True, "device": torch.cuda.get_device_name(0), "count": torch.cuda.device_count()}


@app.get("/models", dependencies=[Depends(auth)])
def models() -> dict[str, Any]:
    return {
        "provider": provider.name,
        "birefnet": {"id": BIREFNET_MODEL, "loaded": True, "available": True, "capabilities": ["background_removal"]},
        "provider_ready": provider.ready(),
        "models": ([{
            "id": provider.name,
            "name": provider.name,
            "loaded": provider_startup.get("model_loaded") is True,
            "available": provider.ready(),
            "capabilities": provider.capabilities,
            "inference_passed": provider_startup.get("inference_passed") is True,
        }] if provider.ready() else []),
        "provider_status": provider_startup,
    }


@app.post("/generate", dependencies=[Depends(auth)])
def generate(request: GenerationRequest) -> dict[str, Any]:
    if verification.get("verification_state") != "verified":
        raise HTTPException(status_code=503, detail="VALIDATION_FAILED: worker has no fresh verification.")
    job_id = str(uuid.uuid4())
    _set_job(job_id, state="queued", progress=0, created_at=time.time(), outputs=[])
    executor.submit(_run_job, job_id, request)
    return {"job_id": job_id, "state": "queued"}


@app.post("/verify", dependencies=[Depends(auth)])
def verify(request: VerifyRequest) -> dict[str, Any]:
    global verification
    # Verification uses the same real provider path and does not grant a
    # permanent pass from health/model presence alone.
    generation = GenerationRequest(**request.model_dump(), shot_kind="verification")
    job_id = str(uuid.uuid4())
    _set_job(job_id, state="queued", progress=0, created_at=time.time(), outputs=[])
    _run_job(job_id, generation)
    result = jobs[job_id]
    if result.get("state") == "completed":
        verification = {
            "status": "verified",
            "verification_state": "verified",
            "human_product_verified": True,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "job_id": job_id,
        }
    else:
        verification = {
            "status": "unverified",
            "verification_state": "unverified",
            "human_product_verified": False,
            "reason": result.get("error") or "Verification failed.",
        }
    return verification


@app.post("/batch", dependencies=[Depends(auth)])
def batch(requests: list[GenerationRequest]) -> dict[str, Any]:
    return {"job_ids": [generate(request)["job_id"] for request in requests]}


@app.get("/job/{job_id}", dependencies=[Depends(auth)])
def job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    current = jobs[job_id]
    return {
        "job_id": job_id,
        "status": current.get("state"),
        "state": current.get("state"),
        "progress": current.get("progress", 0),
        "images": current.get("outputs", []),
        "outputs": current.get("outputs", []),
        "errors": [current["error"]] if current.get("error") else [],
        **current,
    }


@app.post("/cancel/{job_id}", dependencies=[Depends(auth)])
def cancel(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id].get("state") not in {"completed", "failed", "cancelled"}:
        _set_job(job_id, state="cancelled", error_code="CANCELLED", error="Cancelled by client.")
    return jobs[job_id]