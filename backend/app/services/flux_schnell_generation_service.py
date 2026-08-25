"""Local FLUX.1 Schnell image-to-image generation.

FLUX.1 Schnell is an Apache-2.0 open-weight fast preview provider. It is
strictly image-conditioned here; it is never used as a text-only replacement
for an uploaded product reference.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Iterable

from app.config import settings


class FluxSchnellGenerationError(RuntimeError):
    """Raised when local FLUX.1 Schnell is unavailable or fails."""


_active_processes: set[asyncio.subprocess.Process] = set()


def _cuda_status() -> tuple[bool, str]:
    if not Path("/dev/nvidiactl").exists() and not glob("/dev/nvidia[0-9]*"):
        return False, "No NVIDIA GPU detected"
    python = Path(settings.FLUX_SCHNELL_PYTHON).expanduser()
    if not python.is_file():
        return False, "FLUX.1 Schnell Python runtime unavailable"
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import torch; "
                    "available = bool(torch.cuda.is_available()); "
                    "print(('1' if available else '0') + '|' + "
                    "(torch.cuda.get_device_name(0) if available else 'CPU'))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        available, device = result.stdout.strip().split("|", 1)
        return available == "1", device
    except (OSError, subprocess.SubprocessError, ValueError):
        return False, "FLUX.1 Schnell CUDA runtime unavailable"


def status() -> dict[str, object]:
    model = Path(settings.FLUX_SCHNELL_MODEL_PATH).expanduser()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "flux_schnell_runner.py"
    python = Path(settings.FLUX_SCHNELL_PYTHON).expanduser()
    cuda_available, device = _cuda_status()
    model_present = model.is_dir() and (
        (model / "model_index.json").is_file() or (model / "transformer").is_dir()
    )
    return {
        "id": "flux-schnell",
        "name": "FLUX.1 Schnell",
        "mode": "local",
        "license": "Apache-2.0",
        "configured": bool(settings.FLUX_SCHNELL_MODEL_PATH),
        "repository_present": True,
        "runner_present": runner.is_file(),
        "model_present": model_present,
        "cuda_available": cuda_available,
        "device": device,
        "model_type": "image-to-image",
        "model_path": str(model),
        "python_present": python.is_file(),
        "supports_source_images": True,
        "supports_multi_reference": False,
        "active_processes": len(_active_processes),
        "ready": bool(
            model_present and runner.is_file() and python.is_file() and cuda_available
        ),
        "reason": (
            None
            if model_present and runner.is_file() and python.is_file() and cuda_available
            else "FLUX.1 Schnell requires local model weights, its isolated runtime, and a CUDA GPU."
        ),
    }


def _validate_configuration() -> tuple[Path, Path]:
    model = Path(settings.FLUX_SCHNELL_MODEL_PATH).expanduser()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "flux_schnell_runner.py"
    python = Path(settings.FLUX_SCHNELL_PYTHON).expanduser()
    if not model.is_dir():
        raise FluxSchnellGenerationError(
            f"FLUX.1 Schnell model directory not found: {model}"
        )
    if not runner.is_file():
        raise FluxSchnellGenerationError(f"FLUX.1 Schnell runner not found: {runner}")
    if not python.is_file():
        raise FluxSchnellGenerationError(
            f"FLUX.1 Schnell Python runtime not found: {python}"
        )
    cuda_available, _ = _cuda_status()
    if not cuda_available:
        raise FluxSchnellGenerationError(
            "FLUX.1 Schnell requires a CUDA-capable GPU. No hosted provider is enabled."
        )
    return model, runner


async def generate_frame(
    *,
    prompt: str,
    reference_paths: Iterable[str],
    output_path: str,
    seed: int,
) -> bytes:
    model, runner = _validate_configuration()
    references = [str(Path(path)) for path in reference_paths]
    if not references:
        raise FluxSchnellGenerationError(
            "FLUX.1 Schnell requires an uploaded product reference image."
        )
    command = [
        settings.FLUX_SCHNELL_PYTHON,
        str(runner),
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--reference",
        references[0],
        "--output",
        output_path,
        "--width",
        str(settings.FLUX_SCHNELL_WIDTH),
        "--height",
        str(settings.FLUX_SCHNELL_HEIGHT),
        "--steps",
        str(settings.FLUX_SCHNELL_STEPS),
        "--strength",
        str(settings.FLUX_SCHNELL_STRENGTH),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(model),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    _active_processes.add(process)
    try:
        output, _ = await asyncio.wait_for(
            process.communicate(), timeout=settings.FLUX_SCHNELL_TIMEOUT_SECONDS
        )
    except asyncio.CancelledError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        raise
    finally:
        _active_processes.discard(process)
    if process.returncode != 0:
        tail = output.decode("utf-8", errors="replace")[-3000:].strip()
        raise FluxSchnellGenerationError(
            f"FLUX.1 Schnell failed with exit code {process.returncode}: {tail or 'no output'}"
        )
    result = Path(output_path)
    if not result.is_file():
        raise FluxSchnellGenerationError("FLUX.1 Schnell finished without creating an image.")
    return result.read_bytes()