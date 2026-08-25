"""Local FLUX.2 Dev image-to-image generation.

This service invokes the official Black Forest Labs native FLUX.2 repository.
It intentionally has no hosted fallback. The 32B checkpoint and autoencoder
must be installed locally on a CUDA-capable machine.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Iterable

from app.config import settings


class Flux2GenerationError(RuntimeError):
    """Raised when local FLUX.2 generation is not configured or fails."""


_active_processes: set[asyncio.subprocess.Process] = set()


def _cuda_status() -> tuple[bool, str]:
    if not Path("/dev/nvidiactl").exists() and not glob("/dev/nvidia[0-9]*"):
        return False, "No NVIDIA GPU detected"
    python = Path(settings.FLUX2_PYTHON).expanduser()
    if not python.is_file():
        return False, "FLUX.2 Python runtime unavailable"
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
        return False, "FLUX.2 CUDA runtime unavailable"


def status() -> dict[str, object]:
    repo = Path(settings.FLUX2_REPO).expanduser()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "flux2_runner.py"
    model = Path(settings.FLUX2_MODEL_PATH).expanduser()
    autoencoder = Path(settings.FLUX2_AE_MODEL_PATH).expanduser()
    cuda_available, device = _cuda_status()
    repository_present = (repo / "src" / "flux2" / "util.py").is_file()
    model_present = model.is_file()
    ae_present = autoencoder.is_file()
    configured = bool(settings.FLUX2_REPO and settings.FLUX2_MODEL_PATH and settings.FLUX2_AE_MODEL_PATH)
    runner_present = runner.is_file()
    ready = bool(
        configured
        and repository_present
        and runner_present
        and model_present
        and ae_present
        and cuda_available
    )
    missing: list[str] = []
    if not repository_present:
        missing.append("official repository")
    if not runner_present:
        missing.append("Atelier runner")
    if not model_present:
        missing.append("FLUX.2 Dev checkpoint")
    if not ae_present:
        missing.append("ae.safetensors")
    if not cuda_available:
        missing.append(device)
    return {
        "id": "flux2",
        "name": "FLUX.2 Dev",
        "mode": "local",
        "configured": configured,
        "repository_present": repository_present,
        "runner_present": runner_present,
        "model_present": model_present,
        "ae_present": ae_present,
        "cuda_available": cuda_available,
        "device": device,
        "model_type": "dev",
        "model_path": str(model),
        "active_processes": len(_active_processes),
        "ready": ready,
        "reason": None if ready else "FLUX.2 Dev requires: " + ", ".join(missing) + ".",
    }


def _validate_configuration() -> tuple[Path, Path, Path]:
    repo = Path(settings.FLUX2_REPO).expanduser()
    model = Path(settings.FLUX2_MODEL_PATH).expanduser()
    autoencoder = Path(settings.FLUX2_AE_MODEL_PATH).expanduser()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "flux2_runner.py"
    if not (repo / "src" / "flux2" / "util.py").is_file():
        raise Flux2GenerationError(f"FLUX.2 repository not found: {repo}")
    if not runner.is_file():
        raise Flux2GenerationError(f"FLUX.2 runner not found: {runner}")
    if not model.is_file():
        raise Flux2GenerationError(
            f"FLUX.2 Dev weights not found: {model}. Download the gated 60 GB checkpoint after accepting its license."
        )
    if not autoencoder.is_file():
        raise Flux2GenerationError(f"FLUX.2 autoencoder not found: {autoencoder}")
    cuda_available, _ = _cuda_status()
    if not cuda_available:
        raise Flux2GenerationError(
            "FLUX.2 Dev requires a CUDA-capable GPU. No hosted provider is enabled."
        )
    return repo, model, autoencoder


async def generate_frame(
    *,
    prompt: str,
    reference_paths: Iterable[str],
    output_path: str,
    seed: int,
) -> bytes:
    """Run the official FLUX.2 multi-reference editing path for one frame."""
    repo, model, autoencoder = _validate_configuration()
    references = [str(Path(path)) for path in reference_paths]
    if not 1 <= len(references) <= 6:
        raise Flux2GenerationError("FLUX.2 requires between 1 and 6 reference images.")
    runner = Path(__file__).resolve().parents[2] / "scripts" / "flux2_runner.py"
    command = [
        settings.FLUX2_PYTHON,
        str(runner),
        "--repo",
        str(repo),
        "--model",
        str(model),
        "--autoencoder",
        str(autoencoder),
        "--prompt",
        prompt,
        "--output",
        output_path,
        "--width",
        str(settings.FLUX2_WIDTH),
        "--height",
        str(settings.FLUX2_HEIGHT),
        "--steps",
        str(settings.FLUX2_STEPS),
        "--guidance",
        str(settings.FLUX2_GUIDANCE),
        "--seed",
        str(seed),
        "--references",
        *references,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(repo),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    _active_processes.add(process)
    try:
        output, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=settings.FLUX2_TIMEOUT_SECONDS,
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
        raise Flux2GenerationError(
            f"FLUX.2 failed with exit code {process.returncode}: {tail or 'no output'}"
        )
    result = Path(output_path)
    if not result.is_file():
        raise Flux2GenerationError("FLUX.2 finished without creating an image.")
    return result.read_bytes()