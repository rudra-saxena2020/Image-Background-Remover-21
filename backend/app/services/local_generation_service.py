"""Local-only HiDream image generation.

This service deliberately has no network/provider fallback. The model weights
and the official HiDream repository must be present on the machine running the
API, together with a CUDA-capable PyTorch installation.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Iterable

from app.config import settings


class LocalGenerationError(RuntimeError):
    """Raised when local HiDream generation is not configured or fails."""


_active_processes: set[asyncio.subprocess.Process] = set()


def _model_files_present(model: Path) -> bool:
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        return False
    try:
        weight_files = set(json.loads(index.read_text())["weight_map"].values())
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return all((model / filename).is_file() for filename in weight_files)


def _cuda_status() -> tuple[bool, str]:
    # Avoid importing the CUDA-enabled torch wheel on CPU-only hosts. The
    # import can take tens of seconds while probing unavailable CUDA libraries,
    # which would make every /api/health request appear hung.
    if not Path("/dev/nvidiactl").exists() and not glob("/dev/nvidia[0-9]*"):
        return False, "No NVIDIA GPU detected"
    python = Path(settings.HIDREAM_PYTHON).expanduser()
    if not python.is_file():
        return False, "HiDream Python runtime unavailable"
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
        return False, "HiDream CUDA runtime unavailable"


def status() -> dict[str, object]:
    repo = Path(settings.HIDREAM_REPO).expanduser() if settings.HIDREAM_REPO else None
    model = (
        Path(settings.HIDREAM_MODEL_PATH).expanduser()
        if settings.HIDREAM_MODEL_PATH
        else None
    )
    script = repo / "inference.py" if repo else None
    cuda_available, device = _cuda_status()
    configured = bool(settings.HIDREAM_REPO and settings.HIDREAM_MODEL_PATH)
    model_present = bool(model and model.is_dir() and _model_files_present(model))
    repository_present = bool(repo and repo.is_dir())
    runner_present = bool(script and script.is_file())
    ready = bool(
        configured
        and repository_present
        and runner_present
        and model
        and model.is_dir()
        and model_present
        and cuda_available
    )
    missing: list[str] = []
    if not repository_present:
        missing.append("official repository")
    if not runner_present:
        missing.append("inference runner")
    if not model_present:
        missing.append("complete model weights")
    if not cuda_available:
        missing.append(device)
    return {
        "id": "hidream",
        "name": "HiDream-O1 Image",
        "mode": "local",
        "cpu_available": True,
        "configured": configured,
        "repository_present": repository_present,
        "runner_present": runner_present,
        "model_present": model_present,
        "cuda_available": cuda_available,
        "device": device,
        "model_type": settings.HIDREAM_MODEL_TYPE,
        "model_path": str(model) if model else None,
        "active_processes": len(_active_processes),
        "ready": ready,
        "reason": None if ready else "HiDream requires: " + ", ".join(missing) + ".",
    }


def _validate_configuration() -> tuple[Path, Path]:
    repo = Path(settings.HIDREAM_REPO).expanduser()
    model = Path(settings.HIDREAM_MODEL_PATH).expanduser()
    script = repo / "inference.py"
    if not settings.HIDREAM_MODEL_PATH or not settings.HIDREAM_REPO:
        raise LocalGenerationError(
            "Local HiDream is not configured. Set HIDREAM_REPO and HIDREAM_MODEL_PATH "
            "to the downloaded open-source model before starting a shoot."
        )
    if not script.is_file():
        raise LocalGenerationError(f"HiDream runner not found: {script}")
    if not model.is_dir() or not _model_files_present(model):
        raise LocalGenerationError(f"HiDream model directory not found: {model}")
    cuda_available, _ = _cuda_status()
    if not cuda_available:
        raise LocalGenerationError(
            "Local HiDream requires a CUDA-capable GPU. No paid cloud provider is enabled."
        )
    return repo, model


async def generate_frame(
    *,
    prompt: str,
    reference_paths: Iterable[str],
    output_path: str,
    seed: int,
) -> bytes:
    """Run the official HiDream inference script for one frame."""
    repo, model = _validate_configuration()
    references = [str(Path(path)) for path in reference_paths]
    if not 1 <= len(references) <= 6:
        raise LocalGenerationError("Local HiDream requires between 1 and 6 reference images.")

    command = [
        settings.HIDREAM_PYTHON,
        str(repo / "inference.py"),
        "--model_path",
        str(model),
        "--prompt",
        prompt,
        "--ref_images",
        *references,
        "--output_image",
        output_path,
        "--height",
        str(settings.HIDREAM_HEIGHT),
        "--width",
        str(settings.HIDREAM_WIDTH),
        "--model_type",
        settings.HIDREAM_MODEL_TYPE,
        "--seed",
        str(seed),
        "--shift",
        str(settings.HIDREAM_SHIFT),
    ]
    # HiDream's documented single-reference editing path preserves the source
    # aspect ratio. Multi-reference subject personalization uses the normal
    # fixed canvas path.
    if len(references) == 1:
        command.append("--keep_original_aspect")

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
            timeout=settings.HIDREAM_TIMEOUT_SECONDS,
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
        tail = output.decode("utf-8", errors="replace")[-2000:].strip()
        raise LocalGenerationError(
            f"Local HiDream failed with exit code {process.returncode}: {tail or 'no output'}"
        )
    result = Path(output_path)
    if not result.is_file():
        raise LocalGenerationError("Local HiDream finished without creating an image.")
    return result.read_bytes()