"""Local Qwen Image Edit 2511 image-to-image generation.

Qwen Image Edit 2511 is the primary open-source product-preserving provider.
The API process only orchestrates a separate runner so a CUDA-only dependency
cannot make the control plane unavailable on CPU-only machines.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Iterable

from app.config import settings


class QwenImageEditError(RuntimeError):
    """Raised when local Qwen Image Edit is unavailable or fails."""


_active_processes: set[asyncio.subprocess.Process] = set()


def _cuda_status() -> tuple[bool, str]:
    if not Path("/dev/nvidiactl").exists() and not glob("/dev/nvidia[0-9]*"):
        return False, "No NVIDIA GPU detected"
    python = Path(settings.QWEN_PYTHON).expanduser()
    if not python.is_file():
        return False, "Qwen Python runtime unavailable"
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
        return False, "Qwen CUDA runtime unavailable"


def status() -> dict[str, object]:
    model = Path(settings.QWEN_MODEL_PATH).expanduser()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "qwen_image_edit_runner.py"
    python = Path(settings.QWEN_PYTHON).expanduser()
    cuda_available, device = _cuda_status()
    model_present = model.is_dir() and (
        (model / "model_index.json").is_file() or (model / "config.json").is_file()
    )
    return {
        "id": "qwen",
        "name": "Qwen Image Edit 2511",
        "mode": "local",
        "license": "Apache-2.0",
        "configured": bool(settings.QWEN_MODEL_PATH),
        "repository_present": True,
        "runner_present": runner.is_file(),
        "model_present": model_present,
        "cuda_available": cuda_available,
        "device": device,
        "model_type": "image-edit",
        "model_path": str(model),
        "python_present": python.is_file(),
        "supports_source_images": True,
        "supports_multi_reference": True,
        "active_processes": len(_active_processes),
        "ready": bool(
            model_present and runner.is_file() and python.is_file() and cuda_available
        ),
        "reason": (
            None
            if model_present and runner.is_file() and python.is_file() and cuda_available
            else "Qwen Image Edit requires local model weights, its isolated runtime, and a CUDA GPU."
        ),
    }


def _validate_configuration() -> tuple[Path, Path]:
    model = Path(settings.QWEN_MODEL_PATH).expanduser()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "qwen_image_edit_runner.py"
    python = Path(settings.QWEN_PYTHON).expanduser()
    if not model.is_dir():
        raise QwenImageEditError(f"Qwen Image Edit model directory not found: {model}")
    if not runner.is_file():
        raise QwenImageEditError(f"Qwen Image Edit runner not found: {runner}")
    if not python.is_file():
        raise QwenImageEditError(f"Qwen Image Edit Python runtime not found: {python}")
    cuda_available, _ = _cuda_status()
    if not cuda_available:
        raise QwenImageEditError(
            "Qwen Image Edit requires a CUDA-capable GPU. No hosted provider is enabled."
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
    if not 1 <= len(references) <= 6:
        raise QwenImageEditError("Qwen Image Edit requires between 1 and 6 reference images.")
    command = [
        settings.QWEN_PYTHON,
        str(runner),
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--output",
        output_path,
        "--width",
        str(settings.QWEN_WIDTH),
        "--height",
        str(settings.QWEN_HEIGHT),
        "--steps",
        str(settings.QWEN_STEPS),
        "--guidance",
        str(settings.QWEN_GUIDANCE),
        "--seed",
        str(seed),
        "--references",
        *references,
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
            process.communicate(), timeout=settings.QWEN_TIMEOUT_SECONDS
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
        raise QwenImageEditError(
            f"Qwen Image Edit failed with exit code {process.returncode}: {tail or 'no output'}"
        )
    result = Path(output_path)
    if not result.is_file():
        raise QwenImageEditError("Qwen Image Edit finished without creating an image.")
    return result.read_bytes()