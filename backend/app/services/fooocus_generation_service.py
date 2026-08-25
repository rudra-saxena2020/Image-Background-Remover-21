"""Readiness checks for the official local Fooocus installation.

Fooocus is intentionally not selected on this CPU-only host. The service
reports whether the checkout, isolated runtime, checkpoints, and CUDA device
are ready so a future GPU worker can enable it without changing the app's
model router.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
import subprocess

from app.config import settings


def _cuda_available() -> tuple[bool, str]:
    if not Path("/dev/nvidiactl").exists() and not glob("/dev/nvidia[0-9]*"):
        return False, "No NVIDIA GPU detected"
    python = Path(settings.FOOOCUS_PYTHON).expanduser()
    if not python.is_file():
        return False, "Fooocus Python runtime unavailable"
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
        return False, "Fooocus CUDA runtime unavailable"


def _checkpoint_present(root: Path) -> bool:
    checkpoints = root / "models" / "checkpoints"
    return checkpoints.is_dir() and any(
        path.is_file()
        and path.suffix.lower() in {".safetensors", ".ckpt", ".pth"}
        for path in checkpoints.iterdir()
    )


def status() -> dict[str, object]:
    root = Path(settings.FOOOCUS_ROOT).expanduser()
    repository_present = (root / "entry_with_update.py").is_file()
    python_present = Path(settings.FOOOCUS_PYTHON).expanduser().is_file()
    model_present = _checkpoint_present(root)
    cuda_available, cuda_device = _cuda_available()
    ready = repository_present and python_present and model_present and cuda_available
    return {
        "id": "fooocus",
        "name": "Fooocus",
        "mode": "local",
        "configured": repository_present,
        "repository_present": repository_present,
        "runner_present": (root / "entry_with_update.py").is_file(),
        "python_present": python_present,
        "model_present": model_present,
        "cuda_available": cuda_available,
        "device": cuda_device,
        "model_path": str(root / "models" / "checkpoints"),
        "active_processes": 0,
        "ready": ready,
        "reason": (
            None
            if ready
            else (
                "Fooocus source is installed, but this host has no NVIDIA GPU. "
                "Complete the isolated CUDA install on a GPU worker before enabling Fooocus."
                if repository_present and not cuda_available
                else "Fooocus needs its isolated runtime and local SDXL checkpoint."
            )
        ),
    }