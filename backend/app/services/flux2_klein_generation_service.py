"""Local FLUX.2 Klein 4B NVFP4 status.

The official NVFP4 checkpoint is installed locally, but this workspace does
not have a CUDA device or a compatible NVFP4 inference runner. Keep this
engine separate from FLUX.2 Dev so readiness never overclaims support.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.config import settings


def _cuda_status() -> tuple[bool, str]:
    if not Path("/dev/nvidiactl").exists() and not list(Path("/dev").glob("nvidia[0-9]*")):
        return False, "No NVIDIA GPU detected"
    python = Path(settings.FLUX2_KLEIN_PYTHON).expanduser()
    if not python.is_file():
        return False, "FLUX.2 Klein Python runtime unavailable"
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
        return False, "FLUX.2 Klein CUDA runtime unavailable"


def status() -> dict[str, object]:
    checkpoint = Path(settings.FLUX2_KLEIN_MODEL_PATH).expanduser()
    cuda_available, device = _cuda_status()
    python_present = Path(settings.FLUX2_KLEIN_PYTHON).expanduser().is_file()
    if not cuda_available:
        # Do not import the CUDA/NVFP4 stack on a CPU host just to render
        # an unavailable status. This endpoint is polled frequently.
        runtime_importable = False
    else:
        try:
            result = subprocess.run(
                [
                    settings.FLUX2_KLEIN_PYTHON,
                    "-c",
                    "from diffusers import Flux2KleinPipeline; import torchao",
                ],
                capture_output=True,
                timeout=10,
                check=False,
            )
            runtime_importable = result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            runtime_importable = False

    # Diffusers/torchao currently do not load the official single-file NVFP4
    # checkpoint directly. Keep this explicit until a compatible CUDA runner
    # is installed rather than treating imports alone as runtime support.
    runtime_supported = False
    return {
        "id": "flux2-klein",
        "name": "FLUX.2 Klein 4B (NVFP4)",
        "mode": "local",
        "configured": bool(settings.FLUX2_KLEIN_MODEL_PATH),
        "repository_present": checkpoint.is_file(),
        "runner_present": False,
        "model_present": checkpoint.is_file(),
        "checkpoint_present": checkpoint.is_file(),
        "quantized": True,
        "runtime_importable": runtime_importable,
        "runtime_supported": runtime_supported,
        "cuda_available": cuda_available,
        "device": device if cuda_available else "CPU · CUDA required",
        "model_type": "klein-4b-nvfp4",
        "model_path": str(checkpoint),
        "python_present": python_present,
        "active_processes": 0,
        "ready": bool(
            checkpoint.is_file()
            and python_present
            and cuda_available
            and runtime_supported
        ),
        "reason": (
            "Installed checkpoint requires a CUDA-compatible NVFP4 runner"
            + (" and an NVIDIA GPU." if not cuda_available else ".")
        ),
    }


class Flux2KleinGenerationError(RuntimeError):
    """Raised if generation is requested before an NVFP4 runner is installed."""


async def generate_frame(**_: object) -> bytes:
    raise Flux2KleinGenerationError(
        "FLUX.2 Klein 4B NVFP4 is installed but not runnable in this workspace. "
        "Use a CUDA machine with an NVFP4-compatible runner."
    )