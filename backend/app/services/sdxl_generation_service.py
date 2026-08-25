"""Local Stable Diffusion XL image-to-image generation.

SDXL is the local fallback for machines without the larger CUDA engines. It
accepts one uploaded reference directly and never calls a hosted provider.
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


class SdxlGenerationError(RuntimeError):
    """Raised when local SDXL generation is not configured or fails."""


_active_processes: set[asyncio.subprocess.Process] = set()
_worker_process: asyncio.subprocess.Process | None = None
_worker_model: Path | None = None
_worker_ready = False
_worker_lock = asyncio.Lock()
_warmup_task: asyncio.Task[None] | None = None


def _cuda_status() -> tuple[bool, str]:
    if not Path("/dev/nvidiactl").exists() and not glob("/dev/nvidia[0-9]*"):
        return False, "No NVIDIA GPU detected"
    python = Path(settings.SDXL_PYTHON).expanduser()
    if not python.is_file():
        return False, "SDXL Python runtime unavailable"
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
        return False, "SDXL CUDA runtime unavailable"


def _python_available() -> bool:
    python = Path(settings.SDXL_PYTHON).expanduser()
    # Runtime imports happen inside the isolated persistent worker. Importing
    # torch/diffusers during every health probe can time out while the worker
    # is warming and incorrectly hide an otherwise valid local installation.
    return python.is_file()


def _model_present(model: Path) -> bool:
    required = (
        model / "model_index.json",
        model / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        model / "text_encoder" / "model.fp16.safetensors",
        model / "text_encoder_2" / "model.fp16.safetensors",
        model / "vae" / "diffusion_pytorch_model.fp16.safetensors",
    )
    return all(path.is_file() for path in required)


def status() -> dict[str, object]:
    model = Path(settings.SDXL_MODEL_PATH).expanduser()
    python_present = _python_available()
    model_present = model.is_dir() and _model_present(model)
    cuda_available, cuda_device = _cuda_status()
    cpu_ready = python_present and model_present
    return {
        "id": "sdxl",
        "name": "Stable Diffusion XL",
        "mode": "local",
        "cpu_available": cpu_ready,
        "configured": bool(settings.SDXL_MODEL_PATH),
        "repository_present": True,
        "runner_present": (Path(__file__).resolve().parents[2] / "scripts" / "sdxl_runner.py").is_file(),
        "model_present": model_present,
        "cuda_available": cuda_available,
        "device": cuda_device if cuda_available else ("CPU · slower" if cpu_ready else cuda_device),
        "model_path": str(model),
        "active_processes": len(_active_processes),
        "worker_ready": bool(
            _worker_ready
            and _worker_process is not None
            and _worker_process.returncode is None
        ),
        "ready": bool(
            python_present
            and model_present
            and (Path(__file__).resolve().parents[2] / "scripts" / "sdxl_runner.py").is_file()
            and (cuda_available or (cpu_ready and settings.SDXL_ALLOW_CPU_GENERATION))
        ),
        "reason": (
            None
            if cuda_available and _worker_ready
            else (
                "SDXL worker is warming up; fast preview will unlock when it is ready."
                if cpu_ready and settings.SDXL_ALLOW_CPU_GENERATION and not _worker_ready
                else "Ready locally on CPU; generation is slower. Use a CUDA machine for faster campaigns."
                if cpu_ready and settings.SDXL_ALLOW_CPU_GENERATION
                else "SDXL requires its local model/runtime."
            )
        ),
    }


def _validate_configuration() -> tuple[Path, Path]:
    model = Path(settings.SDXL_MODEL_PATH).expanduser()
    runner = Path(__file__).resolve().parents[2] / "scripts" / "sdxl_runner.py"
    if not _python_available():
        raise SdxlGenerationError("The local SDXL Python runtime is unavailable.")
    if not _model_present(model):
        raise SdxlGenerationError(f"SDXL model files are incomplete: {model}")
    if not runner.is_file():
        raise SdxlGenerationError(f"SDXL runner not found: {runner}")
    cuda_available, _ = _cuda_status()
    if not cuda_available and not settings.SDXL_ALLOW_CPU_GENERATION:
        raise SdxlGenerationError(
            "Stable Diffusion XL CPU generation is disabled. "
            "Enable SDXL_ALLOW_CPU_GENERATION or use a CUDA-capable GPU."
        )
    return model, runner


async def _stop_worker() -> None:
    global _worker_process, _worker_model, _worker_ready
    process = _worker_process
    _worker_process = None
    _worker_model = None
    _worker_ready = False
    if process is None:
        return
    if process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
    _active_processes.discard(process)


async def _ensure_worker(model: Path, runner: Path) -> asyncio.subprocess.Process:
    global _worker_process, _worker_model, _worker_ready
    if (
        _worker_process is not None
        and _worker_process.returncode is None
        and _worker_model == model
        and _worker_ready
    ):
        return _worker_process
    await _stop_worker()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cpu_threads = str(max(1, settings.SDXL_CPU_THREADS))
    env["OMP_NUM_THREADS"] = cpu_threads
    env["MKL_NUM_THREADS"] = cpu_threads
    env["OPENBLAS_NUM_THREADS"] = cpu_threads
    env["NUMEXPR_NUM_THREADS"] = cpu_threads
    env.setdefault("MALLOC_ARENA_MAX", "1")
    process = await asyncio.create_subprocess_exec(
        settings.SDXL_PYTHON,
        str(runner),
        "--model",
        str(model),
        "--server",
        cwd=str(model),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    _worker_process = process
    _worker_model = model
    _active_processes.add(process)
    try:
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=settings.SDXL_TIMEOUT_SECONDS)
            if not line:
                raise SdxlGenerationError("SDXL worker exited while loading the local model.")
            if line.decode("utf-8", errors="replace").strip() == "__SDXL_READY__":
                _worker_ready = True
                return process
    except Exception:
        await _stop_worker()
        raise


async def generate_frame(
    *,
    prompt: str,
    reference_paths: Iterable[str],
    output_path: str,
    seed: int,
    reference_index: int = 0,
    strength: float | None = None,
    fast: bool = False,
    human_context: bool = False,
) -> bytes:
    """Run SDXL image-to-image using the reference appropriate to this shot."""
    model, runner = _validate_configuration()
    references = [str(Path(path)) for path in reference_paths]
    if not references:
        raise SdxlGenerationError("SDXL requires at least one reference image.")
    if not 0 <= reference_index < len(references):
        raise SdxlGenerationError(
            f"SDXL reference index {reference_index} is outside the available "
            f"reference range 0..{len(references) - 1}."
        )
    cuda_available, _ = _cuda_status()
    fast_human = fast and human_context and not cuda_available
    steps = (
        settings.SDXL_STEPS
        if cuda_available
        else settings.SDXL_FAST_HUMAN_STEPS
        if fast_human
        else settings.SDXL_CPU_STEPS
    )
    job = {
        "prompt": " ".join(prompt.split())[:460],
        "reference": references[reference_index],
        "output": output_path,
        "width": (
            settings.SDXL_FAST_HUMAN_WIDTH
            if fast_human
            else settings.SDXL_FAST_WIDTH
            if fast and not cuda_available
            else settings.SDXL_WIDTH
        ),
        "height": (
            settings.SDXL_FAST_HUMAN_HEIGHT
            if fast_human
            else settings.SDXL_FAST_HEIGHT
            if fast and not cuda_available
            else settings.SDXL_HEIGHT
        ),
        "steps": (
            steps
            if fast_human or cuda_available
            else max(steps, 4)
            if fast
            else max(steps, 8)
        ),
        "guidance": settings.SDXL_GUIDANCE,
        "strength": (
            settings.SDXL_FAST_HUMAN_STRENGTH
            if fast_human and strength is None
            else settings.SDXL_HUMAN_STRENGTH
            if strength is None
            else strength
        ),
        "seed": seed,
    }
    async with _worker_lock:
        try:
            process = await _ensure_worker(model, runner)
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write((json.dumps(job) + "\n").encode("utf-8"))
            await process.stdin.drain()
            raw_result = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=settings.SDXL_TIMEOUT_SECONDS,
            )
            if not raw_result:
                raise SdxlGenerationError("SDXL worker exited without returning an image.")
            result = json.loads(raw_result.decode("utf-8"))
            if not result.get("ok"):
                raise SdxlGenerationError(str(result.get("error") or "SDXL worker failed."))
        except asyncio.CancelledError:
            await _stop_worker()
            raise
        except Exception:
            await _stop_worker()
            raise
    result = Path(output_path)
    if not result.is_file():
        raise SdxlGenerationError("SDXL finished without creating an image.")
    return result.read_bytes()


async def warm_up() -> None:
    """Load the persistent SDXL worker before the first user request."""
    model, runner = _validate_configuration()
    async with _worker_lock:
        await _ensure_worker(model, runner)


def schedule_warm_up() -> None:
    """Start one background warm-up and allow later requests to recover it."""
    global _warmup_task
    if _warmup_task is not None and not _warmup_task.done():
        return

    async def run() -> None:
        try:
            await warm_up()
        except Exception:
            # The health endpoint exposes the unavailable state. A later
            # request can schedule another attempt without duplicating workers.
            return

    _warmup_task = asyncio.create_task(run())