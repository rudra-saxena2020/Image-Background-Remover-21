"""Run and persist the local human-with-product backend verification matrix."""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import shutil
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable
from urllib.request import urlopen

from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.flux2_generation_service import (
    generate_frame as generate_flux2,
    status as flux2_status,
)
from app.services.flux2_klein_generation_service import (
    generate_frame as generate_flux2_klein,
    status as flux2_klein_status,
)
from app.services.flux_schnell_generation_service import (
    generate_frame as generate_flux_schnell,
    status as flux_schnell_status,
)
from app.services.fooocus_generation_service import status as fooocus_status
from app.services.generation_verification_service import write_verification
from app.services.human_product_validation_service import (
    VALIDATOR_SHA256,
    validate_human_product,
    validator_status,
)
from app.services.local_generation_service import (
    generate_frame as generate_hidream,
    status as hidream_status,
)
from app.services.qwen_image_edit_service import (
    generate_frame as generate_qwen,
    status as qwen_status,
)
from app.services.sdxl_generation_service import (
    generate_frame as generate_sdxl,
    status as sdxl_status,
)

GenerateFn = Callable[..., Awaitable[bytes]]
VALIDATOR_URL = (
    "https://download.pytorch.org/models/"
    "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)
PROMPT = (
    "Photorealistic luxury fashion campaign photograph. One real adult fashion "
    "model is clearly visible from shoulders to waist with natural face, skin, "
    "arms, wrists and hands. The model actively carries the exact handbag from "
    "the supplied reference images by its original handle or strap. Show a hand "
    "gripping the handle, physical contact, strap tension, contact shadow and "
    "believable product scale. Preserve the exact bag silhouette, material, "
    "color, weave, hardware, closure and proportions. One person, one bag, no "
    "mannequin, no floating or pasted product, no text, no watermark."
)


def _download_validator() -> None:
    destination = Path(settings.HUMAN_PRODUCT_VALIDATOR_MODEL_PATH).expanduser()
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urlopen(VALIDATOR_URL, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    digest = hashlib.sha256()
    with partial.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != VALIDATOR_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("Downloaded human-product validator checksum failed.")
    partial.replace(destination)


def _references(paths: list[str]) -> tuple[list[str], list[bytes]]:
    normalized: list[str] = []
    content: list[bytes] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Reference image not found: {path}")
        with Image.open(path) as image:
            image.verify()
        normalized.append(str(path))
        content.append(path.read_bytes())
    if not 2 <= len(normalized) <= 6:
        raise ValueError("The audit requires between 2 and 6 product references.")
    return normalized, content


def _blocked_reason(engine: str, status: dict[str, object]) -> str:
    if engine == "fooocus" and status.get("runner_present") is True:
        return (
            "Fooocus has no non-interactive Atelier runner yet; source, runtime, "
            "checkpoint and CUDA alone are not enough to mark it verified."
        )
    return str(
        status.get("reason")
        or "Model files, isolated runtime, runner, or CUDA device are unavailable."
    )


async def _audit_engine(
    engine: str,
    status_fn: Callable[[], dict[str, object]],
    generate: GenerateFn | None,
    reference_paths: list[str],
    reference_bytes: list[bytes],
    output_dir: Path,
) -> dict[str, object]:
    raw = status_fn()
    if raw.get("ready") is not True or generate is None:
        reason = _blocked_reason(engine, raw)
        return write_verification(engine, raw, status="blocked", reason=reason)
    output_path = output_dir / f"{engine}-human-with-bag.png"
    started = time.perf_counter()
    try:
        kwargs: dict[str, object] = {
            "prompt": PROMPT,
            "reference_paths": reference_paths,
            "output_path": str(output_path),
            "seed": 22081996,
        }
        if engine == "sdxl":
            kwargs.update(
                {
                    "reference_index": 1,
                    "strength": settings.SDXL_HUMAN_STRENGTH,
                    "fast": True,
                    "human_context": True,
                }
            )
        generated = await generate(**kwargs)
        validations = validate_human_product(generated, reference_bytes)
        latency_ms = round((time.perf_counter() - started) * 1000)
        passed = validations.get("passed") is True
        reason = str(validations.get("reason") or "Validator returned no reason.")
        current_status = status_fn()
        if current_status.get("ready") is not True:
            passed = False
            reason = str(
                current_status.get("reason")
                or "Runtime became unavailable during the smoke test."
            )
        return write_verification(
            engine,
            current_status,
            status="passed" if passed else "failed",
            reason=reason,
            latency_ms=latency_ms,
            output_path=str(output_path),
            validations=validations,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000)
        return write_verification(
            engine,
            raw,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            latency_ms=latency_ms,
            output_path=str(output_path) if output_path.is_file() else None,
        )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        action="append",
        choices=[
            "all",
            "qwen",
            "flux-schnell",
            "fooocus",
            "hidream",
            "flux2",
            "flux2-klein",
            "sdxl",
        ],
        default=[],
    )
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / ".local" / "generation-validation" / "outputs"),
    )
    parser.add_argument("--download-validator", action="store_true")
    args = parser.parse_args()
    if args.download_validator:
        _download_validator()
    reference_paths, reference_bytes = _references(args.references)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.engine or ["all"])
    matrix: dict[
        str, tuple[Callable[[], dict[str, object]], GenerateFn | None]
    ] = {
        "qwen": (qwen_status, generate_qwen),
        "flux-schnell": (flux_schnell_status, generate_flux_schnell),
        "fooocus": (fooocus_status, None),
        "hidream": (hidream_status, generate_hidream),
        "flux2": (flux2_status, generate_flux2),
        "flux2-klein": (flux2_klein_status, generate_flux2_klein),
        "sdxl": (sdxl_status, generate_sdxl),
    }
    engines = list(matrix) if "all" in selected else [
        engine for engine in matrix if engine in selected
    ]
    results: dict[str, object] = {}
    for engine in engines:
        status_fn, generate = matrix[engine]
        print(f"[{engine}] auditing", flush=True)
        results[engine] = await _audit_engine(
            engine,
            status_fn,
            generate,
            reference_paths,
            reference_bytes,
            output_dir,
        )
        print(
            f"[{engine}] {results[engine].get('status')}: "
            f"{results[engine].get('reason')}",
            flush=True,
        )
    payload = {
        "validator": validator_status(),
        "engines": results,
        "report_path": settings.GENERATION_VERIFICATION_REPORT_PATH,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(
        result.get("status") == "passed" for result in results.values()
    ) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))