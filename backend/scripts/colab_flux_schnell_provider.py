"""Small JSON-line provider adapter for the Atelier Colab worker.

The worker launches this process for startup probing and generation. It loads
one FLUX.1 Schnell pipeline at a time, uses CPU offload for T4-class GPUs, and
never returns success without a readable output image.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _load_pipeline(model_path: str):
    import torch
    from diffusers import FluxImg2ImgPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("NO_GPU: FLUX Schnell requires CUDA.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipeline = FluxImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload()
    return pipeline


def _run(request: dict[str, object]) -> dict[str, object]:
    model_path = os.environ.get("FLUX_SCHNELL_MODEL_PATH", "").strip()
    if not model_path or not Path(model_path).exists():
        raise RuntimeError("MODEL_LOAD_ERROR: set FLUX_SCHNELL_MODEL_PATH to downloaded FLUX.1 Schnell weights.")
    pipeline = _load_pipeline(model_path)
    action = request.get("action")
    if action == "probe":
        from PIL import Image
        import torch

        output = Path(str(request.get("output") or "/tmp/atelier-flux-probe.png"))
        output.parent.mkdir(parents=True, exist_ok=True)
        image = pipeline(
            prompt="A professional studio photograph of a leather handbag on a white background.",
            num_inference_steps=1,
            guidance_scale=0.0,
            width=512,
            height=512,
            generator=torch.Generator(device="cuda").manual_seed(1),
        ).images[0]
        image.save(output, format="PNG")
        return {"model_loaded": True, "inference_passed": output.is_file(), "output": str(output)}
    if action != "generate":
        raise RuntimeError("PROVIDER_REQUEST_INVALID: expected probe or generate.")
    from PIL import Image
    import torch

    references = request.get("references") or []
    if not isinstance(references, list) or not references:
        raise RuntimeError("PROVIDER_REQUEST_INVALID: at least one reference is required.")
    reference = Image.open(str(references[0])).convert("RGB")
    width = int(os.environ.get("FLUX_SCHNELL_WIDTH", "768"))
    height = int(os.environ.get("FLUX_SCHNELL_HEIGHT", "1024"))
    output = Path(str(request.get("output") or "/tmp/atelier-flux-output.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    image = pipeline(
        prompt=str(request.get("prompt") or ""),
        image=reference.resize((width, height), Image.Resampling.LANCZOS),
        strength=float(os.environ.get("FLUX_SCHNELL_STRENGTH", "0.55")),
        num_inference_steps=int(os.environ.get("FLUX_SCHNELL_STEPS", "4")),
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(int(request.get("seed") or 0)),
        width=width,
        height=height,
    ).images[0]
    image.save(output, format="PNG")
    if not output.is_file():
        raise RuntimeError("PROVIDER_OUTPUT_INVALID: pipeline produced no image.")
    return {"model_loaded": True, "inference_passed": True, "output": str(output)}


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        print(json.dumps(_run(request)), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"inference_passed": False, "reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())