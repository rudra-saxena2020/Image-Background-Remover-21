"""Non-interactive local SDXL image-to-image runner."""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


def load_pipeline(model: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model,
        # Keep the installed fp16 checkpoint on CPU as well; casting the full
        # SDXL pipeline to float32 exceeds the host's memory budget.
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
    else:
        # Attention/VAE slicing saves memory but adds serial work on CPU.
        pipe.disable_attention_slicing()
        pipe.vae.disable_slicing()
    return pipe, device


def generate(pipe, device: str, args) -> None:
    reference = Image.open(args.reference).convert("RGB")
    reference = reference.resize((args.width, args.height), Image.Resampling.LANCZOS)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=(
            "deformed hands, extra fingers, missing fingers, fused fingers, malformed limbs, "
            "duplicate person, duplicate product, floating product, detached strap, warped logo, "
            "melted hardware, unreadable text, invented pockets, wrong material, plastic texture, "
            "generic replacement handbag, different handbag design, different closure, rectangular buckle, "
            "flap bag, zipper, logo plate, extra handles, extra straps, "
            "cropped product, out of frame, low resolution, blurry, cartoon, illustration, CGI, watermark, "
            "letters, captions, typography, labels, signatures, small writing"
        ),
        image=reference,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]
    result.save(args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--reference")
    parser.add_argument("--output")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    pipe, device = load_pipeline(args.model)
    if args.server:
        print("__SDXL_READY__", flush=True)
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            try:
                job = SimpleNamespace(**json.loads(raw_line))
                generate(pipe, device, job)
                print(json.dumps({"ok": True, "output": job.output}), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
        return
    required = ("prompt", "reference", "output", "width", "height", "steps", "guidance", "strength", "seed")
    if any(getattr(args, name) is None for name in required):
        parser.error("single-image mode requires prompt, reference, output, width, height, steps, guidance, strength, and seed")
    generate(pipe, device, args)


if __name__ == "__main__":
    main()