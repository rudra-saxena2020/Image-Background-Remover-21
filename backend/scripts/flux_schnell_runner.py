"""CUDA-only FLUX.1 Schnell image-to-image runner."""

from __future__ import annotations

import argparse

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FLUX.1 Schnell requires CUDA; refusing CPU generation.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = FluxImg2ImgPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    reference = Image.open(args.reference).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        image=reference,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=generator,
        width=args.width,
        height=args.height,
    ).images[0]
    result.save(args.output, format="PNG")


if __name__ == "__main__":
    main()