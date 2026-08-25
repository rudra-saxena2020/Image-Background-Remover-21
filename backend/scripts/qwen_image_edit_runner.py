"""CUDA-only Qwen Image Edit 2511 runner used by the API control plane."""

from __future__ import annotations

import argparse

import torch
from diffusers import QwenImageEditPipeline
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--references", nargs="+", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen Image Edit requires CUDA; refusing CPU generation.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = QwenImageEditPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)
    images = [
        Image.open(reference).convert("RGB")
        for reference in args.references[:6]
    ]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    result = pipe(
        image=images,
        prompt=args.prompt,
        negative_prompt=(
            "text, watermark, extra product, redesigned product, duplicate product, "
            "deformed anatomy, mannequin, floating object, blurry product"
        ),
        true_cfg_scale=args.guidance,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        generator=generator,
    ).images[0]
    result.save(args.output, format="PNG")


if __name__ == "__main__":
    main()