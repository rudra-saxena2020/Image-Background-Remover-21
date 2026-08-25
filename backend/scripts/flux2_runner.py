"""Non-interactive adapter around the official FLUX.2 native inference code."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from einops import rearrange
from PIL import ExifTags, Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--references", nargs="+", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    os.environ["FLUX2_MODEL_PATH"] = str(Path(args.model).resolve())
    os.environ["AE_MODEL_PATH"] = str(Path(args.autoencoder).resolve())
    sys.path.insert(0, str(repo / "src"))
    from flux2.sampling import (  # type: ignore[import-not-found]
        batched_prc_img,
        batched_prc_txt,
        denoise,
        encode_image_refs,
        get_schedule,
        scatter_ids,
    )
    from flux2.util import load_ae, load_flow_model, load_text_encoder  # type: ignore[import-not-found]

    model_name = "flux.2-dev"
    device = torch.device("cuda")
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)
    text_encoder = load_text_encoder(model_name, device=device)
    model.eval()
    ae.eval()
    text_encoder.eval()

    images = [Image.open(path).convert("RGB") for path in args.references]
    reference_tokens, reference_ids = encode_image_refs(ae, images)
    context = text_encoder([args.prompt]).to(torch.bfloat16)
    context, context_ids = batched_prc_txt(context)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    noise = torch.randn(
        (1, 128, args.height // 16, args.width // 16),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    image_tokens, image_ids = batched_prc_img(noise)
    timesteps = get_schedule(args.steps, image_tokens.shape[1])
    output = denoise(
        model,
        image_tokens,
        image_ids,
        context,
        context_ids,
        timesteps=timesteps,
        guidance=args.guidance,
        img_cond_seq=reference_tokens,
        img_cond_seq_ids=reference_ids,
    )
    output = torch.cat(scatter_ids(output, image_ids)).squeeze(2)
    output = ae.decode(output).float().clamp(-1, 1)
    output = rearrange(output[0], "c h w -> h w c")
    image = Image.fromarray((127.5 * (output + 1.0)).cpu().byte().numpy())

    if text_encoder.test_image(image):
        raise RuntimeError("FLUX.2 output was rejected by the official content filter.")
    exif = Image.Exif()
    exif[ExifTags.Base.Software] = "AI generated;flux2"
    exif[ExifTags.Base.Make] = "Black Forest Labs"
    image.save(args.output, exif=exif, quality=95, subsampling=0)


if __name__ == "__main__":
    main()