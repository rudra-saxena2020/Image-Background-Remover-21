"""CPU-safe editorial frame compositor.

This is intentionally not presented as AI generation. It creates a usable
eight-frame product edit from the locally removed-background reference when a
diffusion GPU is unavailable.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps


CANVAS_SIZE = (1024, 1280)
PALETTES = {
    "studio": ((246, 242, 235), (221, 211, 198)),
    "model": ((226, 214, 204), (157, 133, 123)),
    "editorial": ((235, 229, 220), (189, 174, 157)),
    "detail": ((30, 29, 27), (90, 78, 67)),
    "angle": ((238, 235, 229), (185, 178, 166)),
    "lifestyle": ((220, 226, 219), (143, 160, 143)),
    # The second detail frame intentionally leaves the dark craftsmanship
    # treatment behind: a cool architectural backdrop makes it read as a
    # different campaign image, not a duplicate macro.
    "macro": ((224, 239, 235), (105, 153, 154)),
    "hero": ((226, 216, 201), (109, 82, 69)),
}


def _gradient(size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(height):
        ratio = y / max(height - 1, 1)
        color = tuple(round(top[channel] * (1 - ratio) + bottom[channel] * ratio) for channel in range(3))
        for x in range(width):
            pixels[x, y] = color
    return image


def _load_subject(path: str) -> Image.Image:
    subject = Image.open(path).convert("RGBA")
    alpha = subject.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("Background removal produced an empty product cutout.")
    return subject.crop(bbox)


def _paste_subject(canvas: Image.Image, subject: Image.Image, *, scale: float, center_x: int, bottom: int, angle: float = 0) -> None:
    target_height = round(canvas.height * scale)
    ratio = target_height / max(subject.height, 1)
    resized = subject.resize((max(1, round(subject.width * ratio)), target_height), Image.Resampling.LANCZOS)
    if angle:
        resized = resized.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    x = center_x - resized.width // 2
    y = bottom - resized.height

    alpha = resized.getchannel("A")
    shadow = Image.new("RGBA", resized.size, (15, 12, 10, 0))
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(18)).point(lambda value: round(value * 0.32)))
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_layer.alpha_composite(shadow, (x + 14, y + 22))
    canvas.alpha_composite(shadow_layer)
    canvas.alpha_composite(resized, (x, y))


def _decorate(canvas: Image.Image, kind: str, seed: int) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    rng = random.Random(seed)
    if kind in {"editorial", "lifestyle", "hero"}:
        for offset in range(-300, 1400, 110):
            draw.line((offset, 0, offset + 700, 1280), fill=(255, 255, 255, 24), width=2)
    if kind == "detail":
        for y in range(90, 1280, 48):
            draw.line((0, y, 1024, y), fill=(255, 255, 255, 12), width=1)
        draw.rectangle((44, 72, 980, 1208), outline=(236, 211, 177, 80), width=3)
        draw.line((44, 1020, 980, 740), fill=(236, 211, 177, 90), width=2)
    if kind == "macro":
        for radius in range(120, 720, 120):
            draw.ellipse(
                (512 - radius, 640 - radius, 512 + radius, 640 + radius),
                outline=(255, 255, 255, 82),
                width=2,
            )
        draw.rectangle((64, 96, 960, 1184), outline=(26, 70, 73, 105), width=5)
        draw.line((0, 230, 1024, 1030), fill=(255, 255, 255, 115), width=4)
        draw.line((1024, 260, 0, 1060), fill=(26, 70, 73, 72), width=3)
    if kind == "hero":
        draw.ellipse((120, 90, 900, 720), outline=(255, 245, 230, 55), width=3)
    if kind == "angle":
        draw.rectangle((70, 95, 954, 1185), outline=(255, 255, 255, 50), width=2)
    for _ in range(22):
        x = rng.randrange(0, 1024)
        y = rng.randrange(0, 1280)
        radius = rng.randrange(1, 4)
        draw.ellipse((x, y, x + radius, y + radius), fill=(255, 255, 255, 18))


def generate_composite_frame(*, reference_path: str, shot_kind: str, seed: int, output_path: str) -> bytes:
    subject = _load_subject(reference_path)
    top, bottom = PALETTES.get(shot_kind, PALETTES["studio"])
    canvas = _gradient(CANVAS_SIZE, top, bottom).convert("RGBA")
    _decorate(canvas, shot_kind, seed)

    settings = {
        "studio": (0.72, 512, 1140, 0),
        "model": (0.62, 570, 1110, -3),
        "editorial": (0.68, 435, 1135, 4),
        # Detail and macro are intentionally oversized studies. The product is
        # cropped by the canvas so these are not accidental full-product repeats.
        "detail": (1.62, 720, 1080, -13),
        "angle": (0.78, 610, 1130, -7),
        "lifestyle": (0.58, 410, 1115, 5),
        "macro": (2.05, 275, 940, 17),
        "hero": (0.82, 520, 1145, 0),
    }
    scale, center_x, product_bottom, angle = settings.get(shot_kind, settings["studio"])
    _paste_subject(canvas, subject, scale=scale, center_x=center_x, bottom=product_bottom, angle=angle)

    result = canvas.convert("RGB")
    result.save(output_path, format="PNG", optimize=True)
    buffer = io.BytesIO()
    result.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()