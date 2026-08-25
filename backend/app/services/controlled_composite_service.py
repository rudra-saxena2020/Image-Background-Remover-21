"""Controlled human-scene compositing with an immutable product layer."""

from __future__ import annotations

import io
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageStat


PLACEMENTS: dict[str, tuple[float, float, float, float]] = {
    "model": (0.60, 0.72, 0.34, -2.0),
    "editorial": (0.38, 0.73, 0.27, 4.0),
    "lifestyle": (0.25, 0.76, 0.22, 5.0),
    "hero": (0.66, 0.70, 0.31, -3.0),
}


def _load_cutout(path: str) -> Image.Image:
    cutout = Image.open(path).convert("RGBA")
    if cutout.getchannel("A").getbbox() is None:
        raise ValueError("The immutable product mask is empty.")
    return cutout


def composite_product_layer(
    *,
    plate_bytes: bytes,
    product_cutout_path: str,
    shot_kind: str,
    output_path: str,
    placement: dict[str, float] | None = None,
    occlusion_mask_path: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Place exact masked source pixels over a generated human/scene plate.

    The plate is never accepted as a final frame by this function. Callers must
    run semantic and campaign validation on the returned composite.
    """
    plate = Image.open(io.BytesIO(plate_bytes)).convert("RGBA")
    original_plate = plate.copy()
    cutout = _load_cutout(product_cutout_path)
    defaults = PLACEMENTS.get(shot_kind, PLACEMENTS["model"])
    values = placement or {}
    center_x = float(values.get("center_x", defaults[0]))
    bottom = float(values.get("bottom", defaults[1]))
    height_ratio = float(values.get("height_ratio", defaults[2]))
    angle = float(values.get("angle", defaults[3]))
    target_height = max(1, round(plate.height * height_ratio))
    ratio = target_height / max(cutout.height, 1)
    product = cutout.resize(
        (max(1, round(cutout.width * ratio)), target_height),
        Image.Resampling.LANCZOS,
    )
    if angle:
        product = product.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    x = round(plate.width * center_x - product.width / 2)
    y = round(plate.height * bottom - product.height)

    alpha = product.getchannel("A")
    shadow = Image.new("RGBA", product.size, (20, 14, 12, 0))
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(max(4, product.width // 30))).point(lambda value: round(value * 0.25)))
    shadow_layer = Image.new("RGBA", plate.size, (0, 0, 0, 0))
    shadow_layer.alpha_composite(shadow, (x + max(3, product.width // 45), y + max(5, product.height // 35)))
    plate.alpha_composite(shadow_layer)
    plate.alpha_composite(product, (x, y))
    occlusion_applied = False
    if occlusion_mask_path:
        occlusion_mask = Image.open(occlusion_mask_path).convert("L").resize(
            plate.size, Image.Resampling.BILINEAR
        )
        plate.paste(original_plate, (0, 0), occlusion_mask)
        occlusion_applied = True

    result = plate.convert("RGB")
    buffer = io.BytesIO()
    result.save(buffer, format="PNG", optimize=True)
    output = buffer.getvalue()
    Path(output_path).write_bytes(output)
    metadata = {
        "pipeline": "generated-human-scene + immutable-product-composite",
        "product_source_path": str(Path(product_cutout_path).resolve()),
        "product_source_sha256": hashlib.sha256(Path(product_cutout_path).read_bytes()).hexdigest(),
        "placement": {
            "center_x": center_x,
            "bottom": bottom,
            "height_ratio": height_ratio,
            "angle": angle,
            "box": [x, y, x + product.width, y + product.height],
        },
        "shadow": "soft-local-contact-shadow",
        "occlusion": (
            "foreground-mask-restored"
            if occlusion_applied
            else "product-layer-front; explicit placement required"
        ),
    }
    return output, metadata


def validate_immutable_product_composite(
    image_bytes: bytes,
    *,
    metadata: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Validate the non-negotiable source-layer contract before semantic checks."""
    result: dict[str, Any] = {
        "passed": False,
        "source_layer_applied": False,
        "profile_version": profile.get("version"),
        "reason": "",
    }
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        result["reason"] = f"Composite is not a readable image: {exc}"
        return result
    source_hash = str(metadata.get("product_source_sha256") or "")
    primary = profile.get("primary_view") or {}
    if not source_hash or source_hash != str(primary.get("sha256") or ""):
        result["reason"] = "Composite does not point to the profiled immutable product source."
        return result
    box = (metadata.get("placement") or {}).get("box")
    if not isinstance(box, list) or len(box) != 4:
        result["reason"] = "Composite is missing a controlled product placement box."
        return result
    if box[2] <= box[0] or box[3] <= box[1]:
        result["reason"] = "Composite product placement is empty."
        return result
    placement = metadata.get("placement") or {}
    try:
        expected_image = _load_cutout(str(primary["path"]))
        target_height = max(1, round(image.height * float(placement["height_ratio"])))
        ratio = target_height / max(expected_image.height, 1)
        expected_image = expected_image.resize(
            (max(1, round(expected_image.width * ratio)), target_height),
            Image.Resampling.LANCZOS,
        )
        angle = float(placement.get("angle", 0.0))
        if angle:
            expected_image = expected_image.rotate(
                angle, Image.Resampling.BICUBIC, expand=True
            )
        left, top = round(float(placement["box"][0])), round(float(placement["box"][1]))
        expected_rgb = expected_image.convert("RGB")
        observed_rgb = image.convert("RGB").crop(
            (left, top, left + expected_rgb.width, top + expected_rgb.height)
        )
        difference = ImageChops.difference(expected_rgb, observed_rgb)
        opaque_mask = expected_image.getchannel("A").point(
            lambda value: 255 if value >= 250 else 0
        )
        pixel_error = sum(ImageStat.Stat(difference, mask=opaque_mask).mean) / 3
    except (KeyError, TypeError, ValueError, OSError) as exc:
        result["reason"] = f"Composite source-layer geometry could not be checked: {exc}"
        return result
    result["pixel_identity_score"] = round(max(0.0, 1.0 - pixel_error / 255.0), 5)
    if pixel_error > 2.0:
        result["reason"] = "Visible product pixels differ from the profiled immutable source layer."
        return result
    result["source_layer_applied"] = True
    result["passed"] = True
    result["reason"] = "Immutable profiled product source was composited with controlled placement."
    return result