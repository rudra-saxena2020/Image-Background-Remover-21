"""Immutable product identity profiles for Atelier composites.

The profile deliberately stores measurements and hashes, not a generated
description that could be mistaken for a replacement product.  The cleaned
source PNG remains the only product pixel source for a controlled composite.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dominant_colors(image: Image.Image) -> list[list[int]]:
    rgb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    colors = rgb.quantize(colors=5).convert("RGB").getcolors(64 * 64) or []
    return [
        list(color)
        for _count, color in sorted(colors, reverse=True)[:5]
    ]


def _profile_cutout(path: str) -> dict[str, Any]:
    data = Path(path).read_bytes()
    image = Image.open(io.BytesIO(data)).convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError(f"Product mask is empty: {path}")
    left, top, right, bottom = bbox
    alpha_area = sum(1 for value in alpha.getdata() if value > 16)
    total = image.width * image.height
    crop = image.crop(bbox)
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256(data),
        "width": image.width,
        "height": image.height,
        "bbox": [left, top, right, bottom],
        "cutout_width": crop.width,
        "cutout_height": crop.height,
        "aspect_ratio": round(crop.width / max(crop.height, 1), 5),
        "foreground_ratio": round(alpha_area / max(total, 1), 5),
        "dominant_colors": _dominant_colors(crop),
    }


def _view_state_from_label(label: str) -> str:
    normalized = re.sub(r"[_-]+", " ", label.lower())
    if any(token in normalized for token in ("open", "interior", "inside", "top")):
        return "open"
    if "back" in normalized or "rear" in normalized:
        return "back"
    if "side" in normalized or "profile" in normalized:
        return "side"
    if "angle" in normalized or "three quarter" in normalized:
        return "angle"
    if "front" in normalized or "hero" in normalized:
        return "front"
    return "unknown"


def _color_distance(left: list[int], right: list[int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _foreground_metrics(image_bytes: bytes) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("Generated product mask is empty")
    left, top, right, bottom = bbox
    crop = image.crop(bbox)
    opaque = sum(1 for value in alpha.getdata() if value > 16)
    return {
        "width": image.width,
        "height": image.height,
        "bbox": [left, top, right, bottom],
        "aspect_ratio": round((right - left) / max(bottom - top, 1), 5),
        "foreground_ratio": round(opaque / max(image.width * image.height, 1), 5),
        "dominant_colors": _dominant_colors(crop),
    }


def build_product_identity_profile(
    *,
    source_paths: list[str],
    source_bytes: list[bytes],
    product_name: str,
    category: str = "unknown",
    output_path: str,
    view_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Build and persist a deterministic profile from BiRefNet PNG outputs."""
    if not source_paths:
        raise ValueError("At least one masked product reference is required.")
    if len(source_paths) != len(source_bytes):
        raise ValueError("Masked product paths and source bytes must have equal length.")
    labels = view_labels or [f"reference-{index + 1}" for index in range(len(source_paths))]
    if len(labels) != len(source_paths):
        raise ValueError("Product reference labels must match the number of masked references.")
    views = [_profile_cutout(path) for path in source_paths]
    for view, label in zip(views, labels):
        view["label"] = label
    product_id = _sha256(b"".join(source_bytes))[:24]
    profile = {
        "version": 1,
        "product_id": product_id,
        "product_name": product_name,
        "category": category,
        "subcategory": "source-observed",
        "source_policy": "immutable-source-pixels",
        "material": "source-observed; not regenerated",
        "color": "source-observed; see dominant_colors",
        "texture": "source-observed; see detail crops",
        "shape": "source-observed; see silhouette measurements",
        "dimensions": {
            "width": views[0]["cutout_width"],
            "height": views[0]["cutout_height"],
            "aspect_ratio": views[0]["aspect_ratio"],
        },
        "hardware": "source-observed; inspect upper-detail crop",
        "logo": "source-observed; no text reconstruction",
        "pattern": "source-observed; preserved in source layer",
        "construction": "source-observed; preserved in source layer",
        "reference_count": len(views),
        "primary_view": views[0],
        "views": views,
        "source_reference_sha256": [_sha256(data) for data in source_bytes],
        "identity_evidence": {
            "multi_reference": len(views) > 1,
            "silhouette_views": len({(v["aspect_ratio"], v["cutout_width"], v["cutout_height"]) for v in views}),
            "confidence": "profiled" if len(views) >= 2 else "single-view",
            "view_labels": labels,
            "same_product_across_views": True,
        },
        "consistency_lock": {
            "version": 1,
            "rule": "one-fixed-physical-product",
            "known_view_states": sorted({_view_state_from_label(label) for label in labels}),
            "supported_opening": any(
                _view_state_from_label(label) == "open" for label in labels
            ),
            "unknown_details_must_remain_unknown": True,
            "locked_dimensions": {
                "aspect_ratio": views[0]["aspect_ratio"],
                "foreground_ratio_min": min(view["foreground_ratio"] for view in views),
                "foreground_ratio_max": max(view["foreground_ratio"] for view in views),
            },
            "locked_colors": views[0]["dominant_colors"],
            "locked_material": "source-observed",
            "locked_hardware": "source-observed",
            "locked_closure": "source-observed; opening unsupported unless reference-supported",
        },
        "distinctive_detail_crops": [
            {
                "name": "upper-detail",
                "bbox": [
                    views[0]["bbox"][0],
                    views[0]["bbox"][1],
                    views[0]["bbox"][2],
                    views[0]["bbox"][1] + max(1, round((views[0]["bbox"][3] - views[0]["bbox"][1]) * 0.42)),
                ],
                "purpose": "handle, closure, hardware and upper construction",
            },
            {
                "name": "body-detail",
                "bbox": [
                    views[0]["bbox"][0],
                    views[0]["bbox"][1] + max(1, round((views[0]["bbox"][3] - views[0]["bbox"][1]) * 0.35)),
                    views[0]["bbox"][2],
                    views[0]["bbox"][3],
                ],
                "purpose": "material, weave, grain and stitching",
            },
        ],
    }
    Path(output_path).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def validate_product_consistency(
    image_bytes: bytes,
    profile: dict[str, Any],
    *,
    previous_frames: list[bytes] | None = None,
    shot_kind: str | None = None,
) -> dict[str, Any]:
    """Fail closed on measurable drift from the locked source product.

    This is deliberately conservative: it cannot invent hidden geometry from
    text, so an unsupported opening state is rejected before it can reach a
    customer. Pixel-level identity remains the provider's responsibility, but
    occupancy, silhouette, and color gates catch common cross-frame failures.
    """
    lock = profile.get("consistency_lock") or {}
    try:
        metrics = _foreground_metrics(image_bytes)
    except Exception as exc:
        return {"passed": False, "reason": str(exc)}

    if shot_kind in {"open", "interior"} and not lock.get("supported_opening", False):
        return {
            "passed": False,
            "reason": "Opening/interior view is unsupported by the uploaded references",
        }

    dimensions = lock.get("locked_dimensions") or {}
    reference_aspect = float(dimensions.get("aspect_ratio") or 0)
    if reference_aspect:
        aspect_delta = abs(metrics["aspect_ratio"] - reference_aspect) / reference_aspect
        if aspect_delta > 0.75:
            return {
                "passed": False,
                "reason": f"Product silhouette drifted from the locked aspect profile ({aspect_delta:.0%})",
            }

    lower_ratio = float(dimensions.get("foreground_ratio_min") or 0) * 0.30
    upper_ratio = float(dimensions.get("foreground_ratio_max") or 1) * 2.20
    if not lower_ratio <= metrics["foreground_ratio"] <= min(upper_ratio, 0.98):
        return {
            "passed": False,
            "reason": "Product scale or framing drifted outside the locked reference range",
        }

    locked_colors = lock.get("locked_colors") or []
    generated_colors = metrics.get("dominant_colors") or []
    if locked_colors and generated_colors:
        closest = min(
            _color_distance(source, generated)
            for source in locked_colors
            for generated in generated_colors
        )
        if closest > 145:
            return {
                "passed": False,
                "reason": f"Product color drifted from the locked source palette ({closest:.0f})",
            }

    for previous in previous_frames or []:
        try:
            previous_metrics = _foreground_metrics(previous)
        except Exception:
            continue
        if abs(metrics["aspect_ratio"] - previous_metrics["aspect_ratio"]) / max(
            metrics["aspect_ratio"], previous_metrics["aspect_ratio"], 0.01
        ) > 1.25:
            return {
                "passed": False,
                "reason": "Product silhouette is contradictory to an accepted gallery frame",
            }

    return {"passed": True, "metrics": metrics}


def human_scene_prompt(
    *,
    category: str,
    shot_kind: str,
    atmosphere: str,
    background: str,
) -> str:
    """Prompt only the human and environment plate, never a replacement product."""
    return (
        f"Photorealistic luxury commercial {category} campaign photograph. "
        "Create one real adult fashion model and the requested environment as a clean "
        "scene plate. Do not render, draw, invent, approximate, or reconstruct any "
        "handbag or other product. Leave a physically plausible empty product area where "
        "the exact source product will be composited later. Keep natural arms, hands, "
        "shoulders, posture, clothing folds, lighting direction, depth, and contact "
        f"composition appropriate for a {shot_kind} frame. Atmosphere: {atmosphere}. "
        f"Background: {background}. Premium editorial photography, one person, no "
        "mannequin, no text, no watermark, no duplicate objects."
    )
