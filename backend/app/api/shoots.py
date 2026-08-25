"""Luxury product photography shoot orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image, ImageChops, ImageStat

from app.config import settings
from app.services.image_service import process_image, validate_image
from app.services.inference_queue import get_semaphore
from app.services.human_product_validation_service import validate_human_product
from app.services.cpu_composite_service import generate_composite_frame
from app.services.controlled_composite_service import (
    composite_product_layer,
    validate_immutable_product_composite,
)
from app.services.product_identity_service import (
    build_product_identity_profile,
    human_scene_prompt,
    validate_product_consistency,
)
from app.services.remote_worker_service import (
    generate_frame as generate_remote_frame,
    status as remote_worker_status,
)
from app.services.fal_flux2_pro_service import (
    FalFlux2ProError,
    generate_frame as generate_fal_flux2_pro_frame,
    status as fal_flux2_pro_status,
)
from app.services.black_forest_flux2_service import (
    BlackForestFlux2Error,
    cancel_frame as cancel_black_forest_frame,
    generate_frame as generate_black_forest_flux2_frame,
    status as black_forest_flux2_status,
)
from app.services.runpod_qwen_image_edit_service import (
    generate_frame as generate_runpod_qwen_frame,
    status as runpod_qwen_status,
)
from app.services.runpod_flux1_dev_service import (
    RunPodFlux1DevError,
    generate_frame as generate_runpod_flux1_dev_frame,
    status as runpod_flux1_dev_status,
)
from app.services.gemini_image_service import (
    GeminiImageError,
    generate_frame as generate_gemini_image_frame,
    status as gemini_image_status,
)
from app.services.shoot_history_service import (
    load_frame,
    load_shoots,
    save_frame,
    save_shoot,
)
from app.services.shopify_import_service import (
    ShopifyImportError,
    list_products as list_shopify_products,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_REFERENCES = 6
MAX_ACTIVE_SHOOTS = 10
ENGINE_MODES = {
    "rudras",
    "auto",
    "qwen",
    "qwen-runpod",
    "flux1-runpod",
    "gemini-image",
    "flux-schnell",
    "fooocus",
    "hidream",
    "flux2",
    "flux2-pro",
    "bfl-flux2",
    "flux2-klein",
    "sdxl",
    "colab",
    "cpu",
}
REMOTE_MODEL_BY_SELECTION = {
    "qwen": "qwen-edit",
    "flux-schnell": "flux-schnell",
    "fooocus": "fooocus",
    "hidream": "hidream",
    "flux2": "flux2",
    "flux2-klein": "flux2-klein",
    "sdxl": "sdxl",
}
CAMPAIGN_FRAME_COUNT = 8
CAMPAIGN_MAX_ATTEMPTS = 3
MAX_FRAME_SIMILARITY = 0.93
HUMAN_MODEL_CATEGORIES = {"bags", "wallets", "watches", "jewelry", "shoes", "clothing"}
HUMAN_MODEL_SHOTS = {"model", "model-angle", "editorial", "lifestyle", "hero"}
CAMPAIGN_FORMAT_FLEXIBLE = "flexible-8"
CAMPAIGN_FORMAT_FRONT_BACK = "front-back-7"
CAMPAIGN_FORMAT_COMPACT = "compact-6"
CAMPAIGN_FORMATS = {
    CAMPAIGN_FORMAT_FLEXIBLE,
    CAMPAIGN_FORMAT_FRONT_BACK,
    CAMPAIGN_FORMAT_COMPACT,
}
GENERATION_MODE_PRODUCT_ONLY = "product-only"
GENERATION_MODE_HUMAN_MODEL = "human-model"
GENERATION_MODES = {GENERATION_MODE_PRODUCT_ONLY, GENERATION_MODE_HUMAN_MODEL}

STARTER_PRODUCT_IDENTITY = {
    "p01187465": (
        "Reference-specific identity lock: this is a wide, soft rectangular burgundy woven-leather tote with a "
        "single rounded dark-brown top handle, two circular side eyelets, a dark braided cord threaded across the "
        "upper front, and a small gold infinity-shaped knot clasp at its center. Preserve that exact construction. "
        "Do not replace it with a generic handbag, a flap bag, a rectangular buckle, a zipper, a monogram, a logo "
        "plate, extra pockets, extra handles or a different closure."
    ),
}

SOURCE_IMAGE_ENFORCEMENT = (
    "SOURCE IMAGE ENFORCEMENT — HIGHEST PRIORITY: the uploaded product references are the only source of truth. "
    "Generate this scene directly from the supplied reference images using image-to-image conditioning; never "
    "recreate, redesign, approximate or invent the product from text alone. Analyze every reference first and lock "
    "the exact shape, dimensions, material, texture, leather grain, fabric, color, pattern, logo placement, hardware, "
    "stitching, handles, straps, buckles, zips, interior and proportions. Preserve the product immutably. Only change "
    "the background, lighting, camera, composition, environment, pose and human model. If the product differs from the "
    "references in any way, reject the frame and regenerate."
)

MODEL_CONSISTENCY_AND_REALISM_LOCK = (
    "ABSOLUTE HUMAN MODEL CONSISTENCY AND REALISM LOCK — when a human model is included, treat the model as a "
    "second fixed identity, completely separate from the product identity. A dedicated uploaded model reference, or "
    "the first accepted model anchor from this campaign, is the absolute identity source. If multiple references show "
    "the same approved model, use them together as evidence of one person. Preserve the same face, facial proportions, "
    "eyes, nose, lips, jawline, skin tone and texture, hair color, length, style and hairline, body proportions, height "
    "appearance, body shape, age appearance and distinguishing features in every frame. Never generate a similar-looking "
    "person, a new model, or a different ethnicity, facial structure, hairstyle, age appearance, body proportion or "
    "skin tone. The model reference defines WHO; the product references define WHAT. Never merge, replace or confuse "
    "these identities. Unless a different outfit is explicitly requested, keep the exact same outfit, garment design, "
    "clothing colors, material, accessories, shoes, styling and makeup across the set; if a different outfit is "
    "requested, change only the clothing while preserving both identities. "
    "REAL HUMAN PHOTOGRAPHY LOCK — the result must look like a real commercial fashion photograph, with natural pores, "
    "subtle skin variation, realistic asymmetry, authentic hair strands, natural body proportions, physically correct "
    "hands and fingers, realistic fabric interaction, natural shadows, authentic camera optics and believable depth. "
    "Reject plastic, wax-like, over-smoothed or AI-beauty skin, CGI or cartoon features, doll-like symmetry, artificial "
    "body proportions, extra or missing fingers, fused hands, broken anatomy, impossible joints, crossed eyes, warped "
    "limbs or floating contact. "
    "MODEL AND PRODUCT SCALE LOCK — preserve the product's true apparent size and believable scale relative to the "
    "same model in every frame. Do not make the product larger, smaller, wider, taller, thicker or thinner to fit the "
    "composition. The model may change pose, camera, framing, background or explicitly requested environment, but the "
    "person must remain the same person and the product the same physical object. "
    "FINAL TWO-IDENTITY VALIDATION — before accepting a frame, verify the same model face, hair, skin tone, body "
    "proportions and distinguishing features AND the same product shape, color distribution, proportions, material, "
    "straps, hardware, chains, construction, graphics and pattern. If either identity changes, reject and regenerate. "
)


def _engine_unavailable_detail(engine: str, generation: dict[str, object]) -> str:
    if engine == "flux1-runpod":
        return str(
            generation.get("reason")
            or "The RunPod FLUX.1 Dev endpoint is unavailable."
        )
    if engine == "qwen-runpod":
        return str(
            generation.get("reason")
            or "The RunPod Qwen Image Edit endpoint is unavailable."
        )
    if engine == "gemini-image":
        return str(
            generation.get("reason")
            or "The Gemini image generation integration is unavailable."
        )
    if engine == "colab":
        return str(
            generation.get("reason")
            or "Authenticated Colab worker is unavailable or not freshly verified."
        )
    if engine == "fal":
        return str(
            generation.get("reason")
            or "The connected fal.ai FLUX.2 Pro provider is unavailable."
        )
    if engine == "bfl":
        return str(
            generation.get("reason")
            or "The server-side Black Forest FLUX.2 provider is unavailable."
        )
    return "Unsupported generation engine."


def _resolve_generation_backend(requested_engine: str) -> tuple[str, str | None]:
    """Resolve the requested provider without silently substituting a model.

    Open-source choices remain Colab-only; the paid FLUX.2 Pro endpoint is a
    separate authenticated fal.ai route. CPU is the only local preview path.
    """
    if requested_engine == "cpu":
        return "cpu", None
    if requested_engine == "flux2-pro":
        return "fal", "fal-ai/flux-2-pro/edit"
    if requested_engine == "bfl-flux2":
        return "bfl", "flux-2-pro"
    if requested_engine == "qwen-runpod":
        return "qwen-runpod", "qwen-image-edit-2511"
    if requested_engine == "flux1-runpod":
        return "flux1-runpod", "black-forest-labs-flux-1-dev"
    if requested_engine == "gemini-image":
        return "gemini-image", "gemini-2.5-flash-image"
    if requested_engine in {"rudras", "auto", "colab"}:
        return "colab", "auto"
    return "colab", REMOTE_MODEL_BY_SELECTION[requested_engine]


def _remote_model_matches_status(
    requested_model: str | None,
    worker: dict[str, object],
) -> bool:
    if requested_model in {None, "", "auto"}:
        return True
    active_provider = str(worker.get("provider") or "").strip().lower()
    requested = str(requested_model).strip().lower()
    return active_provider == requested


def _current_verified_engine_status(
    engine: str,
    remote_model: str | None = None,
) -> dict[str, object]:
    if engine == "colab":
        current = remote_worker_status()
        if current.get("ready") is True and not _remote_model_matches_status(
            remote_model, current
        ):
            return {
                **current,
                "ready": False,
                "reason": (
                    f"Requested Colab model {remote_model!r} is not the worker's "
                    f"verified provider {current.get('provider')!r}."
                ),
            }
        return current
    if engine == "fal":
        return fal_flux2_pro_status()
    if engine == "bfl":
        return black_forest_flux2_status()
    if engine == "qwen-runpod":
        return runpod_qwen_status()
    if engine == "flux1-runpod":
        return runpod_flux1_dev_status()
    if engine == "gemini-image":
        return gemini_image_status()
    return {
        "id": engine,
        "name": engine,
        "ready": engine == "cpu",
        "runtime_ready": engine == "cpu",
        "reason": None if engine == "cpu" else "Unsupported generation engine.",
    }


def _assert_engine_still_verified(
    engine: str,
    remote_model: str | None = None,
) -> None:
    if engine == "cpu":
        return
    current = _current_verified_engine_status(engine, remote_model)
    if current.get("ready") is not True:
        raise RuntimeError(
            "Generation backend verification is no longer current: "
            + _engine_unavailable_detail(engine, current)
        )

PRODUCT_CATEGORY_RULES = {
    "bags": (
        "Product-aware direction: luxury bag. Use a fashion model only when the frame calls for one, and keep the "
        "bag naturally on the shoulder, in the hand, on the forearm or crossbody according to its construction. "
        "Never float the bag or place it unnaturally. The bag must sit at a physically believable scale."
    ),
    "wallets": (
        "Product-aware direction: wallet or small leather good. Prefer hand-model usage over a full-body model. "
        "Show believable holding, opening, pocket placement or payment gestures only when appropriate; preserve the "
        "wallet's exact scale and construction."
    ),
    "watches": (
        "Product-aware direction: watch. Use a wrist or hand model rather than a full fashion model unless explicitly "
        "requested. Keep the watch size accurate and anatomically grounded in scenes such as checking time, a coffee "
        "table or a luxury office. Hands and wrist anatomy must be perfect."
    ),
    "jewelry": (
        "Product-aware direction: jewelry. Use the smallest useful model context: hand for rings, wrist for bracelets, "
        "upper body for necklaces and close portrait/ear framing for earrings. Keep fingers, ears, neck and skin "
        "anatomically natural; use luxury macro and beauty-campaign lighting."
    ),
    "shoes": (
        "Product-aware direction: footwear. Use a standing or walking model where useful, with side, rear, sole and "
        "close-up views. Never crop the shoe in a product frame; preserve fit, silhouette, sole and proportions."
    ),
    "clothing": (
        "Product-aware direction: clothing. Use a full-body model when modeling is useful and show front, side, rear, "
        "walking and fabric-detail perspectives. Preserve garment fit, drape, seams, texture and proportions; never "
        "let the styling hide the product."
    ),
    "fragrance-beauty": (
        "Product-aware direction: fragrance or beauty product. No model is required for the primary image. Favor "
        "luxury studio, marble, stone, glass, water, vanity or countertop environments with controlled reflections and "
        "a clear bottle/package silhouette."
    ),
    "furniture": (
        "Product-aware direction: furniture or home decor. Never use a fashion model. Build a believable luxury "
        "interior with living-room, bedroom, office or styled-environment perspectives. Preserve scale, geometry, "
        "materials and construction."
    ),
    "electronics": (
        "Product-aware direction: electronics. Favor a precise studio hero, real-world desk or lifestyle context, "
        "detail views of ports and controls, and accurate reflections. Preserve dimensions, controls, screens, logos "
        "and materials with no invented features."
    ),
    "accessories": (
        "Product-aware direction: fashion accessory. Infer the most natural use context from the reference and category, "
        "then choose a believable hand, wrist, model or still-life treatment. Do not force a generic handbag pose."
    ),
}

CATEGORY_KEYWORDS = {
    "watches": ("watch", "wristwatch", "timepiece", "chronograph"),
    "jewelry": ("jewelry", "jewellery", "ring", "bracelet", "necklace", "earring", "diamond", "gold"),
    "shoes": ("shoe", "sneaker", "boot", "boots", "sandal", "loafer", "heel", "footwear"),
    "clothing": ("jacket", "hoodie", "shirt", "dress", "jean", "trouser", "coat", "skirt", "clothing", "apparel"),
    "fragrance-beauty": (
        "perfume",
        "parfum",
        "fragrance",
        "cologne",
        "beauty",
        "skincare",
        "cosmetic",
        "lipstick",
        "serum",
    ),
    "furniture": ("furniture", "sofa", "chair", "table", "desk", "cabinet", "interior", "home decor"),
    "electronics": ("phone", "laptop", "camera", "headphone", "earbud", "tablet", "electronic", "speaker"),
    "wallets": ("wallet", "cardholder", "purse", "small leather"),
    "bags": (
        "bag",
        "handbag",
        "tote",
        "shoulder",
        "crossbody",
        "clutch",
        "backpack",
        "briefcase",
        "luggage",
        "leather goods",
    ),
}

SHOT_PLAN = [
    ("studio", "Luxury Studio Product", "Shopify primary image"),
    ("model", "Model Carrying Product", "Product in natural use"),
    ("editorial", "Editorial Campaign", "Movement and fashion story"),
    ("detail", "Craftsmanship Macro", "Leather, hardware and texture"),
    ("angle", "Alternative Perspective", "Rear, side or 45° product view"),
    ("lifestyle", "Lifestyle Image", "Natural movement in a luxury setting"),
    ("macro", "Luxury Detail", "Handle, logo, interior or hardware"),
    ("hero", "Hero Campaign", "Homepage banner and brand story"),
]

FRONT_BACK_SHOT_PLAN = [
    ("studio", "Clean Product Hero", "Front-facing primary product image"),
    ("model", "Human Model Hero", "Exact product in natural use"),
    ("model-angle", "Model · Different Angle", "Same model and product from a new angle"),
    ("detail", "Product Detail", "Craftsmanship, hardware and texture"),
    ("angle", "Alternative Product Angle", "Back, side or 45° product perspective"),
    ("lifestyle", "Same-Model Lifestyle", "Natural movement in a luxury setting"),
    ("hero", "Same-Model Editorial", "Editorial campaign image with the locked model"),
]

COMPACT_SHOT_PLAN = [
    ("studio", "Clean Product Hero", "Front-facing primary product image"),
    ("angle", "Side / Rear Product View", "Exact side or rear product perspective"),
    ("model", "Model Product Portrait", "Same campaign model wearing or carrying the product"),
    ("model-angle", "Second Model Angle", "Same model and product from a new perspective"),
    ("detail", "Craftsmanship Detail", "Material, stitching, logo or hardware detail"),
    ("lifestyle", "Model Lifestyle Frame", "Same model in a natural premium setting"),
]

CAMPAIGN_PLANS = {
    CAMPAIGN_FORMAT_FLEXIBLE: SHOT_PLAN,
    CAMPAIGN_FORMAT_FRONT_BACK: FRONT_BACK_SHOT_PLAN,
    CAMPAIGN_FORMAT_COMPACT: COMPACT_SHOT_PLAN,
}

CAMPAIGN_FRAME_DIRECTIONS = {
    "studio": (
        "No model. Pure white seamless studio. Front-facing, centered, entire product visible with generous "
        "breathing room. This is the Shopify primary image: clean, neutral, accurate and commercially legible."
    ),
    "model": (
        "Use the one campaign model identity established for this product. Medium half-body shot, standing naturally "
        "with the product carried on the shoulder or in hand. Refined neutral styling, soft luxury studio light, "
        "product fully visible and clearly the subject."
    ),
    "model-angle": (
        "Use the same campaign model identity as the human model hero, with the exact product visibly "
        "used in a different three-quarter or side angle. Change the camera height, body orientation and "
        "hand placement while keeping the model's face, hair, styling and proportions identical."
    ),
    "editorial": (
        "Use the same campaign model identity. The model is walking through a luxury architectural interior, captured "
        "from the side with a distinct stride and body orientation. Magazine advertisement composition, different crop "
        "and lighting from every other frame; never another static standing pose."
    ),
    "detail": (
        "No model, hands or props. Show a supported closed-product craftsmanship detail: authentic material grain, "
        "stitching, hardware and construction only when visible in the references. Do not open the product or invent "
        "an interior, pocket, zipper path or hidden detail."
    ),
    "angle": (
        "No model, hands or props. Show the exact product from a supported rear, side or three-quarter closed-product "
        "perspective. Use a distinct camera position without inventing an opening, interior, pocket or hidden construction."
    ),
    "lifestyle": (
        "Use the same campaign model identity in natural movement, walking through a luxury shopping district, hotel "
        "or modern architectural setting. Wide environmental composition, a different pose and distance from the model "
        "frame, with natural shadow and the product fully readable."
    ),
    "macro": (
        "No model. A second luxury detail story, not a repeat of the craftsmanship frame. Focus tightly on the handle, "
        "logo, interior edge, zipper or distinctive hardware. Change the angle, crop and lighting from the other macro "
        "frame. Never show a generic full-product image."
    ),
    "hero": (
        "Use the same campaign model identity. Wide editorial hero composition in premium architecture, with a bold "
        "asymmetrical subject placement and clear negative space suitable for a homepage banner. Natural fashion pose, "
        "cinematic but physically accurate light, unmistakably a different image from every previous frame."
    ),
}

PRODUCT_ONLY_FRAME_DIRECTIONS = {
    **CAMPAIGN_FRAME_DIRECTIONS,
    "model": (
        "No model or hands. Show the exact product in a clean three-quarter catalog view, centered and fully visible "
        "with standardized margins, natural lens compression and no environmental props."
    ),
    "model-angle": (
        "No model or hands. Show the exact product from a distinct side or three-quarter catalog angle, fully visible "
        "and undistorted with consistent scale and margins."
    ),
    "editorial": (
        "No model, hands or environment. Create a refined alternate catalog composition with a controlled camera "
        "angle and premium studio lighting; the exact product remains centered, complete and sharp."
    ),
    "lifestyle": (
        "No model, hands, props or environment. Use a clean product-only catalog perspective distinct from the hero, "
        "with accurate geometry, standardized framing and soft controlled light."
    ),
    "hero": (
        "No model or environment. Create a premium product-only hero with centered composition, negative space, "
        "complete product visibility, accurate scale and controlled studio lighting."
    ),
}

CAMPAIGN_POLICY = (
    "Create a complete luxury fashion campaign, not eight minor variations of one product shot. Every frame must "
    "have a unique purpose, camera angle, composition, subject placement, pose, lighting, background, crop and story. "
    "The uploaded references are the single source of truth. Preserve the exact product shape, dimensions, color, "
    "material, grain, fabric, stitching, pattern, logo, embossing, hardware, buckles, handles, strap length, zippers, "
    "interior and proportions. No redesigns, hallucinated features, missing details, warped product or extra products. "
    "Use exactly one consistent human model identity for this product across all lifestyle frames; never change the "
    "face, eyes, hair, skin tone, body proportions, age appearance, outfit, accessories or makeup. Only vary pose, "
    "camera, direction, hands, expression, body orientation, background and lighting. Make the model look photographed "
    "by a professional fashion photographer: natural pores, hair strands, clothing wrinkles, eye reflections, "
    "realistic hands and anatomy, correct shadows, reflections, depth of field and lens compression. Reject and "
    "regenerate plastic skin, distorted faces, crossed eyes, bad teeth, extra or fused fingers, broken wrists, twisted "
    "limbs, floating objects, warped leather, wrong logos, melted fabric, strange reflections, blur or low-resolution "
    "product detail. Never repeat a camera position, pose, background, composition, crop, lighting setup or image purpose."
)

PRODUCT_ONLY_MASTER_PROMPT = (
    "PRODUCT-ONLY CATALOG MODE: generate no human, model, hands, props, packaging or environmental scene. "
    "The uploaded product reference is the single source of truth. Preserve the exact product identity, silhouette, "
    "proportions, handles, straps, pockets, zippers, seams, stitching, hardware, logos, patterns, color, material "
    "texture and construction. Never redesign, beautify, simplify or invent unseen details. Use controlled luxury "
    "e-commerce catalog photography comparable to premium Coach product listings: front hero, three-quarter, side "
    "or rear view, and detail views only when supported by the references. Keep every full product centered, fully "
    "visible, undistorted and consistently scaled with generous margins. Use soft controlled studio lighting, accurate "
    "color and realistic material detail. No extreme perspective, fisheye distortion, stretched geometry, cropped "
    "handles, extra products, text or watermark. Do not fabricate an interior or hidden construction."
)

PRODUCT_ONLY_CAMPAIGN_POLICY = (
    "Create a complete luxury product catalog sequence, not minor variations of one shot. Every frame must have a "
    "unique purpose, camera angle, composition, scale, crop and product detail. The uploaded references are the single "
    "source of truth. Preserve the exact product shape, dimensions, color, material, grain, fabric, stitching, pattern, "
    "logo, embossing, hardware, buckles, handles, strap length, zippers, interior and proportions. Treat all supplied "
    "references as views of one fixed physical object. Lock the product's geometry, material, color, hardware, branding, "
    "closure, camera scale, background tone, white balance and shadow treatment across the gallery. Reject redesigns, "
    "hallucinated features, missing details, warped geometry, extra products, text, watermark, color drift, scale drift, "
    "contradictory silhouettes or physically impossible openings."
)


@dataclass
class Shot:
    id: str
    number: int
    kind: str
    title: str
    purpose: str
    status: str = "queued"
    progress: int = 0
    request_id: str | None = None
    image_url: str | None = None
    error: str | None = None
    verification: str = "pending"
    cost_usd: float | None = None


@dataclass
class Shoot:
    id: str
    product_name: str
    category: str
    atmosphere: str
    background: str
    output_format: str
    engine: str
    speed_mode: str
    campaign_format: str
    generation_mode: str
    frame_count: int
    reference_count: int
    created_at: str
    remote_model: str | None = None
    provider: str | None = None
    provider_metadata: dict[str, Any] | None = None
    estimated_provider_cost_usd: float | None = None
    provider_cost_usd: float = 0.0
    provider_request_count: int = 0
    status: str = "queued"
    progress: int = 0
    stage: str = "Waiting to start"
    error: str | None = None
    shots: list[Shot] | None = None
    identity_profile: dict[str, Any] | None = None


_shoots: dict[str, Shoot] = {}
_tasks: dict[str, asyncio.Task[Any]] = {}
_frame_store: dict[str, tuple[bytes, str]] = {}


def _serialize(shoot: Shoot) -> dict[str, Any]:
    data = asdict(shoot)
    data["shots"] = [asdict(shot) for shot in (shoot.shots or [])]
    return data


def _restore_shoot(payload: dict[str, Any]) -> Shoot:
    """Restore a persisted shoot while ignoring unknown future fields."""
    shot_fields = {
        "id", "number", "kind", "title", "purpose", "status", "progress",
        "request_id", "image_url", "error", "verification", "cost_usd",
    }
    shoot_fields = {
        "id", "product_name", "category", "atmosphere", "background",
        "output_format", "engine", "speed_mode", "campaign_format",
        "generation_mode", "frame_count", "reference_count", "created_at",
        "remote_model", "provider", "provider_metadata", "provider_cost_usd",
        "estimated_provider_cost_usd", "provider_request_count", "status", "progress", "stage", "error",
        "identity_profile",
    }
    shots = [
        Shot(**{key: value for key, value in item.items() if key in shot_fields})
        for item in payload.get("shots", [])
        if isinstance(item, dict)
    ]
    values = {key: value for key, value in payload.items() if key in shoot_fields}
    values["shots"] = shots
    return Shoot(**values)


for _persisted_shoot in load_shoots():
    try:
        _restored = _restore_shoot(_persisted_shoot)
        _shoots[_restored.id] = _restored
    except (KeyError, TypeError, ValueError):
        logger.warning("Skipping malformed persisted shoot", exc_info=True)


def _persist_shoot(shoot: Shoot) -> None:
    try:
        save_shoot(_serialize(shoot))
    except OSError:
        logger.warning("Could not persist shoot %s", shoot.id, exc_info=True)


def _record_provider_usage(shoot: Shoot, shot: Shot, cost: float) -> None:
    """Record each accepted provider response, including paid retry attempts."""
    safe_cost = max(0.0, float(cost))
    shot.cost_usd = round((shot.cost_usd or 0.0) + safe_cost, 8)
    shoot.provider_cost_usd = round(shoot.provider_cost_usd + safe_cost, 8)
    shoot.provider_request_count += 1
    _persist_shoot(shoot)


def _estimate_provider_cost(engine: str, frame_count: int) -> float | None:
    if engine == "qwen-runpod":
        return round(frame_count * settings.QWEN_RUNPOD_PRICE_USD_PER_IMAGE, 8)
    if engine == "flux1-runpod":
        megapixels = (
            settings.RUNPOD_FLUX1_DEV_WIDTH
            * settings.RUNPOD_FLUX1_DEV_HEIGHT
            / 1_000_000
        )
        return round(
            frame_count
            * megapixels
            * settings.RUNPOD_FLUX1_DEV_PRICE_USD_PER_MEGAPIXEL,
            8,
        )
    return None


def _campaign_plan(campaign_format: str) -> list[tuple[str, str, str]]:
    try:
        return CAMPAIGN_PLANS[campaign_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported campaign format: {campaign_format}") from exc


def _validate_campaign_request(
    campaign_format: str,
    speed_mode: str,
    reference_count: int,
    generation_mode: str = GENERATION_MODE_PRODUCT_ONLY,
) -> None:
    """Validate the selectable campaign contract before any provider work."""
    if generation_mode not in GENERATION_MODES:
        raise ValueError("Unsupported generation mode; choose product-only or human-model")
    if campaign_format not in CAMPAIGN_FORMATS:
        raise ValueError("Unsupported campaign format")
    if campaign_format == CAMPAIGN_FORMAT_FRONT_BACK:
        if reference_count != 2:
            raise ValueError(
                "The front/back campaign requires exactly 2 references: one front and one back view"
            )
        if speed_mode != "campaign":
            raise ValueError("The front/back campaign always produces exactly 7 final images")
    elif not 1 <= reference_count <= MAX_REFERENCES:
        raise ValueError("Upload between 1 and 6 references")


def _set_shot_progress(
    shoot: Shoot,
    shot_index: int,
    progress: int,
    stage: str,
) -> None:
    """Publish a stage-aware percentage for the current photo and shoot."""
    shot = (shoot.shots or [])[shot_index]
    shot.progress = max(0, min(100, progress))
    shoot.progress = round(
        ((shot_index + (shot.progress / 100)) / max(shoot.frame_count, 1)) * 100
    )
    shoot.stage = (
        f"Frame {shot_index + 1} of {shoot.frame_count} · "
        f"{shot.title} · {stage}"
    )
    _persist_shoot(shoot)


def _detect_product_category(product_name: str, category: str, filenames: list[str]) -> str:
    """Normalize product metadata into the category-specific prompt policy."""
    haystack = " ".join([product_name, category, *filenames]).lower()
    for detected, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return detected
    return "accessories"


def _preferred_model_reference_index(
    reference_files: list[tuple[bytes, str, str]],
) -> int:
    """Prefer an uploaded reference that visibly contains the product in use."""
    model_keywords = (
        "model",
        "wear",
        "worn",
        "carry",
        "lifestyle",
        "person",
        "portrait",
        "b1",
        "b2",
    )
    for index, (_, filename, _) in enumerate(reference_files):
        normalized = filename.lower()
        if any(keyword in normalized for keyword in model_keywords):
            return index
    return 0


async def _read_bounded(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Each reference must be under {settings.MAX_UPLOAD_MB} MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _product_prompt(shoot: Shoot, shot_kind: str, attempt: int = 1, previous_count: int = 0) -> str:
    identity_details = STARTER_PRODUCT_IDENTITY.get(
        shoot.product_name.strip().lower(),
        (
            "Reference-specific identity lock: reproduce the exact silhouette, opening, handles, straps, closure, "
            "hardware, material weave, color, stitching and proportions visible in the uploaded references. Do not "
            "substitute a generic product or invent any closure, pocket, panel, logo or hardware."
        ),
    )
    identity_lock = (
        f"Product identity lock: {shoot.product_name}, category {shoot.category}. "
    )
    human_model_requirement = ""
    if (
        shoot.generation_mode == GENERATION_MODE_HUMAN_MODEL
        and shoot.category in HUMAN_MODEL_CATEGORIES
        and shot_kind in HUMAN_MODEL_SHOTS
    ):
        human_model_requirement = (
            " MANDATORY HUMAN MODEL REQUIREMENT: this frame is invalid without the required human context. "
            "Generate one real, complete professional fashion model with natural skin, face, hair, body proportions "
            "and anatomy; do not use a mannequin, silhouette, empty scene, floating product or product-only composite. "
            "For this product, keep exactly one consistent model identity across all model, editorial, lifestyle and "
            "hero frames. The product must be naturally worn or held by the model, with believable hand placement, "
            "strap tension, contact shadows and correct scale; never pasted, hovering, fused to the body or hidden "
            "behind an arm. Keep both hands and all visible fingers anatomically complete. "
        )
    plan = _campaign_plan(shoot.campaign_format)
    frame_number = next(
        index + 1 for index, (kind, *_rest) in enumerate(plan) if kind == shot_kind
    )
    format_direction = (
        "This is the strict front/back campaign: the two uploaded references are two views of one product, "
        "not two separate products. Preserve their front-versus-back distinction across the seven planned images. "
        if shoot.campaign_format == CAMPAIGN_FORMAT_FRONT_BACK
        else (
            "This is the compact six-frame campaign: deliver one front product hero, one side/rear product view, "
            "two realistic model images, one craftsmanship detail and one lifestyle image. The two model images must "
            "use the same real model identity and the exact same product. "
            if shoot.campaign_format == CAMPAIGN_FORMAT_COMPACT
            else "This is the flexible eight-frame campaign format. Use all supplied references as views of one product. "
        )
    )
    consistency_lock = shoot.identity_profile.get("consistency_lock", {}) if shoot.identity_profile else {}
    known_states = ", ".join(consistency_lock.get("known_view_states", [])) or "unknown"
    opening_policy = (
        "An opening or interior view is supported by the references; reproduce only the mechanically visible "
        "opening and interior details that are actually evidenced."
        if consistency_lock.get("supported_opening") is True
        else "The references do not support an opening or interior view. Keep this product closed and never invent "
        "a lining, pocket, compartment, zipper path, or hidden construction."
    )
    consistency_direction = (
        "PERMANENT PRODUCT CONSISTENCY LOCK: treat all supplied references as one fixed physical object, not separate "
        "design alternatives. Lock geometry, dimensions, silhouette, handle and strap attachment, flap, closure, "
        "hardware, branding, stitching, material, texture, color, camera scale, margins, background tone, white balance "
        "and shadow treatment across every frame. Known reference states: "
        f"{known_states}. {opening_policy} If a requested view cannot be supported by the references, use a supported "
        "closed-product detail or reject and regenerate; do not guess. "
    )
    return (
        SOURCE_IMAGE_ENFORCEMENT
        + " "
        + (PRODUCT_ONLY_MASTER_PROMPT if shoot.generation_mode == GENERATION_MODE_PRODUCT_ONLY else "")
        + " "
        + (
            MODEL_CONSISTENCY_AND_REALISM_LOCK
            if shoot.generation_mode == GENERATION_MODE_HUMAN_MODEL
            else ""
        )
        + " "
        + (
            PRODUCT_ONLY_CAMPAIGN_POLICY
            if shoot.generation_mode == GENERATION_MODE_PRODUCT_ONLY
            else CAMPAIGN_POLICY
        )
        + " "
        + format_direction
        + consistency_direction
        + identity_lock
        + identity_details
        + " "
        + PRODUCT_CATEGORY_RULES.get(shoot.category, PRODUCT_CATEGORY_RULES["accessories"])
        + (
            " Automatically choose the correct model type, gender presentation and pose for this category; do not "
            "ask the user and do not force a full human model when a hand, wrist, still-life or interior scene is correct. "
            if shoot.generation_mode == GENERATION_MODE_HUMAN_MODEL
            else " Product-only mode is explicit: never add a person, hand, model or human interaction to any frame. "
        )
        + human_model_requirement
        + (
            PRODUCT_ONLY_FRAME_DIRECTIONS if shoot.generation_mode == GENERATION_MODE_PRODUCT_ONLY
            else CAMPAIGN_FRAME_DIRECTIONS
        )[shot_kind]
        + f" Atmosphere: {shoot.atmosphere}. Background direction: {shoot.background}. "
        f"This is frame {frame_number} "
        f"of {shoot.frame_count}. This is generation attempt {attempt}; {previous_count} earlier frames are "
        "already accepted, so deliberately make this frame visually unlike them. Photorealistic commercial fashion "
        "photography, no text, no watermark, no extra products. "
        "Reject obvious anatomy errors, duplicated limbs, warped faces, broken hands, detached straps, floating "
        "products, impossible shadows, or any product geometry that differs from the references."
    )


def _uses_reference_identity_lock(shoot: Shoot, shot: Shot) -> bool:
    """Use the exact supplied model reference for the local CPU preview."""
    return (
        shoot.engine == "cpu"
        and shoot.speed_mode == "fast"
        and shoot.generation_mode == GENERATION_MODE_HUMAN_MODEL
        and shot.kind in HUMAN_MODEL_SHOTS
        and shoot.product_name.strip().lower() in STARTER_PRODUCT_IDENTITY
    )


def _visual_similarity(left_bytes: bytes, right_bytes: bytes) -> float:
    """Return a conservative whole-frame similarity score in the range 0..1."""
    left = Image.open(io.BytesIO(left_bytes)).convert("RGB").resize((48, 48), Image.Resampling.BILINEAR)
    right = Image.open(io.BytesIO(right_bytes)).convert("RGB").resize((48, 48), Image.Resampling.BILINEAR)
    color_diff = ImageStat.Stat(ImageChops.difference(left, right)).mean
    left_gray = left.convert("L")
    right_gray = right.convert("L")
    luminance_diff = ImageStat.Stat(ImageChops.difference(left_gray, right_gray)).mean[0]
    color_score = max(0.0, 1.0 - sum(color_diff) / (3 * 255))
    luminance_score = max(0.0, 1.0 - luminance_diff / 255)
    return round((color_score * 0.6) + (luminance_score * 0.4), 4)


def _validate_generated_image(image_bytes: bytes, previous_frames: list[bytes]) -> None:
    """Verify image integrity and reject repeated frames before surfacing them."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
    except Exception as exc:
        raise RuntimeError("Generated output failed image validation") from exc
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.width < 512 or image.height < 512:
            raise RuntimeError("Generated output is below the minimum commercial resolution")
        extrema = image.getextrema()
        if all(high - low < 4 for low, high in extrema):
            raise RuntimeError("Generated output is visually blank")
        for previous in previous_frames:
            similarity = _visual_similarity(image_bytes, previous)
            if similarity >= MAX_FRAME_SIMILARITY:
                raise RuntimeError(
                    f"Generated output repeats an accepted frame ({similarity:.0%} similarity)"
                )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Generated output could not be analyzed") from exc


def _validate_transparent_png(image_bytes: bytes) -> None:
    """Require a real RGBA image with an actual transparent background."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.format != "PNG" or image.mode != "RGBA":
            raise RuntimeError("Campaign output is not a transparent RGBA PNG")
        alpha = image.getchannel("A")
        low, high = alpha.getextrema()
        if low > 16 or high < 200:
            raise RuntimeError("Campaign output does not contain a transparent background")
        opaque_pixels = sum(1 for value in alpha.getdata() if value > 16)
        total_pixels = image.width * image.height
        if opaque_pixels < total_pixels * 0.01:
            raise RuntimeError("Campaign foreground mask is empty")
        if opaque_pixels > total_pixels * 0.995:
            raise RuntimeError("Campaign background-removal mask is effectively opaque")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Campaign output could not be checked for transparency") from exc


def _normalize_fast_output(image_bytes: bytes, fast: bool) -> bytes:
    """Keep the fast model path small, then deliver a commercial-size frame."""
    if not fast:
        return image_bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image.width >= 512 and image.height >= 512:
        return image_bytes
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def _remove_campaign_background(image_bytes: bytes, filename: str) -> bytes:
    """Run the local segmentation stage on each accepted generated frame."""
    sem = get_semaphore()
    async with sem:
        loop = asyncio.get_running_loop()
        transparent, _ = await loop.run_in_executor(
            None,
            process_image,
            image_bytes,
            filename,
            "fast",
        )
    _validate_transparent_png(transparent)
    return transparent


def _validate_campaign_distinctness(frames: list[bytes], expected_count: int) -> None:
    if len(frames) != expected_count:
        raise RuntimeError(f"Campaign validation requires all {expected_count} frames")
    for index, frame in enumerate(frames):
        _validate_generated_image(frame, frames[:index])


async def _preprocess_and_save(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    output_dir: str,
    index: int,
) -> str:
    # Keep BiRefNet as a real local preprocessing stage. Nothing is uploaded.
    sem = get_semaphore()
    async with sem:
        loop = asyncio.get_running_loop()
        cleaned, _ = await loop.run_in_executor(
            None, process_image, file_bytes, filename, "fast"
        )
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in filename.rsplit(".", 1)[0]
    )
    path = os.path.join(output_dir, f"{index:02d}_{safe_name}_clean.png")
    with open(path, "wb") as output:
        output.write(cleaned)
    return path


async def _run_shoot(shoot: Shoot, reference_files: list[tuple[bytes, str, str]]) -> None:
    work_dir = tempfile.mkdtemp(prefix=f"atelier-{shoot.id[:8]}-")
    try:
        _assert_engine_still_verified(shoot.engine, shoot.remote_model)
        shoot.status = "preparing"
        shoot.stage = "Building immutable product identity profile"
        reference_urls: list[str] = []
        for index, (content, filename, content_type) in enumerate(reference_files):
            validate_image(content)
            shoot.stage = (
                f"Extracting exact product mask · reference "
                f"{index + 1} of {len(reference_files)}"
            )
            reference_urls.append(
                await _preprocess_and_save(
                    content, filename, content_type, work_dir, index
                )
            )
        profile_path = os.path.join(work_dir, "product-identity-profile.json")
        shoot.identity_profile = build_product_identity_profile(
            source_paths=reference_urls,
            source_bytes=[content for content, _filename, _content_type in reference_files],
            product_name=shoot.product_name,
            category=shoot.category,
            output_path=profile_path,
            view_labels=(
                ["front", "back"]
                if shoot.campaign_format == CAMPAIGN_FORMAT_FRONT_BACK
                else [f"reference-{index + 1}" for index in range(len(reference_files))]
            ),
        )
        shoot.stage = "Product identity profiled · planning scene plates"

        shoot.status = "generating"
        shoot.stage = "Generating human/scene plates and controlled composites"
        seed_bytes = hashlib.sha256(shoot.id.encode()).digest()
        base_seed = int.from_bytes(seed_bytes[:4], "big")
        accepted_frames: list[bytes] = []
        model_anchor_path: str | None = None
        model_reference_index = _preferred_model_reference_index(reference_files)
        reference_identity_lock_used = False

        for index, shot in enumerate(shoot.shots or []):
            _assert_engine_still_verified(shoot.engine, shoot.remote_model)
            shot.status = "processing"
            request_id = f"{shoot.engine}-{uuid.uuid4()}"
            shot.request_id = request_id
            output_path = os.path.join(work_dir, f"frame-{index + 1:02d}.png")
            image_bytes: bytes | None = None
            last_validation_error: Exception | None = None
            composite_metadata: dict[str, Any] | None = None
            source_preserved_reference = False
            reference_identity_lock = (
                _uses_reference_identity_lock(shoot, shot)
                or (
                    shoot.engine == "cpu"
                    and shoot.speed_mode == "fast"
                    and shoot.generation_mode == GENERATION_MODE_HUMAN_MODEL
                    and shot.kind in HUMAN_MODEL_SHOTS
                    and shoot.category in HUMAN_MODEL_CATEGORIES
                )
            )
            for attempt in range(1, CAMPAIGN_MAX_ATTEMPTS + 1):
                attempt_base = round(((attempt - 1) / CAMPAIGN_MAX_ATTEMPTS) * 90)
                _set_shot_progress(
                    shoot,
                    index,
                    attempt_base + 3,
                    f"quality pass {attempt}/{CAMPAIGN_MAX_ATTEMPTS}",
                )
                try:
                    _assert_engine_still_verified(shoot.engine)
                    frame_references = reference_urls.copy()
                    human_scene_frame = (
                        shoot.generation_mode == GENERATION_MODE_HUMAN_MODEL
                        and shoot.category in HUMAN_MODEL_CATEGORIES
                        and shot.kind in HUMAN_MODEL_SHOTS
                    )
                    if model_anchor_path and shot.kind in HUMAN_MODEL_SHOTS:
                        frame_references = (
                            [model_anchor_path]
                            + frame_references[: MAX_REFERENCES - 1]
                        )
                    generation_references = reference_files
                    if model_anchor_path and human_scene_frame:
                        with open(model_anchor_path, "rb") as anchor_file:
                            model_anchor_bytes = anchor_file.read()
                        generation_references = [
                            (model_anchor_bytes, "model-identity-anchor.png", "image/png"),
                            reference_files[0],
                        ]
                    generation_prompt = (
                        human_scene_prompt(
                            category=shoot.category,
                            shot_kind=shot.kind,
                            atmosphere=shoot.atmosphere,
                            background=shoot.background,
                        )
                        if human_scene_frame
                        else _product_prompt(
                            shoot,
                            shot.kind,
                            attempt=attempt,
                            previous_count=len(accepted_frames),
                        )
                    )
                    if shoot.engine == "cpu" and human_scene_frame:
                        source_candidates = [
                            model_reference_index,
                            *[
                                candidate
                                for candidate in range(len(reference_files))
                                if candidate != model_reference_index
                            ],
                        ]
                        source_image: bytes | None = None
                        source_validation: dict[str, object] | None = None
                        for candidate in source_candidates:
                            candidate_image = reference_files[candidate][0]
                            candidate_validation = validate_human_product(
                                candidate_image,
                                [
                                    content
                                    for content, _filename, _content_type in reference_files
                                ],
                            )
                            source_validation = candidate_validation
                            if candidate_validation.get("passed") is True:
                                source_image = candidate_image
                                break
                        if source_image is None:
                            raise RuntimeError(
                                "Reference-locked human preview requires an uploaded "
                                "model-carrying reference that passes validation: "
                                + str((source_validation or {}).get("reason"))
                            )
                        # The reference already contains the real person carrying the
                        # real product. Overlaying another cutout would create a
                        # duplicate handbag, so preserve this validated source frame
                        # verbatim and label it explicitly as source-preserved.
                        image_bytes = source_image
                        source_preserved_reference = True
                    elif shoot.engine == "colab":
                        image_bytes = await generate_remote_frame(
                            references=generation_references,
                            identity_profile=shoot.identity_profile or {},
                            prompt=generation_prompt,
                            shot_kind=shot.kind,
                            seed=base_seed + (index * 101) + (attempt * 1009),
                            model=shoot.remote_model or "auto",
                        )
                    elif shoot.engine == "fal":
                        fal_result = await generate_fal_flux2_pro_frame(
                            references=generation_references,
                            prompt=generation_prompt,
                            shot_kind=shot.kind,
                            seed=base_seed + (index * 101) + (attempt * 1009),
                        )
                        image_bytes = fal_result.image_bytes
                        shot.request_id = f"fal:{fal_result.request_id}"
                    elif shoot.engine == "bfl":
                        bfl_result = await generate_black_forest_flux2_frame(
                            references=generation_references,
                            prompt=generation_prompt,
                            shot_kind=shot.kind,
                            seed=base_seed + (index * 101) + (attempt * 1009),
                        )
                        image_bytes = bfl_result.image_bytes
                        shot.request_id = f"bfl:{bfl_result.request_id}"
                    elif shoot.engine == "qwen-runpod":
                        runpod_result = await generate_runpod_qwen_frame(
                            references=generation_references[:3],
                            prompt=generation_prompt,
                            seed=base_seed + (index * 101) + (attempt * 1009),
                        )
                        image_bytes = runpod_result.image_bytes
                        shot.request_id = f"runpod:{runpod_result.request_id}"
                        _record_provider_usage(shoot, shot, runpod_result.cost)
                    elif shoot.engine == "flux1-runpod":
                        runpod_flux_result = await generate_runpod_flux1_dev_frame(
                            prompt=generation_prompt,
                            seed=base_seed + (index * 101) + (attempt * 1009),
                        )
                        image_bytes = runpod_flux_result.image_bytes
                        shot.request_id = f"runpod:{runpod_flux_result.request_id}"
                        _record_provider_usage(shoot, shot, runpod_flux_result.cost)
                    elif shoot.engine == "gemini-image":
                        gemini_result = await generate_gemini_image_frame(
                            references=generation_references[:3],
                            prompt=generation_prompt,
                            seed=base_seed + (index * 101) + (attempt * 1009),
                        )
                        image_bytes = gemini_result.image_bytes
                        shot.request_id = f"gemini:{gemini_result.request_id}"
                    else:
                        image_bytes = await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda attempt=attempt: generate_composite_frame(
                                reference_path=reference_urls[0],
                                shot_kind=shot.kind,
                                seed=base_seed + (index * 101) + (attempt * 1009),
                                output_path=output_path,
                            ),
                        )
                    _set_shot_progress(
                        shoot,
                        index,
                        attempt_base + 18,
                        f"quality pass {attempt}/{CAMPAIGN_MAX_ATTEMPTS} · generation complete",
                    )
                    image_bytes = _normalize_fast_output(
                        image_bytes,
                        shoot.speed_mode == "fast",
                    )
                    if human_scene_frame and not source_preserved_reference:
                        _set_shot_progress(
                            shoot,
                            index,
                            attempt_base + 19,
                            f"quality pass {attempt}/{CAMPAIGN_MAX_ATTEMPTS} · compositing immutable product",
                        )
                        image_bytes, composite_metadata = await asyncio.get_running_loop().run_in_executor(
                            None,
                            lambda: composite_product_layer(
                                plate_bytes=image_bytes or b"",
                                product_cutout_path=reference_urls[0],
                                shot_kind=shot.kind,
                                output_path=output_path,
                            ),
                        )
                        composite_validation = validate_immutable_product_composite(
                            image_bytes,
                            metadata=composite_metadata,
                            profile=shoot.identity_profile or {},
                        )
                        if composite_validation.get("passed") is not True:
                            raise RuntimeError(
                                "Immutable product composite validation failed: "
                                + str(composite_validation.get("reason"))
                            )
                    if human_scene_frame and shoot.category == "bags":
                        _set_shot_progress(
                            shoot,
                            index,
                            attempt_base + 20,
                            f"quality pass {attempt}/{CAMPAIGN_MAX_ATTEMPTS} · checking human and product",
                        )
                        semantic_validation = validate_human_product(
                            image_bytes,
                            [content for content, _filename, _content_type in reference_files],
                        )
                        if semantic_validation.get("passed") is not True:
                            raise RuntimeError(
                                "Human-with-product validation failed: "
                                + str(semantic_validation.get("reason"))
                            )
                    _set_shot_progress(
                        shoot,
                        index,
                        attempt_base + 21,
                        f"quality pass {attempt}/{CAMPAIGN_MAX_ATTEMPTS} · removing background",
                    )
                    image_bytes = await _remove_campaign_background(
                        image_bytes,
                        f"{shoot.product_name}_{index + 1:02d}_{shot.kind}.png",
                    )
                    if shoot.generation_mode == GENERATION_MODE_PRODUCT_ONLY:
                        consistency_validation = validate_product_consistency(
                            image_bytes,
                            shoot.identity_profile or {},
                            previous_frames=accepted_frames,
                            shot_kind=shot.kind,
                        )
                        if consistency_validation.get("passed") is not True:
                            raise RuntimeError(
                                "Permanent product consistency validation failed: "
                                + str(consistency_validation.get("reason"))
                            )
                    _set_shot_progress(
                        shoot,
                        index,
                        attempt_base + 26,
                        f"quality pass {attempt}/{CAMPAIGN_MAX_ATTEMPTS} · validating output",
                    )
                    _validate_generated_image(image_bytes, accepted_frames)
                    break
                except Exception as exc:
                    last_validation_error = exc
                    image_bytes = None
                    if isinstance(
                        exc, (
                            FalFlux2ProError,
                            BlackForestFlux2Error,
                            RunPodFlux1DevError,
                            GeminiImageError,
                        )
                    ) and not exc.retryable:
                        break
                    if "no remaining image credits" in str(exc).lower():
                        break
                    if attempt < CAMPAIGN_MAX_ATTEMPTS:
                        logger.warning(
                            "Shoot %s rejected frame %s attempt %s: %s",
                            shoot.id,
                            index + 1,
                            attempt,
                            exc,
                        )
            if _uses_reference_identity_lock(shoot, shot):
                reference_identity_lock_used = True
                _set_shot_progress(
                    shoot,
                    index,
                    92,
                    "preserving supplied model reference",
                )
                source_candidates = [
                    model_reference_index,
                    *[
                        candidate
                        for candidate in range(len(reference_files))
                        if candidate != model_reference_index
                    ],
                ]
                source_image: bytes | None = None
                source_validation: dict[str, object] | None = None
                for candidate in source_candidates:
                    candidate_image = reference_files[candidate][0]
                    candidate_validation = validate_human_product(
                        candidate_image,
                        [
                            content
                            for content, _filename, _content_type in reference_files
                        ],
                    )
                    if candidate_validation.get("passed") is True:
                        source_image = candidate_image
                        source_validation = candidate_validation
                        break
                    source_validation = candidate_validation
                if source_image is None:
                    raise RuntimeError(
                        "None of the supplied references passed the local "
                        "human-with-product validator: "
                        + str((source_validation or {}).get("reason"))
                    )
                image_bytes = await _remove_campaign_background(
                    source_image,
                    f"{shoot.product_name}_{index + 1:02d}_{shot.kind}_reference.png",
                )
            if image_bytes is None:
                raise RuntimeError(
                    f"Frame {index + 1} failed campaign quality validation after "
                    f"{CAMPAIGN_MAX_ATTEMPTS} attempts: {last_validation_error}"
                )
            accepted_frames.append(image_bytes)
            if shot.kind in HUMAN_MODEL_SHOTS and shoot.category in HUMAN_MODEL_CATEGORIES and model_anchor_path is None:
                model_anchor_path = output_path
            if shoot.output_format != "png":
                converted = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                buffer = io.BytesIO()
                converted.save(buffer, format="JPEG" if shoot.output_format == "jpeg" else "WEBP", quality=95)
                image_bytes = buffer.getvalue()
            _frame_store[shot.id] = (
                image_bytes,
                "image/jpeg" if shoot.output_format == "jpeg" else f"image/{shoot.output_format}",
            )
            save_frame(
                shot.id,
                image_bytes,
                "image/jpeg" if shoot.output_format == "jpeg" else f"image/{shoot.output_format}",
            )
            shot.image_url = f"/api/shoots/{shoot.id}/frames/{shot.id}"
            shot.verification = (
                "reference-locked · source preserved"
                if reference_identity_lock
                else "validated · immutable product composite · human + contact + identity"
                if composite_metadata is not None
                else "validated · human + bag + identity proxy"
                if human_scene_frame and shoot.category == "bags"
                else "validated · integrity · distinctness"
            )
            shot.status = "completed"
            shot.progress = 100
            shoot.progress = round(((index + 1) / shoot.frame_count) * 100)
            shoot.stage = f"Frame {index + 1} of {shoot.frame_count} validated"
            _persist_shoot(shoot)

        _validate_campaign_distinctness(accepted_frames, shoot.frame_count)
        shoot.status = "completed"
        shoot.stage = (
            "Fast preview reference lock applied"
            if reference_identity_lock_used
            else "Fast preview validated"
            if shoot.speed_mode == "fast"
            else "Eight-frame campaign validated"
        )
        shoot.progress = 100
        _persist_shoot(shoot)
    except asyncio.CancelledError:
        if shoot.engine == "bfl":
            for shot in shoot.shots or []:
                if shot.status == "processing" and shot.request_id:
                    await cancel_black_forest_frame(
                        shot.request_id.removeprefix("bfl:")
                    )
        shoot.status = "cancelled"
        shoot.stage = "Shoot cancelled"
        _persist_shoot(shoot)
        raise
    except Exception as exc:
        logger.exception("Shoot %s failed", shoot.id)
        shoot.status = "failed"
        shoot.stage = "Generation stopped"
        shoot.error = str(exc)
        for shot in shoot.shots or []:
            if shot.status in {"queued", "processing"}:
                shot.status = "failed"
                shot.error = "Generation stopped before this frame completed"
        _persist_shoot(shoot)
    finally:
        _tasks.pop(shoot.id, None)
        _persist_shoot(shoot)
        shutil.rmtree(work_dir, ignore_errors=True)


@router.post("/shoots")
async def create_shoot(
    product_name: str = Form(default="Untitled product"),
    category: str = Form(default="Luxury handbag"),
    atmosphere: str = Form(default="Quiet, architectural luxury"),
    background: str = Form(default="Luxury white studio"),
    output_format: str = Form(default="png"),
    engine: str = Form(default="auto"),
    speed_mode: str = Form(default="fast"),
    campaign_format: str = Form(default=CAMPAIGN_FORMAT_FLEXIBLE),
    generation_mode: str = Form(default=GENERATION_MODE_PRODUCT_ONLY),
    references: list[UploadFile] = File(...),
):
    try:
        _validate_campaign_request(campaign_format, speed_mode, len(references), generation_mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(_tasks) >= MAX_ACTIVE_SHOOTS:
        raise HTTPException(status_code=429, detail="The studio queue is currently full")
    if output_format not in {"png", "jpeg", "webp"}:
        raise HTTPException(status_code=422, detail="Unsupported output format")
    if not product_name.strip():
        raise HTTPException(status_code=422, detail="Product name is required")
    if engine not in ENGINE_MODES:
        raise HTTPException(status_code=422, detail="Unsupported generation engine")
    if speed_mode not in {"fast", "campaign"}:
        raise HTTPException(status_code=422, detail="Unsupported speed mode")
    selected_engine, remote_model = _resolve_generation_backend(engine)
    if selected_engine == "colab":
        colab = remote_worker_status()
        if colab.get("ready") is not True:
            raise HTTPException(status_code=503, detail=_engine_unavailable_detail("colab", colab))
        if not _remote_model_matches_status(remote_model, colab):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Requested Colab model {remote_model!r} is not the worker's "
                    f"verified provider {colab.get('provider')!r}."
                ),
            )
    elif selected_engine == "fal":
        fal = fal_flux2_pro_status()
        if fal.get("ready") is not True:
            raise HTTPException(status_code=503, detail=_engine_unavailable_detail("fal", fal))
    elif selected_engine == "bfl":
        bfl = black_forest_flux2_status()
        if bfl.get("ready") is not True:
            raise HTTPException(status_code=503, detail=_engine_unavailable_detail("bfl", bfl))
    elif selected_engine == "qwen-runpod":
        if len(references) > 3:
            raise HTTPException(
                status_code=422,
                detail="RunPod Qwen Image Edit accepts at most 3 reference images",
            )
        runpod = runpod_qwen_status()
        if runpod.get("ready") is not True:
            raise HTTPException(
                status_code=503,
                detail=_engine_unavailable_detail("qwen-runpod", runpod),
            )
    elif selected_engine == "flux1-runpod":
        runpod_flux1_dev = runpod_flux1_dev_status()
        if runpod_flux1_dev.get("ready") is not True:
            raise HTTPException(
                status_code=503,
                detail=_engine_unavailable_detail("flux1-runpod", runpod_flux1_dev),
            )
    elif selected_engine == "gemini-image":
        if len(references) > 3:
            raise HTTPException(
                status_code=422,
                detail="Gemini image generation accepts at most 3 reference images",
            )
        gemini = gemini_image_status()
        if gemini.get("ready") is not True:
            raise HTTPException(
                status_code=503,
                detail=_engine_unavailable_detail("gemini-image", gemini),
            )
    reference_files: list[tuple[bytes, str, str]] = []
    for file in references:
        content = await _read_bounded(file)
        try:
            validate_image(content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        reference_files.append((content, file.filename or "reference.png", file.content_type or ""))

    shoot_id = str(uuid.uuid4())
    detected_category = _detect_product_category(
        product_name,
        category,
        [filename for _, filename, _ in reference_files],
    )
    if (
        generation_mode == GENERATION_MODE_HUMAN_MODEL
        and selected_engine == "cpu"
        and detected_category in HUMAN_MODEL_CATEGORIES
        and speed_mode != "fast"
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                f"{detected_category.title()} campaign mode requires a verified Colab "
                "human-generation backend. CPU can only create a fast, source-preserved "
                "preview from an uploaded model-carrying reference."
            ),
        )
    campaign_plan = _campaign_plan(campaign_format)
    if speed_mode == "fast":
        fast_kind = (
            "model"
            if generation_mode == GENERATION_MODE_HUMAN_MODEL
            and detected_category in HUMAN_MODEL_CATEGORIES
            else "studio"
        )
        selected_shot_plan = [plan for plan in campaign_plan if plan[0] == fast_kind]
    else:
        selected_shot_plan = campaign_plan
    shoot = Shoot(
        id=shoot_id,
        product_name=product_name.strip()[:120],
        category=detected_category,
        atmosphere=atmosphere.strip()[:180],
        background=background.strip()[:120],
        output_format=output_format,
        engine=selected_engine,
        remote_model=remote_model,
        provider=(
            "fal.ai"
            if selected_engine == "fal"
            else "Black Forest Labs"
            if selected_engine == "bfl"
            else "colab"
            if selected_engine == "colab"
            else "RunPod"
            if selected_engine == "qwen-runpod"
            else "RunPod"
            if selected_engine == "flux1-runpod"
            else "Google Gemini"
            if selected_engine == "gemini-image"
            else "local"
        ),
        provider_metadata=(
            {
                "provider": "fal.ai",
                "model": "fal-ai/flux-2-pro/edit",
                "billing": "fal.ai usage-based API",
                "multi_reference": True,
                "validation": "per-frame identity, human/product, anatomy, composition, and quality gates",
            }
            if selected_engine == "fal"
            else {
                "provider": "Black Forest Labs",
                "model": remote_model,
                "billing": "Black Forest Labs usage-based API",
                "multi_reference": True,
                "reference_transport": "server-routed binary data URIs",
                "validation": "per-frame identity, human/product, anatomy, composition, and quality gates",
            }
            if selected_engine == "bfl"
            else {
                "provider": "RunPod",
                "model": "qwen-image-edit-2511",
                "endpoint": settings.QWEN_RUNPOD_ENDPOINT,
                "billing": "RunPod public endpoint usage-based API",
                "multi_reference": True,
                "max_references": 3,
                "reference_transport": "server-routed base64 data URLs",
                "validation": "per-frame product identity, anatomy, composition, and quality gates",
            }
            if selected_engine == "qwen-runpod"
            else {
                "provider": "RunPod",
                "model": "black-forest-labs-flux-1-dev",
                "endpoint": settings.RUNPOD_FLUX1_DEV_ENDPOINT,
                "billing": (
                    f"${settings.RUNPOD_FLUX1_DEV_PRICE_USD_PER_MEGAPIXEL:.2f} "
                    "per megapixel"
                ),
                "reference_transport": "references are not supported by this text-to-image endpoint",
                "validation": "per-frame product identity, anatomy, composition, and quality gates",
            }
            if selected_engine == "flux1-runpod"
            else {
                "provider": "Google Gemini",
                "model": settings.GEMINI_IMAGE_MODEL,
                "billing": "Managed Gemini image integration",
                "multi_reference": True,
                "max_references": 3,
                "reference_transport": "server-routed inline image data",
                "validation": "per-frame product identity, composition, distinctness, and quality gates",
            }
            if selected_engine == "gemini-image"
            else None
        ),
        estimated_provider_cost_usd=_estimate_provider_cost(
            selected_engine, len(selected_shot_plan)
        ),
        speed_mode=speed_mode,
        campaign_format=campaign_format,
        generation_mode=generation_mode,
        frame_count=len(selected_shot_plan),
        reference_count=len(reference_files),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        shots=[
            Shot(id=f"{shoot_id}-{index + 1}", number=index + 1, kind=kind, title=title, purpose=purpose)
            for index, (kind, title, purpose) in enumerate(selected_shot_plan)
        ],
    )
    _shoots[shoot_id] = shoot
    _persist_shoot(shoot)
    _tasks[shoot_id] = asyncio.create_task(_run_shoot(shoot, reference_files))
    return _serialize(shoot)


@router.get("/shoots/history")
async def shoot_history():
    """Return durable paid-generation history, newest first."""
    return [
        _serialize(shoot)
        for shoot in sorted(
            _shoots.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )
    ]


@router.get("/shopify/products")
async def shopify_products(query: str = ""):
    """List Shopify products available to import into the studio."""
    try:
        return await list_shopify_products(query)
    except ShopifyImportError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/shoots/{shoot_id}")
async def get_shoot(shoot_id: str):
    shoot = _shoots.get(shoot_id)
    if shoot is None:
        raise HTTPException(status_code=404, detail="Shoot not found")
    return _serialize(shoot)


@router.post("/shoots/{shoot_id}/cancel")
async def cancel_shoot(shoot_id: str):
    shoot = _shoots.get(shoot_id)
    task = _tasks.get(shoot_id)
    if shoot is None:
        raise HTTPException(status_code=404, detail="Shoot not found")
    if task and not task.done():
        task.cancel()
    return _serialize(shoot)


@router.get("/shoots/{shoot_id}/frames/{frame_id}")
async def get_local_frame(shoot_id: str, frame_id: str):
    shoot = _shoots.get(shoot_id)
    if shoot is None:
        raise HTTPException(status_code=404, detail="Shoot not found")
    frame = _frame_store.get(frame_id)
    if frame is None:
        frame = load_frame(frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    from fastapi.responses import Response

    return Response(
        content=frame[0],
        media_type=frame[1],
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/shoots/{shoot_id}/export")
async def export_shoot(shoot_id: str):
    """Package validated frames and a Shopify-oriented manifest into a ZIP."""
    shoot = _shoots.get(shoot_id)
    if shoot is None:
        raise HTTPException(status_code=404, detail="Shoot not found")
    shots = shoot.shots or []
    if shoot.status != "completed" or len([shot for shot in shots if shot.image_url]) != shoot.frame_count:
        raise HTTPException(status_code=409, detail="The validated frames are not ready yet")

    archive = io.BytesIO()
    manifest = {
        "product_name": shoot.product_name,
        "category": shoot.category,
        "atmosphere": shoot.atmosphere,
        "background": shoot.background,
        "output_format": shoot.output_format,
        "generated_at": shoot.created_at,
        "identity_lock": True,
        "product_identity_profile": shoot.identity_profile,
        "product_pipeline": (
            "immutable-source-product-composite"
            if shoot.identity_profile
            else "legacy-validated-local-generation"
        ),
        "frames": [],
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for index, shot in enumerate(shots, start=1):
            frame = _frame_store.get(shot.id)
            if frame is None:
                frame = load_frame(shot.id)
            if frame is None:
                raise HTTPException(status_code=502, detail=f"Could not export frame {index}")
            extension = "jpg" if shoot.output_format == "jpeg" else shoot.output_format
            filename = f"{shoot.product_name}_{index:02d}_{shot.kind}.{extension}"
            bundle.writestr(filename, frame[0])
            manifest["frames"].append({
                "position": index,
                "filename": filename,
                "title": shot.title,
                "purpose": shot.purpose,
                "verification": shot.verification,
            })
        bundle.writestr("shopify_manifest.json", json.dumps(manifest, indent=2))
    archive.seek(0)
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in shoot.product_name)
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_atelier_export.zip"'},
    )


@router.get("/admin/metrics")
async def get_admin_metrics():
    """Lightweight operational metrics for the current service process."""
    shoots = list(_shoots.values())
    frames = [shot for shoot in shoots for shot in (shoot.shots or [])]
    completed = [shoot for shoot in shoots if shoot.status == "completed"]
    usage_definitions = (
        (
            "qwen-runpod",
            "Qwen Image Edit 2511",
            "$0.02 per image",
            settings.QWEN_RUNPOD_PRICE_USD_PER_IMAGE,
        ),
        (
            "flux1-runpod",
            "FLUX.1 Dev",
            f"${settings.RUNPOD_FLUX1_DEV_PRICE_USD_PER_MEGAPIXEL:.2f} per megapixel",
            None,
        ),
    )
    provider_usage = []
    for engine, model, price, estimated_per_image in usage_definitions:
        matching = [shoot for shoot in shoots if shoot.engine == engine]
        provider_usage.append(
            {
                "provider": "RunPod",
                "model": model,
                "price": price,
                "requests": sum(shoot.provider_request_count for shoot in matching),
                "cost_usd": round(
                    sum(shoot.provider_cost_usd for shoot in matching), 8
                ),
                "estimated_cost_per_image_usd": estimated_per_image,
            }
        )
    provider_cost_usd = round(
        sum(entry["cost_usd"] for entry in provider_usage),
        8,
    )
    return {
        "total_shoots": len(shoots),
        "active_shoots": len(_tasks),
        "completed_shoots": len(completed),
        "failed_shoots": len([shoot for shoot in shoots if shoot.status == "failed"]),
        "cancelled_shoots": len([shoot for shoot in shoots if shoot.status == "cancelled"]),
        "frames_generated": len([shot for shot in frames if shot.verification == "validated"]),
        "frames_failed": len([shot for shot in frames if shot.status == "failed"]),
        "provider": "Local and hosted providers; RunPod usage tracked below",
        "preprocessor": "BiRefNet",
        "provider_cost_usd": provider_cost_usd,
        "provider_usage": provider_usage,
    }
