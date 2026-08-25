"""Local visual checks for a human visibly carrying or wearing a bag."""

from __future__ import annotations

import io
import hashlib
import math
import threading
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageFilter, ImageStat

from app.config import settings

_model: Any | None = None
_weights: Any | None = None
_model_lock = threading.Lock()
_BAG_LABELS = {"handbag", "backpack", "suitcase"}
VALIDATOR_SHA256 = "dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf"


def validator_status() -> dict[str, object]:
    checkpoint = Path(settings.HUMAN_PRODUCT_VALIDATOR_MODEL_PATH).expanduser()
    return {
        "id": "human-product-validator",
        "name": "Local COCO person + bag validator",
        "model_path": str(checkpoint),
        "model_present": checkpoint.is_file(),
        "ready": checkpoint.is_file(),
        "reason": (
            None
            if checkpoint.is_file()
            else "Local Faster R-CNN validator weights are missing; smoke tests fail closed."
        ),
    }


def _load_model() -> tuple[Any, Any]:
    global _model, _weights
    if _model is not None and _weights is not None:
        return _model, _weights
    checkpoint = Path(settings.HUMAN_PRODUCT_VALIDATOR_MODEL_PATH).expanduser()
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Local human-product validator checkpoint is missing: {checkpoint}"
        )
    with _model_lock:
        if _model is not None and _weights is not None:
            return _model, _weights
        digest = hashlib.sha256()
        with checkpoint.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != VALIDATOR_SHA256:
            raise RuntimeError("Local human-product validator checkpoint checksum failed.")
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
        _weights = weights
    return _model, _weights


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_distance(left: list[float], right: list[float]) -> float:
    horizontal = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(horizontal, vertical)


def _histogram(image: Image.Image) -> list[float]:
    sample = image.convert("HSV").resize((96, 96), Image.Resampling.LANCZOS)
    values: list[float] = []
    for channel in sample.split():
        histogram = channel.histogram()
        for start in range(0, 256, 16):
            values.append(float(sum(histogram[start : start + 16])))
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _identity_similarity(product_crop: Image.Image, references: Iterable[bytes]) -> float:
    target = _histogram(product_crop)
    similarities: list[float] = []
    for reference in references:
        try:
            image = Image.open(io.BytesIO(reference)).convert("RGB")
        except Exception:
            continue
        candidate = _histogram(image)
        color = sum(left * right for left, right in zip(target, candidate))
        target_edges = ImageStat.Stat(
            product_crop.convert("L").resize((96, 96)).filter(ImageFilter.FIND_EDGES)
        ).mean[0]
        candidate_edges = ImageStat.Stat(
            image.convert("L").resize((96, 96)).filter(ImageFilter.FIND_EDGES)
        ).mean[0]
        edge = max(0.0, 1.0 - abs(target_edges - candidate_edges) / 255.0)
        similarities.append((color * 0.8) + (edge * 0.2))
    return max(similarities, default=0.0)


def validate_human_product(
    image_bytes: bytes, reference_images: Iterable[bytes]
) -> dict[str, object]:
    result: dict[str, object] = {
        "passed": False,
        "person_detected": False,
        "bag_detected": False,
        "physical_contact": False,
        "believable_scale": False,
        "identity_proxy_passed": False,
        "anatomy_screening_passed": False,
        "anatomy_screening": "not-run",
        "person_score": 0.0,
        "bag_score": 0.0,
        "identity_score": 0.0,
        "person_area_ratio": 0.0,
        "bag_area_ratio": 0.0,
        "bag_to_person_ratio": 0.0,
        "reason": "",
    }
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
    except Exception as exc:
        result["reason"] = f"Output is not a readable image: {exc}"
        return result
    if image.width < 512 or image.height < 512:
        result["reason"] = "Output is below the 512px minimum validation size."
        return result
    try:
        model, weights = _load_model()
        import torch

        tensor = weights.transforms()(image)
        with torch.inference_mode():
            prediction = model([tensor])[0]
    except Exception as exc:
        result["reason"] = f"Local semantic validator unavailable: {exc}"
        return result

    categories = weights.meta["categories"]
    people: list[tuple[float, list[float]]] = []
    bags: list[tuple[float, list[float]]] = []
    for score_value, label_value, box_value in zip(
        prediction["scores"].tolist(),
        prediction["labels"].tolist(),
        prediction["boxes"].tolist(),
    ):
        label = categories[int(label_value)]
        box = [float(value) for value in box_value]
        if label == "person" and score_value >= settings.HUMAN_PRODUCT_PERSON_SCORE:
            people.append((float(score_value), box))
        if label in _BAG_LABELS and score_value >= settings.HUMAN_PRODUCT_BAG_SCORE:
            bags.append((float(score_value), box))
    if not people:
        result["reason"] = "No real person was detected at the required confidence."
        return result
    result["person_detected"] = True
    result["person_score"] = round(people[0][0], 4)
    person_box = people[0][1]
    person_width = max(0.0, person_box[2] - person_box[0])
    person_height = max(0.0, person_box[3] - person_box[1])
    anatomy_screening_passed = (
        person_width >= 32
        and person_height >= 64
        and 0.12 <= person_width / max(person_height, 1.0) <= 1.5
    )
    result["anatomy_screening_passed"] = anatomy_screening_passed
    result["anatomy_screening"] = (
        "basic person-box proportions passed"
        if anatomy_screening_passed
        else "person-box proportions are too small or implausible for anatomy review"
    )
    if not bags:
        result["reason"] = "No handbag, backpack, or suitcase was detected with the person."
        return result
    result["bag_detected"] = True
    result["bag_score"] = round(bags[0][0], 4)
    image_area = float(image.width * image.height)
    pairs: list[
        tuple[bool, bool, float, float, list[float], list[float], float, float]
    ] = []
    for person_score, person_candidate in people:
        person_area = _box_area(person_candidate)
        person_diagonal = math.hypot(
            person_candidate[2] - person_candidate[0],
            person_candidate[3] - person_candidate[1],
        ) or 1.0
        for bag_score, bag_candidate in bags:
            bag_area = _box_area(bag_candidate)
            scale_ratio = bag_area / max(person_area, 1.0)
            contact = (
                _box_distance(person_candidate, bag_candidate)
                <= person_diagonal * 0.12
            )
            believable_scale = (
                person_area / image_area >= 0.08
                and bag_area / image_area >= 0.008
                and 0.015 <= scale_ratio <= 0.75
            )
            pairs.append(
                (
                    contact,
                    believable_scale,
                    person_score,
                    bag_score,
                    person_candidate,
                    bag_candidate,
                    person_area,
                    bag_area,
                )
            )
    pairs.sort(
        key=lambda pair: (
            pair[0] and pair[1],
            pair[0],
            pair[1],
            pair[2] + pair[3],
        ),
        reverse=True,
    )
    (
        contact,
        believable_scale,
        person_score,
        bag_score,
        person_box,
        bag_box,
        person_area,
        bag_area,
    ) = pairs[0]
    scale_ratio = bag_area / max(person_area, 1.0)
    result["person_score"] = round(person_score, 4)
    result["bag_score"] = round(bag_score, 4)
    result["person_area_ratio"] = round(person_area / image_area, 4)
    result["bag_area_ratio"] = round(bag_area / image_area, 4)
    result["bag_to_person_ratio"] = round(scale_ratio, 4)
    result["physical_contact"] = contact
    result["believable_scale"] = believable_scale

    crop_box = tuple(max(0, round(value)) for value in bag_box)
    product_crop = image.crop(crop_box)
    identity_score = _identity_similarity(product_crop, reference_images)
    identity_passed = identity_score >= settings.HUMAN_PRODUCT_IDENTITY_SCORE
    result["identity_score"] = round(identity_score, 4)
    result["identity_proxy_passed"] = identity_passed
    result["passed"] = bool(
        contact and believable_scale and identity_passed and anatomy_screening_passed
    )
    if not contact:
        result["reason"] = "The detected bag is not in physical contact with the model."
    elif not believable_scale:
        result["reason"] = "The model or bag scale is not commercially believable."
    elif not identity_passed:
        result["reason"] = "The detected bag does not match the uploaded references closely enough."
    elif not anatomy_screening_passed:
        result["reason"] = "Basic human anatomy screening failed; the person crop is not commercially usable."
    else:
        result["reason"] = "Person, bag, physical contact, scale, and reference identity proxy passed."
    return result