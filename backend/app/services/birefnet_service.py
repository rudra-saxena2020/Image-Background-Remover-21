"""
BiRefNet model lifecycle manager.

Loads the model once at startup, moves it to GPU when available,
enables FP16 on supported CUDA GPUs, warms up the model, and exposes
inference. Never re-initialises the model inside a request.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation

from app.config import settings
from app.utils.timing import TimingContext

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    spec = settings.DEVICE.strip().lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device(spec)


# ── Inference resolution table ────────────────────────────────────────────────
# (max_side_px, inference_size) pairs for fast mode; high-quality uses larger.
# Fast mode uses a small input-dependent resolution to keep CPU previews within
# the interactive budget; high-quality mode remains at the full 2048px mask.
_FAST_RESOLUTION_TABLE = [
    (512, 256),
    (1024, 512),
    (2048, 768),
    (float("inf"), 1024),
]
_HQ_RESOLUTION = 2048


def _pick_inference_size(orig_w: int, orig_h: int, quality: str) -> int:
    if quality == "high":
        return _HQ_RESOLUTION
    max_side = max(orig_w, orig_h)
    for threshold, size in _FAST_RESOLUTION_TABLE:
        if max_side <= threshold:
            return size
    return 1024


# ── Mean / std for BiRefNet normalisation ────────────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


class BiRefNetService:
    """Singleton-style service that owns the BiRefNet model lifecycle."""

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._device: Optional[torch.device] = None
        self._fp16: bool = False
        self._model_id: str = ""
        self._warmup_done: bool = False
        self._loaded: bool = False

    # ── Startup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load BiRefNet once. Called at application startup."""
        if self._loaded:
            return

        device = _resolve_device()
        use_fp16 = (
            settings.USE_FP16
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        model_id = settings.BIREFNET_MODEL

        logger.info("Loading BiRefNet model %s on %s (fp16=%s)", model_id, device, use_fp16)
        t0 = time.perf_counter()

        model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        model = model.to(device)  # type: ignore[attr-defined]
        if use_fp16:
            model = model.half()
        model.eval()

        # Store references
        self._model = model
        self._device = device
        self._fp16 = use_fp16
        self._model_id = model_id
        self._loaded = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("BiRefNet loaded in %.0f ms", elapsed)

        if settings.MODEL_WARMUP:
            self._warmup()

    def _warmup(self) -> None:
        """Run a dummy forward pass so the first real request is not penalised."""
        logger.info("Warming up BiRefNet …")
        # Always use float32 for warmup — inference_mode converts internally
        dummy = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
        dummy = dummy.to(self._device)
        if self._fp16:
            dummy = dummy.half()
        try:
            expected_dtype = next(self._model.parameters()).dtype
            if dummy.dtype != expected_dtype:
                dummy = dummy.to(dtype=expected_dtype)
            with torch.inference_mode():
                self._model(dummy)  # type: ignore[operator]
            if self._device and self._device.type == "cuda":
                torch.cuda.synchronize()
            self._warmup_done = True
            logger.info("Warmup complete")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)

    # ── Inference ─────────────────────────────────────────────────────────────

    def remove_background(
        self,
        image_bytes: bytes,
        quality: str = "fast",
    ) -> tuple[bytes, dict]:
        """
        Run BiRefNet on image_bytes and return (transparent_png_bytes, metadata).

        All processing stays in memory — no disk I/O on the hot path.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded")

        timing = TimingContext()

        # ── Decode ────────────────────────────────────────────────────────────
        timing.start("decode")
        orig_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = orig_image.size
        timing.end("decode")

        # ── Preprocess ────────────────────────────────────────────────────────
        timing.start("preprocess")
        inf_size = _pick_inference_size(orig_w, orig_h, quality)
        resized = orig_image.resize((inf_size, inf_size), Image.BILINEAR)

        img_tensor = torch.from_numpy(np.array(resized)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC → CHW
        img_tensor = (img_tensor - _MEAN[:, None, None]) / _STD[:, None, None]
        img_tensor = img_tensor.unsqueeze(0)  # add batch dim
        img_tensor = img_tensor.to(self._device)
        # Match the model's dtype (FP16 model on CPU/GPU can fail if input is float32)
        expected_dtype = next(self._model.parameters()).dtype
        if img_tensor.dtype != expected_dtype:
            img_tensor = img_tensor.to(dtype=expected_dtype)
        timing.end("preprocess")

        # ── GPU inference ─────────────────────────────────────────────────────
        timing.start("inference")
        with torch.inference_mode():
            preds = self._model(img_tensor)  # type: ignore[operator]
            # BiRefNet returns a list of tensors; use the last (finest) prediction
            if isinstance(preds, (list, tuple)):
                pred = preds[-1]
            else:
                pred = preds
            # Shape: (1, 1, H, W) — sigmoid to [0, 1]
            pred = pred.sigmoid()

        if self._device and self._device.type == "cuda":
            torch.cuda.synchronize()
        timing.end("inference")

        # ── Mask post-processing ──────────────────────────────────────────────
        timing.start("postprocess")
        # Upsample mask to original resolution
        mask_tensor = F.interpolate(
            pred.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        mask_np = mask_tensor.squeeze().cpu().numpy()
        # Convert to 0-255 uint8
        mask_uint8 = (mask_np * 255).clip(0, 255).astype(np.uint8)
        alpha_channel = Image.fromarray(mask_uint8, mode="L")

        # Composite: paste alpha onto original RGBA
        orig_rgba = orig_image.convert("RGBA")
        orig_rgba.putalpha(alpha_channel)
        timing.end("postprocess")

        # ── Encode PNG ────────────────────────────────────────────────────────
        timing.start("png_encode")
        buf = io.BytesIO()
        orig_rgba.save(buf, format="PNG", optimize=False)
        result_bytes = buf.getvalue()
        timing.end("png_encode")

        if settings.LOG_TIMING:
            logger.info("Processing timings:\n%s", timing.log())

        metadata = {
            "inference_time_ms": timing.stages.get("inference", 0),
            "total_time_ms": timing.total_ms(),
            "input_width": orig_w,
            "input_height": orig_h,
            "inference_width": inf_size,
            "inference_height": inf_size,
            "device": str(self._device),
            "mode": quality,
            "stages": dict(timing.stages),
        }
        return result_bytes, metadata

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "model_loaded": self._loaded,
            "device": str(self._device) if self._device else "unknown",
            "cuda_available": torch.cuda.is_available(),
            "model": self._model_id if self._loaded else None,
            "fp16_enabled": self._fp16,
            "warmup_done": self._warmup_done,
        }


# Global singleton
birefnet = BiRefNetService()
