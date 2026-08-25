import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_HIDREAM_ROOT = _WORKSPACE_ROOT / ".local" / "hidream"
_FLUX2_ROOT = _WORKSPACE_ROOT / ".local" / "flux2"
_QWEN_ROOT = _WORKSPACE_ROOT / ".local" / "qwen"
_FLUX_SCHNELL_ROOT = _WORKSPACE_ROOT / ".local" / "flux-schnell"
_SDXL_ROOT = _WORKSPACE_ROOT / ".local" / "sdxl"
_FOOOCUS_ROOT = _WORKSPACE_ROOT / ".local" / "fooocus"
_VALIDATION_ROOT = _WORKSPACE_ROOT / ".local" / "generation-validation"


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    DEVICE: str = os.environ.get("DEVICE", "auto")
    QUALITY_DEFAULT: str = os.environ.get("QUALITY_DEFAULT", "fast")
    USE_FP16: bool = _parse_bool(os.environ.get("USE_FP16"), True)
    MODEL_WARMUP: bool = _parse_bool(os.environ.get("MODEL_WARMUP"), True)
    MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "20"))
    MAX_GPU_CONCURRENCY: int = int(os.environ.get("MAX_GPU_CONCURRENCY", "1"))
    ENABLE_RESULT_CACHE: bool = _parse_bool(os.environ.get("ENABLE_RESULT_CACHE"), True)
    RESULT_CACHE_TTL: int = int(os.environ.get("RESULT_CACHE_TTL", "3600"))
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in (os.environ.get("CORS_ORIGINS") or "").split(",")
        if o.strip()
    ] or ["*"]
    BIREFNET_MODEL: str = os.environ.get("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")
    BIREFNET_MODEL_LITE: str = os.environ.get(
        "BIREFNET_MODEL_LITE", "ZhengPeng7/BiRefNet_lite"
    )
    LOG_TIMING: bool = _parse_bool(os.environ.get("LOG_TIMING"), True)
    # Local/open-source generation engines.
    HIDREAM_REPO: str = os.environ.get(
        "HIDREAM_REPO", str(_HIDREAM_ROOT / "HiDream-O1-Image")
    )
    HIDREAM_MODEL_PATH: str = os.environ.get(
        "HIDREAM_MODEL_PATH", str(_HIDREAM_ROOT / "HiDream-O1-Image-Dev")
    )
    HIDREAM_PYTHON: str = os.environ.get(
        "HIDREAM_PYTHON", str(_HIDREAM_ROOT / ".venv" / "bin" / "python")
    )
    HIDREAM_MODEL_TYPE: str = os.environ.get("HIDREAM_MODEL_TYPE", "dev")
    HIDREAM_WIDTH: int = int(os.environ.get("HIDREAM_WIDTH", "2048"))
    HIDREAM_HEIGHT: int = int(os.environ.get("HIDREAM_HEIGHT", "2048"))
    HIDREAM_SHIFT: float = float(os.environ.get("HIDREAM_SHIFT", "1"))
    HIDREAM_TIMEOUT_SECONDS: int = int(
        os.environ.get("HIDREAM_TIMEOUT_SECONDS", "2700")
    )
    FLUX2_REPO: str = os.environ.get(
        "FLUX2_REPO", str(_FLUX2_ROOT / "flux2")
    )
    FLUX2_MODEL_PATH: str = os.environ.get(
        "FLUX2_MODEL_PATH", str(_FLUX2_ROOT / "models" / "flux2-dev.safetensors")
    )
    FLUX2_AE_MODEL_PATH: str = os.environ.get(
        "FLUX2_AE_MODEL_PATH", str(_FLUX2_ROOT / "models" / "ae.safetensors")
    )
    FLUX2_PYTHON: str = os.environ.get(
        "FLUX2_PYTHON", str(_HIDREAM_ROOT / ".venv" / "bin" / "python")
    )
    FLUX2_WIDTH: int = int(os.environ.get("FLUX2_WIDTH", "1360"))
    FLUX2_HEIGHT: int = int(os.environ.get("FLUX2_HEIGHT", "768"))
    FLUX2_STEPS: int = int(os.environ.get("FLUX2_STEPS", "50"))
    FLUX2_GUIDANCE: float = float(os.environ.get("FLUX2_GUIDANCE", "4"))
    FLUX2_TIMEOUT_SECONDS: int = int(
        os.environ.get("FLUX2_TIMEOUT_SECONDS", "2700")
    )
    FLUX2_KLEIN_MODEL_PATH: str = os.environ.get(
        "FLUX2_KLEIN_MODEL_PATH",
        str(_FLUX2_ROOT / "models" / "klein-4b-nvfp4" / "flux-2-klein-4b-nvfp4.safetensors"),
    )
    FLUX2_KLEIN_PYTHON: str = os.environ.get(
        "FLUX2_KLEIN_PYTHON",
        str(_HIDREAM_ROOT / ".venv" / "bin" / "python"),
    )
    QWEN_MODEL_PATH: str = os.environ.get(
        "QWEN_MODEL_PATH", str(_QWEN_ROOT / "Qwen-Image-Edit-2511")
    )
    QWEN_PYTHON: str = os.environ.get(
        "QWEN_PYTHON", str(_QWEN_ROOT / ".venv" / "bin" / "python")
    )
    QWEN_WIDTH: int = int(os.environ.get("QWEN_WIDTH", "1024"))
    QWEN_HEIGHT: int = int(os.environ.get("QWEN_HEIGHT", "1024"))
    QWEN_STEPS: int = int(os.environ.get("QWEN_STEPS", "40"))
    QWEN_GUIDANCE: float = float(os.environ.get("QWEN_GUIDANCE", "4.0"))
    QWEN_TIMEOUT_SECONDS: int = int(
        os.environ.get("QWEN_TIMEOUT_SECONDS", "2700")
    )
    RUNPOD_API_KEY: str = os.environ.get("RUNPOD_API_KEY", "").strip()
    QWEN_RUNPOD_ENABLED: bool = _parse_bool(
        os.environ.get("QWEN_RUNPOD_ENABLED"),
        bool(RUNPOD_API_KEY),
    )
    QWEN_RUNPOD_ENDPOINT: str = os.environ.get(
        "QWEN_RUNPOD_ENDPOINT",
        "https://api.runpod.ai/v2/qwen-image-edit-2511/runsync",
    ).strip()
    QWEN_RUNPOD_SIZE: str = os.environ.get("QWEN_RUNPOD_SIZE", "1024*1024").strip()
    QWEN_RUNPOD_PRICE_USD_PER_IMAGE: float = float(
        os.environ.get("QWEN_RUNPOD_PRICE_USD_PER_IMAGE", "0.02")
    )
    QWEN_RUNPOD_REQUEST_TIMEOUT_SECONDS: int = int(
        os.environ.get("QWEN_RUNPOD_REQUEST_TIMEOUT_SECONDS", "180")
    )
    QWEN_RUNPOD_MAX_RESULT_BYTES: int = int(
        os.environ.get("QWEN_RUNPOD_MAX_RESULT_BYTES", str(32 * 1024 * 1024))
    )
    RUNPOD_FLUX1_DEV_ENABLED: bool = _parse_bool(
        os.environ.get("RUNPOD_FLUX1_DEV_ENABLED"),
        bool(RUNPOD_API_KEY),
    )
    RUNPOD_FLUX1_DEV_ENDPOINT: str = os.environ.get(
        "RUNPOD_FLUX1_DEV_ENDPOINT",
        "https://api.runpod.ai/v2/black-forest-labs-flux-1-dev/runsync",
    ).strip()
    RUNPOD_FLUX1_DEV_WIDTH: int = int(
        os.environ.get("RUNPOD_FLUX1_DEV_WIDTH", "1024")
    )
    RUNPOD_FLUX1_DEV_HEIGHT: int = int(
        os.environ.get("RUNPOD_FLUX1_DEV_HEIGHT", "1024")
    )
    RUNPOD_FLUX1_DEV_STEPS: int = int(
        os.environ.get("RUNPOD_FLUX1_DEV_STEPS", "28")
    )
    RUNPOD_FLUX1_DEV_GUIDANCE: float = float(
        os.environ.get("RUNPOD_FLUX1_DEV_GUIDANCE", "7.5")
    )
    RUNPOD_FLUX1_DEV_PRICE_USD_PER_MEGAPIXEL: float = float(
        os.environ.get("RUNPOD_FLUX1_DEV_PRICE_USD_PER_MEGAPIXEL", "0.02")
    )
    RUNPOD_FLUX1_DEV_REQUEST_TIMEOUT_SECONDS: int = int(
        os.environ.get("RUNPOD_FLUX1_DEV_REQUEST_TIMEOUT_SECONDS", "180")
    )
    RUNPOD_FLUX1_DEV_MAX_RESULT_BYTES: int = int(
        os.environ.get(
            "RUNPOD_FLUX1_DEV_MAX_RESULT_BYTES", str(32 * 1024 * 1024)
        )
    )
    _GEMINI_PERSONAL_KEY: str = os.environ.get("GEMINI_API_KEY", "").strip()
    _GEMINI_MANAGED_KEY: str = os.environ.get(
        "AI_INTEGRATIONS_GEMINI_API_KEY", ""
    ).strip()
    _GEMINI_PERSONAL_BASE_URL: str = os.environ.get(
        "GEMINI_BASE_URL", ""
    ).strip()
    _GEMINI_MANAGED_BASE_URL: str = os.environ.get(
        "AI_INTEGRATIONS_GEMINI_BASE_URL", ""
    ).strip()
    GEMINI_IMAGE_ENABLED: bool = _parse_bool(
        os.environ.get("GEMINI_IMAGE_ENABLED"),
        bool(
            (_GEMINI_PERSONAL_KEY and (_GEMINI_PERSONAL_BASE_URL or True))
            or (_GEMINI_MANAGED_BASE_URL and _GEMINI_MANAGED_KEY)
        ),
    )
    GEMINI_IMAGE_BASE_URL: str = (
        _GEMINI_PERSONAL_BASE_URL
        or ("https://generativelanguage.googleapis.com" if _GEMINI_PERSONAL_KEY else "")
        or _GEMINI_MANAGED_BASE_URL
    )
    GEMINI_IMAGE_API_KEY: str = _GEMINI_PERSONAL_KEY or _GEMINI_MANAGED_KEY
    GEMINI_IMAGE_MODEL: str = os.environ.get(
        "GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"
    ).strip()
    GEMINI_IMAGE_REQUEST_TIMEOUT_SECONDS: int = int(
        os.environ.get("GEMINI_IMAGE_REQUEST_TIMEOUT_SECONDS", "180")
    )
    GEMINI_IMAGE_MAX_RESULT_BYTES: int = int(
        os.environ.get("GEMINI_IMAGE_MAX_RESULT_BYTES", str(32 * 1024 * 1024))
    )
    FLUX_SCHNELL_MODEL_PATH: str = os.environ.get(
        "FLUX_SCHNELL_MODEL_PATH",
        str(_FLUX_SCHNELL_ROOT / "FLUX.1-schnell"),
    )
    FLUX_SCHNELL_PYTHON: str = os.environ.get(
        "FLUX_SCHNELL_PYTHON", str(_FLUX_SCHNELL_ROOT / ".venv" / "bin" / "python")
    )
    FLUX_SCHNELL_WIDTH: int = int(os.environ.get("FLUX_SCHNELL_WIDTH", "1024"))
    FLUX_SCHNELL_HEIGHT: int = int(os.environ.get("FLUX_SCHNELL_HEIGHT", "1024"))
    FLUX_SCHNELL_STEPS: int = int(os.environ.get("FLUX_SCHNELL_STEPS", "4"))
    FLUX_SCHNELL_STRENGTH: float = float(
        os.environ.get("FLUX_SCHNELL_STRENGTH", "0.55")
    )
    FLUX_SCHNELL_TIMEOUT_SECONDS: int = int(
        os.environ.get("FLUX_SCHNELL_TIMEOUT_SECONDS", "1800")
    )
    SDXL_MODEL_PATH: str = os.environ.get(
        "SDXL_MODEL_PATH", str(_SDXL_ROOT / "StableDiffusionXL")
    )
    SDXL_PYTHON: str = os.environ.get(
        "SDXL_PYTHON", str(_HIDREAM_ROOT / ".venv" / "bin" / "python")
    )
    SDXL_WIDTH: int = int(os.environ.get("SDXL_WIDTH", "512"))
    SDXL_HEIGHT: int = int(os.environ.get("SDXL_HEIGHT", "512"))
    SDXL_FAST_WIDTH: int = int(os.environ.get("SDXL_FAST_WIDTH", "256"))
    SDXL_FAST_HEIGHT: int = int(os.environ.get("SDXL_FAST_HEIGHT", "256"))
    SDXL_FAST_HUMAN_WIDTH: int = int(os.environ.get("SDXL_FAST_HUMAN_WIDTH", "512"))
    SDXL_FAST_HUMAN_HEIGHT: int = int(os.environ.get("SDXL_FAST_HUMAN_HEIGHT", "512"))
    SDXL_STEPS: int = int(os.environ.get("SDXL_STEPS", "12"))
    SDXL_CPU_STEPS: int = int(os.environ.get("SDXL_CPU_STEPS", "2"))
    SDXL_FAST_HUMAN_STEPS: int = int(os.environ.get("SDXL_FAST_HUMAN_STEPS", "8"))
    SDXL_CPU_THREADS: int = int(
        os.environ.get("SDXL_CPU_THREADS", str(min(os.cpu_count() or 4, 4)))
    )
    SDXL_GUIDANCE: float = float(os.environ.get("SDXL_GUIDANCE", "7.0"))
    SDXL_STRENGTH: float = float(os.environ.get("SDXL_STRENGTH", "0.62"))
    SDXL_HUMAN_STRENGTH: float = float(
        os.environ.get("SDXL_HUMAN_STRENGTH", "0.48")
    )
    SDXL_FAST_HUMAN_STRENGTH: float = float(
        os.environ.get("SDXL_FAST_HUMAN_STRENGTH", "0.20")
    )
    SDXL_ALLOW_CPU_GENERATION: bool = os.environ.get(
        "SDXL_ALLOW_CPU_GENERATION", "true"
    ).lower() in {"1", "true", "yes", "on"}
    SDXL_TIMEOUT_SECONDS: int = int(
        os.environ.get("SDXL_TIMEOUT_SECONDS", "1800")
    )
    FOOOCUS_ROOT: str = os.environ.get("FOOOCUS_ROOT", str(_FOOOCUS_ROOT))
    FOOOCUS_PYTHON: str = os.environ.get(
        "FOOOCUS_PYTHON", str(_FOOOCUS_ROOT / ".venv" / "bin" / "python")
    )
    GENERATION_VERIFICATION_REPORT_PATH: str = os.environ.get(
        "GENERATION_VERIFICATION_REPORT_PATH",
        str(_VALIDATION_ROOT / "human-product-report.json"),
    )
    GENERATION_VERIFICATION_TTL_HOURS: int = int(
        os.environ.get("GENERATION_VERIFICATION_TTL_HOURS", "168")
    )
    HUMAN_PRODUCT_VALIDATOR_MODEL_PATH: str = os.environ.get(
        "HUMAN_PRODUCT_VALIDATOR_MODEL_PATH",
        str(_VALIDATION_ROOT / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"),
    )
    HUMAN_PRODUCT_PERSON_SCORE: float = float(
        os.environ.get("HUMAN_PRODUCT_PERSON_SCORE", "0.72")
    )
    HUMAN_PRODUCT_BAG_SCORE: float = float(
        os.environ.get("HUMAN_PRODUCT_BAG_SCORE", "0.45")
    )
    HUMAN_PRODUCT_IDENTITY_SCORE: float = float(
        os.environ.get("HUMAN_PRODUCT_IDENTITY_SCORE", "0.42")
    )
    WORKER_URL: str = os.environ.get("COLAB_WORKER_URL", "").strip()
    WORKER_TOKEN: str = os.environ.get("COLAB_WORKER_TOKEN", "").strip()
    WORKER_TIMEOUT_SECONDS: int = int(os.environ.get("COLAB_WORKER_TIMEOUT_SECONDS", "45"))
    WORKER_JOB_TIMEOUT_SECONDS: int = int(os.environ.get("COLAB_WORKER_JOB_TIMEOUT_SECONDS", "1800"))
    WORKER_POLL_SECONDS: float = float(os.environ.get("COLAB_WORKER_POLL_SECONDS", "3"))
    # Paid FLUX.2 Pro runs through the attached Replit fal.ai connector. There
    # is deliberately no provider API key setting in this backend.
    FAL_FLUX2_PRO_ENABLED: bool = _parse_bool(
        os.environ.get("FAL_FLUX2_PRO_ENABLED"),
        bool(os.environ.get("REPLIT_CONNECTORS_HOSTNAME")),
    )
    FAL_FLUX2_PRO_CONNECTOR_NAME: str = os.environ.get(
        "FAL_FLUX2_PRO_CONNECTOR_NAME", "falai"
    ).strip()
    FAL_FLUX2_PRO_BRIDGE: str = os.environ.get(
        "FAL_FLUX2_PRO_BRIDGE",
        str(_WORKSPACE_ROOT / "backend" / "scripts" / "fal_connector_bridge.mjs"),
    )
    FAL_FLUX2_PRO_NODE: str = os.environ.get("FAL_FLUX2_PRO_NODE", "node")
    FAL_FLUX2_PRO_REQUEST_TIMEOUT_SECONDS: int = int(
        os.environ.get("FAL_FLUX2_PRO_REQUEST_TIMEOUT_SECONDS", "45")
    )
    FAL_FLUX2_PRO_JOB_TIMEOUT_SECONDS: int = int(
        os.environ.get("FAL_FLUX2_PRO_JOB_TIMEOUT_SECONDS", "900")
    )
    FAL_FLUX2_PRO_POLL_SECONDS: float = float(
        os.environ.get("FAL_FLUX2_PRO_POLL_SECONDS", "2")
    )
    FAL_FLUX2_PRO_MAX_RESULT_BYTES: int = int(
        os.environ.get("FAL_FLUX2_PRO_MAX_RESULT_BYTES", str(32 * 1024 * 1024))
    )
    # Black Forest Labs is intentionally a separate provider from the
    # Replit-attached MCP and fal.ai routes. The API key stays server-side.
    BFL_FLUX2_ENABLED: bool = _parse_bool(
        os.environ.get("BFL_FLUX2_ENABLED"),
        bool(os.environ.get("BFL_API_KEY")),
    )
    BFL_API_KEY: str = os.environ.get("BFL_API_KEY", "").strip()
    BFL_API_BASE_URL: str = os.environ.get(
        "BFL_API_BASE_URL", "https://api.bfl.ai/v1"
    ).rstrip("/")
    BFL_FLUX2_MODEL_ENDPOINT: str = os.environ.get(
        "BFL_FLUX2_MODEL_ENDPOINT", "flux-2-pro"
    ).strip("/")
    BFL_FLUX2_REQUEST_TIMEOUT_SECONDS: int = int(
        os.environ.get("BFL_FLUX2_REQUEST_TIMEOUT_SECONDS", "45")
    )
    BFL_FLUX2_JOB_TIMEOUT_SECONDS: int = int(
        os.environ.get("BFL_FLUX2_JOB_TIMEOUT_SECONDS", "900")
    )
    BFL_FLUX2_POLL_SECONDS: float = float(
        os.environ.get("BFL_FLUX2_POLL_SECONDS", "2")
    )
    BFL_FLUX2_MAX_RESULT_BYTES: int = int(
        os.environ.get("BFL_FLUX2_MAX_RESULT_BYTES", str(32 * 1024 * 1024))
    )
    # RunPod FLUX image generation bridge (server-side proxy, secret URL).
    RUNPOD_URL: str = os.environ.get("RUNPOD_URL", "").strip()
    RUNPOD_GENERATE_ENABLED: bool = _parse_bool(
        os.environ.get("RUNPOD_GENERATE_ENABLED"),
        bool(os.environ.get("RUNPOD_URL", "").strip()),
    )
    RUNPOD_GENERATE_REQUEST_TIMEOUT_SECONDS: int = int(
        os.environ.get("RUNPOD_GENERATE_REQUEST_TIMEOUT_SECONDS", "130")
    )
    RUNPOD_GENERATE_MAX_RESULT_BYTES: int = int(
        os.environ.get("RUNPOD_GENERATE_MAX_RESULT_BYTES", str(32 * 1024 * 1024))
    )


settings = Settings()
