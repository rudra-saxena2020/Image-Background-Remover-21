from fastapi import APIRouter
from app.services.birefnet_service import birefnet
from app.services.flux2_generation_service import status as flux2_generation_status
from app.services.flux2_klein_generation_service import status as flux2_klein_generation_status
from app.services.flux_schnell_generation_service import status as flux_schnell_generation_status
from app.services.fooocus_generation_service import status as fooocus_generation_status
from app.services.local_generation_service import status as local_generation_status
from app.services.qwen_image_edit_service import status as qwen_generation_status
from app.services.sdxl_generation_service import status as sdxl_generation_status
from app.services.generation_verification_service import verified_status
from app.services.human_product_validation_service import validator_status
from app.services.remote_worker_service import status as remote_worker_status
from app.services.fal_flux2_pro_service import status as fal_flux2_pro_status
from app.services.black_forest_flux2_service import status as black_forest_flux2_status
from app.services.runpod_qwen_image_edit_service import status as runpod_qwen_status
from app.services.runpod_flux1_dev_service import status as runpod_flux1_dev_status
from app.services.gemini_image_service import status as gemini_image_status
from app.services.model_registry import decorate, reference_preview_status

router = APIRouter()


@router.get("")
async def api_root_health():
    """Lightweight readiness response for the mounted /api artifact path."""
    return {"status": "ok"}


def available_generation_engines(
    generation: dict[str, dict],
) -> list[dict[str, object]]:
    """Expose only providers that passed the current generation contract."""
    available: list[dict[str, object]] = []
    flux = generation.get("flux2_pro") or {}
    if flux.get("ready") is True:
        available.append(
            {
                "id": "flux2-pro",
                "label": "FLUX.2 Pro · fal.ai",
                "provider": flux.get("provider") or "fal.ai",
                "status": flux,
            }
        )
    black_forest = generation.get("black_forest_flux2") or {}
    if black_forest.get("ready") is True:
        available.append(
            {
                "id": "bfl-flux2",
                "label": "FLUX.2 Pro · Black Forest",
                "provider": black_forest.get("provider") or "Black Forest Labs",
                "status": black_forest,
            }
        )
    runpod_qwen = generation.get("qwen_runpod") or {}
    if runpod_qwen.get("ready") is True:
        available.append(
            {
                "id": "qwen-runpod",
                "label": "RunPod image generator · Qwen Edit",
                "provider": runpod_qwen.get("provider") or "RunPod",
                "status": runpod_qwen,
            }
        )
    runpod_flux1_dev = generation.get("flux1_dev_runpod") or {}
    if runpod_flux1_dev.get("ready") is True:
        available.append(
            {
                "id": "flux1-runpod",
                "label": "FLUX.1 Dev · RunPod",
                "provider": runpod_flux1_dev.get("provider") or "RunPod",
                "status": runpod_flux1_dev,
            }
        )
    gemini = generation.get("gemini_image") or {}
    if gemini.get("ready") is True:
        available.append(
            {
                "id": "gemini-image",
                "label": "Gemini image generation",
                "provider": gemini.get("provider") or "Google Gemini",
                "status": gemini,
            }
        )
    colab = generation.get("remote_worker") or {}
    if colab.get("ready") is True:
        available.append(
            {
                "id": "colab",
                "label": f"{colab.get('provider') or 'Verified Colab'} · Colab",
                "provider": colab.get("provider"),
                "status": colab,
            }
        )
    return available


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/health")
async def model_health():
    status = birefnet.status()
    generation = {
        "qwen": verified_status("qwen", qwen_generation_status()),
        "flux_schnell": verified_status(
            "flux-schnell", flux_schnell_generation_status()
        ),
        "fooocus": verified_status("fooocus", fooocus_generation_status()),
        "hidream": verified_status("hidream", local_generation_status()),
        "flux2": verified_status("flux2", flux2_generation_status()),
        "flux2_pro": fal_flux2_pro_status(),
        "black_forest_flux2": black_forest_flux2_status(),
        "qwen_runpod": runpod_qwen_status(),
        "flux1_dev_runpod": runpod_flux1_dev_status(),
        "gemini_image": gemini_image_status(),
        "flux2_klein": verified_status(
            "flux2-klein", flux2_klein_generation_status()
        ),
        "sdxl": verified_status("sdxl", sdxl_generation_status()),
    }
    decorated_remote = decorate("colab", remote_worker_status())
    return {
        "status": "ready" if status["model_loaded"] else "loading",
        **status,
        "queue_size": 0,
        "generation": {
            **generation,
            "remote_worker": decorated_remote,
            "available_engines": available_generation_engines(
                {**generation, "remote_worker": decorated_remote}
            ),
        },
        "human_product_validator": validator_status(),
        "remote_worker": decorate("colab", remote_worker_status()),
        "reference_preview": reference_preview_status(),
    }
