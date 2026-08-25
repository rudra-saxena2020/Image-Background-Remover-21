# Atelier — AI Luxury Product Photography Studio

Atelier is an AI luxury product photography studio for fashion and e-commerce teams. Upload one to six product references, lock product identity, and queue an eight-frame local HiDream-O1 Image, FLUX.2 Dev, FLUX.2 Klein 4B, or Stable Diffusion XL campaign with BiRefNet preprocessing, live progress, validated output, and a Shopify-oriented ZIP export. The original background-removal utility remains available as a retained tool.

## Run & Operate

- `cd backend && python -m uvicorn main:app --reload --port 8080 --host 0.0.0.0` — run Python FastAPI backend (serves /api)
- `pnpm --filter @workspace/bgremover run dev` — run React frontend
- `pnpm run typecheck` — full TypeScript typecheck
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API client hooks from OpenAPI spec
- `python backend/scripts/benchmark.py --dir /path/to/images --quality fast` — benchmark BiRefNet
- Local HiDream setup (CUDA machine):
  1. `git clone https://github.com/HiDream-ai/HiDream-O1-Image.git /models/HiDream-O1-Image`
  2. `hf download HiDream-ai/HiDream-O1-Image-Dev --local-dir /models/HiDream-O1-Image-Dev`
  3. Set `HIDREAM_REPO=/models/HiDream-O1-Image`, `HIDREAM_MODEL_PATH=/models/HiDream-O1-Image-Dev`, and `HIDREAM_PYTHON=/models/hidream-venv/bin/python`
  4. Restart the API, then confirm `GET /api/health` reports `generation.ready=true`
- Local FLUX.2 Dev setup (CUDA machine):
  1. Clone `https://github.com/black-forest-labs/flux2` and install its native runtime dependencies in the FLUX2 Python environment.
  2. Accept the gated FLUX.2 Dev terms at `https://huggingface.co/black-forest-labs/FLUX.2-dev`.
  3. Download `flux2-dev.safetensors` and `ae.safetensors` into the configured FLUX2 model directory.
  4. Set `FLUX2_REPO`, `FLUX2_MODEL_PATH`, `FLUX2_AE_MODEL_PATH`, and `FLUX2_PYTHON`, then confirm `generation.flux2.ready=true`.
- Local Stable Diffusion XL setup:
  1. Install the SDXL Diffusers files into the configured `SDXL_MODEL_PATH`.
  2. Set `SDXL_MODEL_PATH` and `SDXL_PYTHON`, then confirm `generation.sdxl.ready=true`.
- Local FLUX.2 Klein 4B NVFP4 setup:
  1. Install the official Apache-2.0 `flux-2-klein-4b-nvfp4.safetensors` checkpoint.
  2. Set `FLUX2_KLEIN_MODEL_PATH` and `FLUX2_KLEIN_PYTHON`.
  3. Use a CUDA machine with an NVFP4-compatible runner; this CPU-only repl reports the checkpoint as installed but does not claim it is ready.

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Frontend: React + Vite + Tailwind CSS (`artifacts/bgremover/`)
- Backend: Python 3.11, FastAPI, PyTorch, BiRefNet (HuggingFace `ZhengPeng7/BiRefNet`)
- API codegen: Orval (from `lib/api-spec/openapi.yaml`)

## Where things live

- `backend/` — Python FastAPI backend
  - `main.py` — FastAPI app entry, model loaded at startup
  - `app/services/birefnet_service.py` — BiRefNet model lifecycle (load once, GPU, FP16, warmup)
  - `app/services/image_service.py` — image validation, result caching, session history
  - `app/services/product_identity_service.py` — BiRefNet-derived immutable product profile and scene-only generation prompt
  - `app/services/controlled_composite_service.py` — exact source-layer compositing, placement, shadow, occlusion hook, and pixel identity gate
  - `app/services/inference_queue.py` — GPU semaphore to prevent OOM
  - `app/api/background.py` — POST /api/remove-background, GET /api/history
- `app/api/shoots.py` — POST/GET /api/shoots, local frame serving, cancellation, validated ZIP export, operations metrics
- `app/services/local_generation_service.py` — local-only HiDream runner using the official open-source repository and local model weights
- `app/services/runpod_qwen_image_edit_service.py` — server-side RunPod bridge for Qwen Image Edit 2511
- `app/services/flux2_generation_service.py` — local-only FLUX.2 Dev runner using the official native repository and local model weights
- `app/services/flux2_klein_generation_service.py` — FLUX.2 Klein NVFP4 checkpoint and runtime status
- `app/services/sdxl_generation_service.py` — local-only Stable Diffusion XL image-to-image runner
  - `app/api/health.py` — GET /api/health, GET /api/healthz
  - `app/config.py` — all config from environment variables
  - `scripts/benchmark.py` — latency benchmark script
- `artifacts/bgremover/src/` — React frontend
- `lib/api-spec/openapi.yaml` — OpenAPI spec (source of truth)

## Architecture decisions

- BiRefNet loads **once** at startup via HuggingFace `AutoModelForImageSegmentation`. Never reloaded per-request.
- Fast mode: 1024px inference resolution. High Quality mode: 2048px. Mask is upsampled back to original dimensions.
- All image processing is in-memory (BytesIO) — no disk writes on the hot path.
- GPU semaphore (`MAX_GPU_CONCURRENCY=1`) prevents CUDA OOM from concurrent requests.
- Result cache: SHA256(image + quality + model_id) → cached PNG, configurable TTL.
- Session history stored in-memory (not persisted). No user images stored permanently.
- Atelier references are preprocessed by BiRefNet and kept in a temporary local directory. The uploaded references are passed directly to the official local image-to-image runners; there is no hosted provider.
- Human-model frames now use a scene-only generation prompt and composite the profiled BiRefNet cutout afterward. The final composite must pass both the source-layer pixel identity gate and the existing human/product semantic gate; an unverified backend still fails closed.
- Each shoot contains eight planned frames, polls provider state, validates returned image bytes, and exports a Shopify manifest with the ZIP.

## Product

- Upload product image (JPG/PNG/WEBP up to 20MB)
- Choose Fast (default) or High Quality mode
- BiRefNet runs inference → returns transparent PNG
- Before/after comparison slider, background color preview
- Batch mode: process up to 20 images, download as ZIP
- Session history: re-download any recently processed image
- Atelier studio: one to six references, identity lock, creative direction, eight-frame plan, live progress, cancellation, per-frame review, ZIP export, and operations metrics
- Open-source generation: Apache-2.0 Qwen Image Edit 2511 is the preferred multi-reference engine; Apache-2.0 FLUX.1 Schnell is the fast image-to-image fallback. Both are local-only and require CUDA.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Environment variables (backend)

See `backend/.env.example`. Key vars:
- `DEVICE=auto` — auto-detects CUDA or falls back to CPU
- `USE_FP16=true` — FP16 on CUDA GPUs
- `BIREFNET_MODEL=ZhengPeng7/BiRefNet`
- Fast background removal uses a 256px mask for small inputs, scaling up for larger references; the standalone Cutout Utility requests `fast` by default. Use API `quality=high` only when maximum edge detail is worth the extra latency.
- `MAX_GPU_CONCURRENCY=1` — tune based on VRAM
- `HIDREAM_REPO=/path/to/HiDream-O1-Image` — local checkout of the official HiDream repository
- `HIDREAM_MODEL_PATH=/path/to/HiDream-O1-Image-Dev` — local downloaded model weights
- `HIDREAM_MODEL_TYPE=dev` — use `full` for the full model
- `QWEN_MODEL_PATH=/path/to/Qwen-Image-Edit-2511` and `QWEN_PYTHON=/path/to/qwen/.venv/bin/python` — Apache-2.0 local Qwen Image Edit 2511 weights and runtime
- `RUNPOD_API_KEY` — Replit Secret used server-side by the optional `qwen-runpod` engine
- `QWEN_RUNPOD_ENABLED=true`, `QWEN_RUNPOD_ENDPOINT=https://api.runpod.ai/v2/qwen-image-edit-2511/runsync`, `QWEN_RUNPOD_SIZE=1024*1024` — RunPod Qwen configuration
- `RUNPOD_FLUX1_DEV_ENABLED=true`, `RUNPOD_FLUX1_DEV_ENDPOINT=https://api.runpod.ai/v2/black-forest-labs-flux-1-dev/runsync`, `RUNPOD_FLUX1_DEV_WIDTH=1024`, and `RUNPOD_FLUX1_DEV_HEIGHT=1024` — RunPod FLUX.1 Dev text-to-image configuration
- `FLUX_SCHNELL_MODEL_PATH=/path/to/FLUX.1-schnell` and `FLUX_SCHNELL_PYTHON=/path/to/flux-schnell/.venv/bin/python` — Apache-2.0 fast image-to-image weights and runtime
- `SDXL_HUMAN_STRENGTH=0.48` — reference-preserving human/model transformation for the regular campaign path
- `SDXL_FAST_HUMAN_WIDTH=512`, `SDXL_FAST_HUMAN_HEIGHT=512`, `SDXL_FAST_HUMAN_STEPS=8`, and `SDXL_FAST_HUMAN_STRENGTH=0.20` — the near-reference CPU fast profile for model-carrying frames; product-only fast frames retain the lightweight 256px profile
- `SDXL_ALLOW_CPU_GENERATION=true` — allows the installed local SDXL checkpoint to attempt local image-to-image generation without CUDA. It is not selectable for human campaigns until the current smoke audit passes.
- `FOOOCUS_ROOT=/path/to/.local/fooocus` — official Fooocus checkout and isolated Python 3.10 runtime; `scripts/install-fooocus.sh` skips Fooocus dependencies and checkpoints on CPU-only hosts and completes the CUDA install when run on an NVIDIA worker
- `GENERATION_VERIFICATION_TTL_HOURS=168` — expires backend approval after seven days; model/runtime fingerprint changes invalidate it immediately
- `HUMAN_PRODUCT_VALIDATOR_MODEL_PATH=/path/to/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` — local open-source person/bag detector used by generation and smoke tests

## Gotchas

- `torch` and `transformers` are large — model download on first run takes time (~1GB+)
- Always run `pnpm --filter @workspace/api-spec run codegen` after spec changes
- Do not add `format: binary` to OpenAPI request body schemas — causes TS2304 (`File`/`Blob` not in Node lib)
- The api-client-react `useRemoveBackground` hook is a typing stub only; use native fetch for binary upload/download
- Qwen Image Edit 2511, FLUX.1 Schnell, Fooocus/SDXL, HiDream-O1-Image, FLUX.2 Klein 4B, and Stable Diffusion XL are open-source local options. The optional `qwen-runpod` engine uses the configured RunPod public endpoint and accepts one to three references. FLUX.2 Dev is gated and its open weights are covered by the FLUX Dev license. Qwen, FLUX Schnell, Fooocus, HiDream, and FLUX.2 require CUDA for usable local generation; the installed SDXL fallback can run locally on CPU when `SDXL_ALLOW_CPU_GENERATION=true`.
- Run `scripts/install-open-source-generators.sh` on a CUDA machine to install Qwen Image Edit 2511 and FLUX.1 Schnell into `.local/`. The current Replit development runtime is CPU-only, so it reports GPU-required for the faster open-source engines; the installed SDXL fallback can still run real local CPU image-to-image generation, while the separate CPU edit remains product-only.
- Runtime/model presence is not generation readiness. From `backend/`, run `python scripts/audit_human_product_generation.py --engine all --references <2-6 local product images>` on the target worker. Running from the API directory preserves the same Python environment fingerprint. Rudras only selects engines with a current passing human-with-product result.
- The local validator checks a visible person, detected handbag/backpack/suitcase, physical proximity, believable relative scale, and a conservative visual identity proxy. It cannot certify fine-grained finger anatomy or exact hardware by itself; generated outputs that alter distinctive product construction must remain rejected during visual quality review.
- RunPod usage is session-scoped in the Operations view. Qwen Image Edit 2511 is priced at $0.02 per output image; FLUX.1 Dev is priced at $0.02 per megapixel (about $0.02097 at 1024×1024). Provider-reported response cost is preferred when available.
