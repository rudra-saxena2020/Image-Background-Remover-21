---
name: Local HiDream runtime
description: Atelier generation is intentionally local-only and requires the official HiDream checkout, weights, and CUDA.
---

Atelier must not fall back to hosted image providers. BiRefNet can run locally on CPU, but campaign generation requires a CUDA-capable machine with a configured local image-to-image model such as HiDream-O1-Image, Qwen Image Edit 2511, or FLUX.1 Schnell. One reference uses an aspect-preserving edit path; multiple references use the provider's multi-reference capability where supported.

**Why:** The user explicitly wants generation on their own system with no paid per-image service.

**How to apply:** Keep references local, use the official runner for the selected provider, expose readiness through the health endpoint, and fail clearly when CUDA or model assets are missing. CPU compositing must remain explicitly product-only.

The Replit development host has no NVIDIA device, so the local runtime must report `ready=false` even when the official repository, all model shards, and an isolated CUDA-enabled PyTorch environment are installed. Hugging Face's multi-file snapshot downloader can exceed the host's memory/quota; download the eight weight shards one at a time.

**Why:** The host has CPU-only hardware and limited memory, while the Dev snapshot is about 35 GB across eight safetensors shards.

**How to apply:** Use the isolated `.local/hidream/.venv`, point `HIDREAM_PYTHON` at it, validate the model index before marking weights present, and use `scripts/install-hidream.sh` for resumable low-memory setup.

SDXL's CLIP text encoder truncates prompts beyond 77 tokens; long campaign policies must be compacted into a model-specific prompt before SDXL inference or the shot direction is silently lost. On CPU, keep SDXL loaded in a persistent worker because reloading the checkpoint per frame makes a campaign impractically slow.

**Why:** The local SDXL smoke output improved when the shot direction was kept inside the encoder limit, and the API's first campaign run showed model reload cost dominating CPU generation time.

**How to apply:** Keep the full identity policy for Qwen/FLUX-capable engines, but pass a concise SDXL prompt and reuse one local SDXL worker for sequential frames.