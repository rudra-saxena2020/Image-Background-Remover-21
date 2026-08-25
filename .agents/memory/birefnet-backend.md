---
name: BiRefNet Python backend setup
description: Packages required beyond the obvious to run BiRefNet with transformers on this workspace
---

BiRefNet via `AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)` requires:

- `torch` + `torchvision` — CPU-only builds work (`torch==2.13.0+cpu`)
- `transformers`, `huggingface_hub`, `accelerate`, `safetensors` — install via `python3 -m pip install` (not installLanguagePackages — uv has version resolution issues)
- `einops`, `kornia`, `timm` — NOT listed in transformers deps but required by BiRefNet's custom model code; missing causes ImportError at startup

**Why:** BiRefNet uses `trust_remote_code=True` which downloads birefnet.py from HuggingFace and imports these packages dynamically. They're not declared in transformers' dependency graph.

**How to apply:** Whenever BiRefNet is set up, always install all three: `python3 -m pip install einops kornia timm`

Model download ~441MB on first run. CPU warmup has dtype mismatch on the FP16 check — warmup is non-fatal, model still loads correctly.
