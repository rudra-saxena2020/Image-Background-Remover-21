---
name: Campaign generation policy
description: Atelier campaigns must be product-aware, visually diverse, identity-locked, and validated before completion.
---

Atelier generation should treat the reference product as immutable, build a BiRefNet-derived identity profile, generate human/scene plates separately, composite the profiled source layer, and reject repeated or visibly defective frames before export.

**Why:** Repeated generic prompts produced near-identical frames, and prompt-only generation cannot preserve product identity; source-layer compositing plus pixel-level identity checking is the reliable boundary, while category-specific direction and a strict retry/validation gate protect the campaign.

**How to apply:** Preserve the category-aware prompt rules, product/model identity locks, source-layer pixel gate, frame diversity checks, and automatic regeneration behavior when changing shoot generation. CPU cutout compositing cannot satisfy human-model categories; require a current verified local human-scene backend instead of silently falling back or using a hosted provider. Product-only shots may continue using the clearest source cutout. FLUX.2 Dev's gated 60 GB checkpoint may need to live on a larger CUDA machine than the development workspace.

For a source-preserved human reference that already contains the model carrying the product, do not overlay a second cutout; preserve the validated reference frame verbatim and label it source-preserved.

**Why:** A detector can accept a duplicated-bag composite even though the result visibly violates the one-product requirement.

**How to apply:** Only controlled-composite a product layer onto a scene plate that does not already contain the product. If the uploaded model-carrying reference is the accepted plate, skip the overlay.

Rudras is the product-facing name for the local backend router, not a merged checkpoint: it selects the strongest ready local engine in priority order and never routes to a hosted provider.

**Why:** Qwen, FLUX, HiDream, FLUX.2, Klein and SDXL use incompatible architectures, and this CPU workspace cannot honestly expose a single merged model.

**How to apply:** Keep `auto` as a backwards-compatible API alias, expose Rudras in the UI, report the resolved backend, and keep unavailable engines unavailable rather than pretending the router can use them.