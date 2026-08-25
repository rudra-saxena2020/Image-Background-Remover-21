---
name: BFL generation constraints
description: Black Forest Labs FLUX.2 Pro input-size and moderation behavior for product-image campaigns
---

For FLUX.2 Pro, the combined megapixels of reference images and requested output must remain below the provider's 9MP limit. Full-size reference uploads can consume most of the budget, so use a compact subset of product views and lower output dimensions when needed.

**Why:** A campaign request with eight full-size references was rejected before generation because the combined input/output size exceeded the provider limit.

**How to apply:** Prefer three user-provided product views plus at most one compact official reference, and keep campaign outputs near 1–1.5MP unless more resolution is necessary.

Brand names and catalog identifiers in prompts or references may trigger protected-content moderation. Neutral visible-description prompts can be accepted, while still using the user's uploaded product references.

**Why:** A prompt naming a branded product and official catalog references was blocked as protected content, while a neutral construction-based description using the uploaded references was accepted.

**How to apply:** If moderation blocks a legitimate product campaign, remove brand/catalog wording and describe only visible materials, dimensions, hardware, and construction; do not claim the resulting output is identity-verified without running Atelier validation.