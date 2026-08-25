---
name: Single-view image-edit prompting
description: Preventing image-edit models from copying multi-view reference layouts
---

Image-edit requests must remove campaign-level angle lists and set language before submission. A collage or labeled multi-view reference should be treated as identity-only input, while the request contains one explicit camera direction and one-image output constraint.

**Why:** Image-edit models can prioritize the visual composition of a collage or the phrase “complete set,” producing another collage even when later text says not to.

**How to apply:** Build each batch item as an independent single-view prompt; sanitize user/preset text that requests multiple views, and do not reproduce reference panels, labels, or layout.

For standalone batches, use the first uploaded or imported image as the identity master rather than merging multiple references; additional references can introduce colorways, accessories, or alternate products.

**Why:** Even images selected from one catalog record may contain different variants, and Qwen can blend them into a new product.

**How to apply:** Keep the master-reference behavior explicit in the UI and send only that image for every angle request.

Output validation must inspect short panoramic images too; do not skip quality checks based only on pixel height.

**Why:** A contact sheet can be returned as a wide, short image and bypass a validator that only analyzes outputs above a square-like minimum size.

**How to apply:** Reject strongly panoramic generated outputs as likely collages before gallery insertion, then run connected-component checks on all reasonably sized images.