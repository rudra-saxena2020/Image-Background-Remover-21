---
name: Deployed Colab worker contract
description: The live Atelier Colab worker may expose a legacy product_id/source_images request schema while still using the shipped route and identity contract.
---

The Replit bridge must inspect the worker OpenAPI schema and translate between the structured `references` payload and the deployed `product_id` plus data-URL `source_images` payload when necessary.

**Why:** The public worker can be updated independently of Replit, and assuming only the newest request schema caused real verification requests to fail with HTTP 422 even though authentication and required routes passed.

**How to apply:** Keep identity and verification fail-closed; schema translation is compatibility only and must not turn `NOT_VERIFIED`, empty models, or failed validation into readiness.

Worker readiness is a conjunction of CUDA reachability, selected-provider configuration, successful provider model load, a readable provider inference output, and fresh human/product verification. Model metadata alone is never enough.

**Why:** A live Colab runtime can have a GPU and downloaded files while its provider is empty or inference is broken; reporting those as available makes the studio route campaign jobs into a dead worker.

**How to apply:** Preserve separate diagnostics and next actions for empty configuration, model-load failure, inference failure, inference-ready/unverified, and fully verified states in both `/health` and the Replit bridge.

All AI campaign choices are Colab-only. A selected model is sent as a requested
provider identifier to the authenticated worker; CPU is reserved for the
source-preserved reference preview and local preprocessing.

**Why:** Falling back to a local runtime makes the selected engine misleading
and bypasses the worker's provider and verification checks.

**How to apply:** Do not restore local generation branches or warm-up jobs in
shoot orchestration. Reject AI jobs until the Colab worker is ready, and reject
unsupported requested providers at the worker instead of silently substituting
another model.