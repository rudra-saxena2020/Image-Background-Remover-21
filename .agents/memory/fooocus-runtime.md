---
name: Fooocus runtime
description: Constraints for using the official Fooocus checkout as an Atelier generation backend
---

Fooocus must be treated as a CUDA-only backend in Atelier. On CPU-only hosts, keep the official checkout and isolated Python 3.10 environment available for later provisioning, but do not install checkpoints, start inference, or route Auto jobs to it.

**Why:** Fooocus's normal SDXL inference is not a viable CPU production path, and installing its full dependency set can consume substantial workspace quota without making generation usable.

**How to apply:** Gate readiness on the isolated runtime, local checkpoints, and `torch.cuda.is_available()`. A missing NVIDIA device should produce an explicit unavailable status, not a silent CPU fallback.