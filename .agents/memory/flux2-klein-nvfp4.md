---
name: FLUX.2 Klein NVFP4 runtime
description: The official lightweight Klein checkpoint is installed, but its single-file NVFP4 format needs a compatible CUDA runner.
---

The official Apache-2.0 FLUX.2 Klein 4B NVFP4 checkpoint is suitable for the lightweight local engine, but the current CPU workspace cannot execute it. The installed Diffusers/torchao versions can import the general Klein pipeline but do not directly load this official single-file NVFP4 checkpoint, and no NVIDIA CUDA device is available.

**Why:** The smaller checkpoint avoids the large FLUX.2 Dev download, but falsely marking it ready would make campaign requests fail at runtime.

**How to apply:** Keep FLUX.2 Klein separate from FLUX.2 Dev and report checkpoint presence, CUDA availability, and NVFP4 runtime support independently. Keep SDXL as the verified CPU fallback until a compatible CUDA/NVFP4 runner is installed and tested with reference-conditioned editing.