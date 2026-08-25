---
name: CPU SDXL fast path
description: Readiness and speed constraints for Atelier's local SDXL fallback on the CPU-only development host.
---

The SDXL fallback needs two distinct health concepts: the local model/runtime can be configured, while the persistent worker is still loading. The API and UI must use the worker-ready signal before accepting a generation request; otherwise the first request waits behind model loading and looks unpredictably slow.

**Why:** On this CPU-only host, startup warm-up takes about a minute and a cold request can spend that time behind the worker lock. The worker is the only practical way to keep sequential campaign frames from reloading SDXL.

**How to apply:** Keep warm-up in the background, expose `worker_ready` separately from configured/model readiness, block fast preview until it is true, and treat the separate CPU compositor as product-only.

The interactive CPU preview uses a deliberately small generation and segmentation tier, then upscales to the 512px commercial minimum and still runs local background removal plus transparency/visual validation. Full campaign mode keeps the heavier settings and eight-frame validation.

**Why:** The tested CPU host reaches roughly the target interactive window only with the small fast tier; increasing preview resolution or segmentation size pushes total latency beyond the target.

**How to apply:** Keep the fast/campaign distinction visible in the UI and do not describe the fast result as equivalent to the full campaign.

Cancelling an active SDXL generation stops the worker process, so cancellation must immediately schedule a replacement warm-up. Otherwise the next request remains blocked behind a false “warming” state until the API restarts.

**Why:** The CPU worker is intentionally terminated to interrupt an in-progress subprocess; without an explicit recovery task, the configured model remains healthy but `worker_ready` stays false indefinitely.

**How to apply:** Trigger the same single-flight warm-up path at startup, when Auto/SDXL receives a warming request, and when the user cancels an SDXL shoot.

The local SDXL subprocess exposes stage completion rather than per-denoising-step callbacks, so long generation stages need an explicitly approximate client-side progress tick between backend updates.

**Why:** Without an estimate, a CPU human frame can visibly remain at its initial percentage for several minutes even though work is progressing.

**How to apply:** Mark interim UI percentages with `~`, let backend stage percentages override the estimate immediately, and never present the client tick as exact model progress.

Plain CPU SDXL image-to-image is not a reliable exact-identity guarantee for distinctive product hardware. For a known model-carrying source reference, preserve the supplied source and label it as source-preserved rather than misrepresenting a drifted redraw as validated AI output.

**Why:** Repeated real-reference tests kept the material and color but changed the bag’s distinctive closure and hardware even with stronger prompts and lower denoising strength.

**How to apply:** Keep strict identity fallbacks explicit in API verification and UI copy; reserve true generated identity claims for backends with stronger reference-conditioning support.