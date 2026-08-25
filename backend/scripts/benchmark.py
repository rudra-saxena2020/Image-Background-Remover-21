"""
Benchmark BiRefNet processing performance.

Usage:
    python scripts/benchmark.py --dir /path/to/test/images --quality fast
    python scripts/benchmark.py --dir /path/to/test/images --quality high
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def run_benchmark(image_dir: str, quality: str, warmup: bool = True) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.services.birefnet_service import birefnet

    logger.info("Loading model …")
    birefnet.load()
    logger.info("Model loaded")

    paths = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in SUPPORTED]
    if not paths:
        logger.error("No images found in %s", image_dir)
        sys.exit(1)

    logger.info("Found %d images", len(paths))

    inference_times: list[float] = []
    total_times: list[float] = []
    failures = 0
    peak_vram_mb = 0.0

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    for path in paths:
        logger.info("Processing %s …", path.name)
        data = path.read_bytes()
        t_start = time.perf_counter()
        try:
            _, meta = birefnet.remove_background(data, quality)
            t_total = (time.perf_counter() - t_start) * 1000
            inference_times.append(meta["inference_time_ms"])
            total_times.append(t_total)
            logger.info(
                "  inference=%.0fms  total=%.0fms",
                meta["inference_time_ms"],
                t_total,
            )
            if has_cuda:
                try:
                    import torch
                    vram = torch.cuda.max_memory_allocated() / 1024**2
                    peak_vram_mb = max(peak_vram_mb, vram)
                except Exception:
                    pass
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            failures += 1

    if not inference_times:
        logger.error("All images failed")
        sys.exit(1)

    arr_inf = np.array(inference_times)
    arr_tot = np.array(total_times)

    logger.info("\n═══ Benchmark Results (quality=%s) ═══", quality)
    logger.info("Images processed : %d / %d", len(inference_times), len(paths))
    logger.info("Failures         : %d", failures)
    logger.info("")
    logger.info("Inference time:")
    logger.info("  Mean : %.0f ms", arr_inf.mean())
    logger.info("  P50  : %.0f ms", np.percentile(arr_inf, 50))
    logger.info("  P95  : %.0f ms", np.percentile(arr_inf, 95))
    logger.info("")
    logger.info("Total time (incl. encode):")
    logger.info("  Mean : %.0f ms", arr_tot.mean())
    logger.info("  P50  : %.0f ms", np.percentile(arr_tot, 50))
    logger.info("  P95  : %.0f ms", np.percentile(arr_tot, 95))
    logger.info("")
    logger.info(
        "Throughput       : %.2f images/sec",
        1000.0 / arr_tot.mean() if arr_tot.mean() > 0 else 0,
    )
    if peak_vram_mb > 0:
        logger.info("Peak VRAM        : %.0f MB", peak_vram_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiRefNet benchmark")
    parser.add_argument("--dir", required=True, help="Directory of test images")
    parser.add_argument("--quality", choices=["fast", "high"], default="fast")
    args = parser.parse_args()
    run_benchmark(args.dir, args.quality)
