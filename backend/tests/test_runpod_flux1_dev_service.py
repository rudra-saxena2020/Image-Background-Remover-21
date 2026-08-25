import asyncio
import unittest
from unittest.mock import patch

from app.services.runpod_flux1_dev_service import (
    estimate_cost,
    generate_frame,
    status,
)


class RunPodFlux1DevServiceTests(unittest.TestCase):
    def test_status_reports_documented_per_megapixel_rate(self):
        result = status()
        pricing = result["pricing"]
        self.assertEqual(pricing["rate_usd"], 0.02)
        self.assertEqual(pricing["estimated_cost_usd"], estimate_cost())

    def test_completed_result_is_downloaded_and_cost_is_preserved(self):
        with patch(
            "app.services.runpod_flux1_dev_service._request",
            return_value={
                "id": "sync-flux-123",
                "status": "COMPLETED",
                "output": {
                    "image_url": "https://image.runpod.ai/example/output.png",
                    "cost": 0.02097152,
                },
            },
        ) as request, patch(
            "app.services.runpod_flux1_dev_service._download_result",
            return_value=b"png-bytes",
        ):
            result = asyncio.run(
                generate_frame(
                    prompt="A precise product photograph.",
                    seed=42,
                )
            )
        self.assertEqual(result.image_bytes, b"png-bytes")
        self.assertEqual(result.request_id, "sync-flux-123")
        self.assertEqual(result.cost, 0.02097152)
        payload = request.call_args.args[0]["input"]
        self.assertEqual(payload["width"], 1024)
        self.assertEqual(payload["height"], 1024)
        self.assertEqual(payload["num_inference_steps"], 28)
        self.assertEqual(payload["guidance"], 7.5)
        self.assertEqual(payload["image_format"], "png")

    def test_completed_result_accepts_runpod_image_key(self):
        with patch(
            "app.services.runpod_flux1_dev_service._request",
            return_value={
                "id": "sync-flux-image-key",
                "status": "COMPLETED",
                "output": {
                    "image": "https://image.runpod.ai/example/output.png",
                },
            },
        ), patch(
            "app.services.runpod_flux1_dev_service._download_result",
            return_value=b"png-bytes",
        ):
            result = asyncio.run(
                generate_frame(prompt="A product photograph.", seed=7)
            )
        self.assertEqual(result.image_bytes, b"png-bytes")
        self.assertEqual(result.result_url, "https://image.runpod.ai/example/output.png")

    def test_completed_result_accepts_inline_base64_image(self):
        with patch(
            "app.services.runpod_flux1_dev_service._request",
            return_value={
                "id": "sync-flux-inline",
                "status": "COMPLETED",
                "output": {"image": "aW1hZ2UtYnl0ZXM="},
            },
        ):
            result = asyncio.run(
                generate_frame(prompt="A product photograph.", seed=8)
            )
        self.assertEqual(result.image_bytes, b"image-bytes")
        self.assertEqual(result.result_url, "")


if __name__ == "__main__":
    unittest.main()