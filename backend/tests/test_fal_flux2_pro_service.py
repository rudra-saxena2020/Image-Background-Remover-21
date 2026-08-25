import asyncio
import base64
import json
import unittest
from unittest.mock import Mock, patch

import app.services.fal_flux2_pro_service as fal_service
from app.services.fal_flux2_pro_service import (
    FalFlux2ProError,
    _data_url_references,
    _request_queue,
    _response_url,
    generate_frame,
)


class FalFlux2ProServiceTests(unittest.TestCase):
    def tearDown(self):
        fal_service._runtime_block_reason = None

    def test_references_are_sent_as_image_data_uris(self):
        urls = _data_url_references([(b"pixels", "bag.jpg", "image/jpeg")])
        self.assertTrue(urls[0].startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(urls[0].split(",", 1)[1]), b"pixels")

    def test_fallback_queue_url_keeps_edit_endpoint(self):
        self.assertEqual(
            _response_url({}, "request-123", "status"),
            "https://queue.fal.run/fal-ai/flux-2-pro/edit/requests/request-123/status",
        )

    def test_exhausted_balance_blocks_retries_and_marks_provider_unavailable(self):
        completed = Mock(
            stdout=json.dumps(
                {
                    "ok": True,
                    "status": 403,
                    "body": '{"detail":"User is locked. Reason: Exhausted balance. Top up your balance at fal.ai/dashboard/billing."}',
                }
            )
        )
        with patch("app.services.fal_flux2_pro_service.subprocess.run", return_value=completed):
            with self.assertRaises(FalFlux2ProError) as context:
                _request_queue("https://queue.fal.run/fal-ai/flux-2-pro/edit", "POST", {})
        self.assertEqual(context.exception.code, "FAL_BALANCE_EXHAUSTED")
        self.assertFalse(context.exception.retryable)
        self.assertFalse(fal_service.status()["ready"])
        self.assertIn("exhausted its balance", str(context.exception))

    def test_completed_queue_downloads_first_provider_image(self):
        queue_calls = []

        def queue_response(url, method="GET", payload=None):
            queue_calls.append((url, method, payload))
            if method == "POST":
                return {
                    "request_id": "request-123",
                    "status_url": "https://queue.fal.run/fal-ai/flux-2-pro/requests/request-123/status",
                    "response_url": "https://queue.fal.run/fal-ai/flux-2-pro/requests/request-123",
                }
            if url.endswith("/status"):
                return {"status": "COMPLETED"}
            return {
                "images": [{"url": "https://v3b.fal.media/atelier/output.png"}],
                "seed": 101,
            }

        with patch(
            "app.services.fal_flux2_pro_service._request_queue",
            side_effect=queue_response,
        ), patch(
            "app.services.fal_flux2_pro_service._download_result",
            return_value=b"generated-image",
        ):
            result = asyncio.run(
                generate_frame(
                    references=[(b"product", "product.png", "image/png")],
                    prompt="luxury campaign",
                    shot_kind="hero",
                    seed=101,
                )
            )

        self.assertEqual(result.image_bytes, b"generated-image")
        self.assertEqual(result.request_id, "request-123")
        self.assertEqual(result.seed, 101)
        self.assertEqual(queue_calls[0][1], "POST")
        self.assertEqual(queue_calls[0][2]["image_size"], "landscape_4_3")
        self.assertEqual(queue_calls[0][2]["seed"], 101)
        self.assertEqual(len(queue_calls[0][2]["image_urls"]), 1)

    def test_failed_queue_returns_provider_error(self):
        with patch(
            "app.services.fal_flux2_pro_service._request_queue",
            side_effect=[
                {"request_id": "request-123"},
                {"status": "FAILED", "error": "safety rejected"},
            ],
        ):
            with self.assertRaises(FalFlux2ProError) as context:
                asyncio.run(
                    generate_frame(
                        references=[(b"product", "product.png", "image/png")],
                        prompt="luxury campaign",
                        shot_kind="studio",
                        seed=101,
                    )
                )
        self.assertEqual(context.exception.code, "FAL_JOB_FAILED")
        self.assertIn("safety rejected", str(context.exception))


if __name__ == "__main__":
    unittest.main()