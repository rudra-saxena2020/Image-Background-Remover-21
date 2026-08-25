import asyncio
import base64
import unittest
from unittest.mock import patch

import app.services.black_forest_flux2_service as bfl_service
from app.services.black_forest_flux2_service import (
    BlackForestFlux2Error,
    _data_uri_references,
    cancel_frame,
    generate_frame,
)


class BlackForestFlux2ServiceTests(unittest.TestCase):
    def tearDown(self):
        bfl_service._runtime_block_reason = None

    def test_binary_references_are_forwarded_as_data_uris(self):
        references = _data_uri_references([(b"product-pixels", "bag.jpg", "image/jpeg")])
        self.assertTrue(references[0].startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(references[0].split(",", 1)[1]), b"product-pixels")

    def test_missing_server_key_keeps_provider_unavailable(self):
        with patch.object(bfl_service.settings, "BFL_FLUX2_ENABLED", True), patch.object(
            bfl_service.settings, "BFL_API_KEY", ""
        ):
            provider = bfl_service.status()
        self.assertFalse(provider["ready"])
        self.assertIn("BFL_API_KEY", str(provider["reason"]))

    def test_completed_request_polls_then_downloads_result(self):
        calls = []

        def request(url, *, method="GET", payload=None):
            calls.append((url, method, payload))
            if method == "POST":
                return {
                    "id": "bfl-request-123",
                    "polling_url": "https://api.bfl.ai/v1/requests/bfl-request-123",
                }
            return {
                "status": "Ready",
                "result": {
                    "sample": "https://cdn.example.com/result.png",
                    "seed": 123,
                },
            }

        with patch(
            "app.services.black_forest_flux2_service._request_json", side_effect=request
        ), patch(
            "app.services.black_forest_flux2_service._download_result",
            return_value=b"generated-image",
        ):
            result = asyncio.run(
                generate_frame(
                    references=[(b"product", "product.png", "image/png")],
                    prompt="exact handbag product image",
                    shot_kind="hero",
                    seed=123,
                )
            )
        self.assertEqual(result.request_id, "bfl-request-123")
        self.assertEqual(result.image_bytes, b"generated-image")
        self.assertEqual(result.seed, 123)
        self.assertEqual(calls[0][1], "POST")
        self.assertEqual(calls[0][2]["input_images"][0].split(",", 1)[0], "data:image/png;base64")
        self.assertEqual(calls[0][2]["width"], 1536)

    def test_terminal_provider_failure_is_not_silently_replaced(self):
        with patch(
            "app.services.black_forest_flux2_service._request_json",
            side_effect=[
                {"id": "bfl-request-123"},
                {"status": "Failed", "error": "provider moderation"},
            ],
        ):
            with self.assertRaises(BlackForestFlux2Error) as context:
                asyncio.run(
                    generate_frame(
                        references=[(b"product", "product.png", "image/png")],
                        prompt="exact handbag product image",
                        shot_kind="studio",
                        seed=1,
                    )
                )
        self.assertEqual(context.exception.code, "BFL_JOB_FAILED")

    def test_cancellation_uses_server_side_request_id_only(self):
        with patch.object(bfl_service.settings, "BFL_FLUX2_ENABLED", True), patch.object(
            bfl_service.settings, "BFL_API_KEY", "server-only-key"
        ), patch(
            "app.services.black_forest_flux2_service._request_json",
        ) as request:
            asyncio.run(cancel_frame("bfl-request-123"))
        request.assert_called_once()
        url = request.call_args.args[0]
        self.assertIn("bfl-request-123/cancel", url)


if __name__ == "__main__":
    unittest.main()