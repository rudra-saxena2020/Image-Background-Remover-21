import asyncio
import base64
import unittest
from unittest.mock import patch

from app.services.gemini_image_service import (
    GeminiImageError,
    generate_frame,
    status,
)


class GeminiImageServiceTests(unittest.TestCase):
    def test_status_reports_server_side_image_engine(self):
        with patch("app.services.gemini_image_service.settings.GEMINI_IMAGE_ENABLED", True), \
             patch("app.services.gemini_image_service.settings.GEMINI_IMAGE_BASE_URL", "https://gemini.test"), \
             patch("app.services.gemini_image_service.settings.GEMINI_IMAGE_API_KEY", "configured"):
            result = status()
        self.assertTrue(result["ready"])
        self.assertTrue(result["supports_source_images"])
        self.assertEqual(result["model"], "gemini-2.5-flash-image")

    def test_extracts_inline_image_from_gemini_response(self):
        encoded = base64.b64encode(b"generated-image").decode("ascii")
        response = {
            "responseId": "gemini-response-1",
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "done"},
                            {"inlineData": {"mimeType": "image/png", "data": encoded}},
                        ]
                    }
                }
            ],
        }
        with patch("app.services.gemini_image_service._request", return_value=response):
            result = asyncio.run(
                generate_frame(
                    references=[(b"reference", "product.png", "image/png")],
                    prompt="Create a distinct product image.",
                    seed=19,
                )
            )
        self.assertEqual(result.image_bytes, b"generated-image")
        self.assertEqual(result.request_id, "gemini-response-1")

    def test_limits_references_before_provider_call(self):
        with self.assertRaisesRegex(Exception, "up to three"):
            asyncio.run(
                generate_frame(
                    references=[(b"ref", "product.png", "image/png")] * 4,
                    prompt="Create a product image.",
                    seed=1,
                )
            )

    def test_falls_back_to_managed_provider_after_personal_rate_limit(self):
        encoded = base64.b64encode(b"managed-image").decode("ascii")
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"data": encoded}},
                        ]
                    }
                }
            ]
        }
        with patch("app.services.gemini_image_service.settings.GEMINI_IMAGE_BASE_URL", "https://personal.test"), \
             patch("app.services.gemini_image_service.settings.GEMINI_IMAGE_API_KEY", "personal"), \
             patch("app.services.gemini_image_service.settings._GEMINI_MANAGED_BASE_URL", "https://managed.test"), \
             patch("app.services.gemini_image_service.settings._GEMINI_MANAGED_KEY", "managed"), \
             patch(
                 "app.services.gemini_image_service._request",
                 side_effect=[
                     GeminiImageError("GEMINI_RATE_LIMITED", "limited"),
                     response,
                 ],
             ) as request:
            result = asyncio.run(
                generate_frame(
                    references=[(b"reference", "product.png", "image/png")],
                    prompt="Create a distinct product image.",
                    seed=21,
                )
            )
        self.assertEqual(result.image_bytes, b"managed-image")
        self.assertEqual(request.call_count, 2)
