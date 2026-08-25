import asyncio
import unittest
from unittest.mock import patch

from app.services.runpod_qwen_image_edit_service import (
    _data_url_references,
    generate_frame,
    status,
)


class RunPodQwenImageEditServiceTests(unittest.TestCase):
    def test_status_fails_closed_without_an_api_key(self):
        with patch(
            "app.services.runpod_qwen_image_edit_service.settings.RUNPOD_API_KEY", ""
        ):
            result = status()
        self.assertFalse(result["ready"])
        self.assertIn("RUNPOD_API_KEY", str(result["reason"]))

    def test_references_are_sent_as_data_urls(self):
        references = _data_url_references(
            [(b"pixels", "bag.png", "image/png")]
        )
        self.assertEqual(references, ["data:image/png;base64,cGl4ZWxz"])

    def test_completed_result_is_downloaded_for_validation(self):
        with patch(
            "app.services.runpod_qwen_image_edit_service._request",
            return_value={
                "id": "sync-123",
                "status": "COMPLETED",
                "output": {
                    "image_url": "https://image.runpod.ai/example/output.png",
                    "cost": 0.02,
                },
            },
        ) as request, patch(
            "app.services.runpod_qwen_image_edit_service._download_result",
            return_value=b"png-bytes",
        ):
            result = asyncio.run(
                generate_frame(
                    references=[(b"pixels", "bag.png", "image/png")],
                    prompt="Place the product in a white studio.",
                    seed=123,
                )
            )
        self.assertEqual(result.image_bytes, b"png-bytes")
        self.assertEqual(result.request_id, "sync-123")
        self.assertEqual(result.cost, 0.02)
        payload = request.call_args.args[0]
        self.assertEqual(payload["input"]["seed"], 123)
        self.assertEqual(payload["input"]["output_format"], "png")
        self.assertEqual(payload["input"]["images"], ["data:image/png;base64,cGl4ZWxz"])


if __name__ == "__main__":
    unittest.main()