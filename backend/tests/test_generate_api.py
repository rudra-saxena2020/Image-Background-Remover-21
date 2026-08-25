"""Tests for the RunPod FLUX image generation proxy endpoint."""

import unittest
from unittest.mock import MagicMock, patch
import urllib.error
from PIL import Image, ImageDraw
from io import BytesIO

from fastapi.testclient import TestClient


class GenerateApiTests(unittest.TestCase):
    def setUp(self):
        self.qwen_enabled = patch(
            "app.api.generate.settings.QWEN_RUNPOD_ENABLED", False
        )
        self.qwen_enabled.start()
        self.addCleanup(self.qwen_enabled.stop)

    def _client(self):
        from main import app
        return TestClient(app)

    def test_status_reports_configured_when_runpod_url_set(self):
        with patch("app.api.generate.settings.RUNPOD_GENERATE_ENABLED", True), \
             patch("app.api.generate.settings.RUNPOD_URL", "https://pod.example/generate"):
            client = self._client()
            response = client.get("/api/generate/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["configured"])
        self.assertTrue(data["ready"])

    def test_status_reports_unconfigured_when_no_url(self):
        with patch("app.api.generate.settings.RUNPOD_GENERATE_ENABLED", False), \
             patch("app.api.generate.settings.RUNPOD_URL", ""):
            client = self._client()
            response = client.get("/api/generate/status")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["configured"])

    def test_generate_returns_image_bytes(self):
        fake_response = MagicMock()
        fake_response.__enter__ = lambda s: s
        fake_response.__exit__ = MagicMock(return_value=False)
        fake_response.headers.get.return_value = "image/png"
        output = Image.new("RGB", (512, 512), "white")
        pixels = output.load()
        for y in range(128, 384):
            for x in range(160, 352):
                pixels[x, y] = (30, 30, 30)
        output_buffer = BytesIO()
        output.save(output_buffer, format="PNG")
        fake_response.read.return_value = output_buffer.getvalue()

        with patch("app.api.generate.settings.RUNPOD_URL", "https://pod.example/generate"), \
             patch("app.api.generate.settings.RUNPOD_GENERATE_ENABLED", True), \
             patch("app.api.generate.settings.RUNPOD_GENERATE_REQUEST_TIMEOUT_SECONDS", 130), \
             patch("app.api.generate.settings.RUNPOD_GENERATE_MAX_RESULT_BYTES", 33554432), \
             patch("urllib.request.urlopen", return_value=fake_response):
            client = self._client()
            response = client.post(
                "/api/generate",
                data={"prompt": "A red circle"},
                files={"image": ("reference.png", b"source-image", "image/png")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("image/", response.headers["content-type"])

    def test_generate_rejects_empty_prompt(self):
        client = self._client()
        response = client.post(
            "/api/generate",
            data={"prompt": "   "},
            files={"image": ("reference.png", b"source-image", "image/png")},
        )
        self.assertIn(response.status_code, (400, 422))

    def test_generate_accepts_prompt_over_4000_characters(self):
        with patch("app.api.generate.settings.RUNPOD_GENERATE_ENABLED", False), \
             patch("app.api.generate.settings.RUNPOD_URL", ""):
            client = self._client()
            response = client.post(
                "/api/generate",
                data={"prompt": "x" * 4001},
                files={"image": ("reference.png", b"source-image", "image/png")},
            )
        self.assertEqual(response.status_code, 503)

    def test_generate_rejects_panoramic_multi_view_reference(self):
        image = Image.new("RGB", (690, 301), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        client = self._client()
        response = client.post(
            "/api/generate",
            data={"prompt": "Create one clean product photograph."},
            files={"image": ("collage.png", buffer.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("one original product photo", response.json()["detail"])

    def test_generate_rejects_two_panel_reference_under_previous_threshold(self):
        image = Image.new("RGB", (608, 321), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        client = self._client()
        response = client.post(
            "/api/generate",
            data={"prompt": "Create one clean product photograph."},
            files={"image": ("two-panel.png", buffer.getvalue(), "image/png")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("multi-view collage", response.json()["detail"])

    def test_output_validator_rejects_two_large_foreground_objects(self):
        from app.api.generate import _validate_single_product_output

        image = Image.new("RGB", (1024, 1024), "white")
        pixels = image.load()
        for left, top, right, bottom, color in (
            (100, 250, 400, 750, (20, 20, 20)),
            (600, 250, 900, 750, (20, 20, 20)),
        ):
            for y in range(top, bottom):
                for x in range(left, right):
                    pixels[x, y] = color
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with self.assertRaises(Exception) as raised:
            _validate_single_product_output(buffer.getvalue())
        self.assertIn("more than one substantial", str(raised.exception))

    def test_output_validator_rejects_short_panoramic_contact_sheet(self):
        from app.api.generate import _validate_single_product_output

        image = Image.new("RGB", (677, 325), (248, 247, 244))
        draw = ImageDraw.Draw(image)
        for left in (48, 198, 348, 498):
            draw.rectangle((left, 152, left + 128, 282), outline=(210, 208, 202), width=2)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with self.assertRaises(Exception) as raised:
            _validate_single_product_output(buffer.getvalue())
        self.assertIn("multi-view collage", str(raised.exception.detail))

    def test_output_validator_rejects_obvious_product_color_swap(self):
        from app.api.generate import _validate_product_identity

        reference = Image.new("RGB", (512, 512), "white")
        reference_draw = ImageDraw.Draw(reference)
        reference_draw.ellipse((120, 120, 392, 420), fill=(235, 235, 230))
        output = Image.new("RGB", (512, 512), "white")
        output_draw = ImageDraw.Draw(output)
        output_draw.ellipse((120, 120, 392, 420), fill=(20, 20, 20))
        reference_buffer = BytesIO()
        output_buffer = BytesIO()
        reference.save(reference_buffer, format="PNG")
        output.save(output_buffer, format="PNG")

        with self.assertRaises(Exception) as raised:
            _validate_product_identity(reference_buffer.getvalue(), output_buffer.getvalue())
        self.assertIn("color or material", str(raised.exception.detail))

    def test_output_validator_allows_reasonable_lighting_change(self):
        from app.api.generate import _validate_product_identity

        reference = Image.new("RGB", (512, 512), "white")
        reference_draw = ImageDraw.Draw(reference)
        reference_draw.ellipse((120, 120, 392, 420), fill=(235, 235, 230))
        output = Image.new("RGB", (512, 512), "white")
        output_draw = ImageDraw.Draw(output)
        output_draw.ellipse((120, 120, 392, 420), fill=(180, 180, 175))
        reference_buffer = BytesIO()
        output_buffer = BytesIO()
        reference.save(reference_buffer, format="PNG")
        output.save(output_buffer, format="PNG")

        _validate_product_identity(reference_buffer.getvalue(), output_buffer.getvalue())

    def test_generate_returns_502_on_runpod_http_error(self):
        exc = urllib.error.HTTPError(
            url="https://pod.example/generate",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=MagicMock(read=lambda: b"upstream error"),
        )
        with patch("app.api.generate.settings.RUNPOD_URL", "https://pod.example/generate"), \
             patch("app.api.generate.settings.RUNPOD_GENERATE_ENABLED", True), \
             patch("app.api.generate.settings.RUNPOD_GENERATE_REQUEST_TIMEOUT_SECONDS", 130), \
             patch("app.api.generate.settings.RUNPOD_GENERATE_MAX_RESULT_BYTES", 33554432), \
             patch("urllib.request.urlopen", side_effect=exc):
            client = self._client()
            response = client.post(
                "/api/generate",
                data={"prompt": "A red circle"},
                files={"image": ("reference.png", b"source-image", "image/png")},
            )
        self.assertEqual(response.status_code, 502)

    def test_generate_returns_503_when_not_configured(self):
        with patch("app.api.generate.settings.RUNPOD_URL", ""):
            client = self._client()
            response = client.post(
                "/api/generate",
                data={"prompt": "A red circle"},
                files={"image": ("reference.png", b"source-image", "image/png")},
            )
        self.assertEqual(response.status_code, 503)
