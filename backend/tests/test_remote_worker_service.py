import base64
import unittest
from unittest.mock import patch

from app.services.remote_worker_service import RemoteWorkerError, _encode_references, status


class RemoteWorkerServiceTests(unittest.TestCase):
    def _health(self, **overrides):
        payload = {
            "worker": "atelier-colab",
            "service": "atelier-colab-worker",
            "provider": "flux-schnell",
            "runtime_ready": True,
            "gpu": {"cuda_available": True},
            "gpu_info": {"cuda_available": True, "device": "T4"},
            "models": [],
            "inference_available": False,
            "inference_passed": False,
            "verification_state": "unverified",
            "human_product_verified": False,
            "provider_status": {
                "provider": "flux-schnell",
                "model_loaded": False,
                "inference_passed": False,
                "error_code": "MODEL_LOAD_ERROR",
                "reason": "weights missing",
            },
        }
        payload.update(overrides)
        return payload

    def test_unconfigured_worker_is_fail_closed(self):
        with patch("app.services.remote_worker_service.settings.WORKER_URL", ""), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", ""
        ):
            result = status()
        self.assertFalse(result["ready"])
        self.assertIn("WORKER_UNAVAILABLE", result["reason"])

    def test_reference_payload_is_encoded_without_paths(self):
        result = _encode_references([(b"pixels", "bag.jpg", "image/jpeg")])
        self.assertEqual(result[0]["filename"], "bag.jpg")
        self.assertEqual(base64.b64decode(result[0]["data_base64"]), b"pixels")
        self.assertNotIn("path", result[0])

    def test_http_auth_failure_is_explicit(self):
        with patch("app.services.remote_worker_service.settings.WORKER_URL", "https://worker.example"), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", "secret"
        ), patch(
            "app.services.remote_worker_service.urllib.request.urlopen",
            side_effect=__import__("urllib").error.HTTPError(
                "https://worker.example/health", 401, "unauthorized", {}, None
            ),
        ):
            with self.assertRaises(RemoteWorkerError) as context:
                from app.services.remote_worker_service import _request
                _request("/health")
        self.assertEqual(context.exception.code, "WORKER_AUTH_FAILED")

    def test_http_404_identifies_stale_colab_endpoint(self):
        with patch("app.services.remote_worker_service.settings.WORKER_URL", "https://worker.example"), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", "secret"
        ), patch(
            "app.services.remote_worker_service.urllib.request.urlopen",
            side_effect=__import__("urllib").error.HTTPError(
                "https://worker.example/health", 404, "not found", {}, None
            ),
        ):
            with self.assertRaises(RemoteWorkerError) as context:
                from app.services.remote_worker_service import _request
                _request("/health")
        self.assertEqual(context.exception.code, "WORKER_ENDPOINT_NOT_FOUND")
        self.assertIn("COLAB_WORKER_URL", str(context.exception))

    def test_legacy_generation_ready_model_is_inference_ready_but_unverified(self):
        legacy_health = {
            "worker": "atelier-colab",
            "service": "atelier-colab-worker",
            "gpu": True,
            "gpu_name": "Tesla T4",
            "models": ["Qwen/Qwen-Image-Edit-2509"],
            "generation_ready": True,
            "verification": "NOT_VERIFIED",
        }
        with patch("app.services.remote_worker_service.settings.WORKER_URL", "https://worker.example"), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", "secret"
        ), patch("app.services.remote_worker_service._request", return_value=legacy_health):
            result = status()
        self.assertFalse(result["ready"])
        self.assertTrue(result["runtime_ready"])
        self.assertTrue(result["model_loaded"])
        self.assertTrue(result["inference_passed"])
        self.assertEqual(result["worker_state"], "inference-ready-unverified")
        self.assertIn("verify", result["next_action"].lower())

    def test_empty_legacy_worker_does_not_count_background_model(self):
        legacy_health = {
            "worker": "atelier-colab",
            "service": "atelier-colab-worker",
            "gpu": True,
            "models": ["BiRefNet"],
            "generation_ready": False,
            "verification": "NOT_VERIFIED",
        }
        with patch("app.services.remote_worker_service.settings.WORKER_URL", "https://worker.example"), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", "secret"
        ), patch("app.services.remote_worker_service._request", return_value=legacy_health):
            result = status()
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["inference_passed"])
        self.assertEqual(result["worker_state"], "reachable-empty")

    def test_reachable_empty_worker_is_not_loaded(self):
        with patch("app.services.remote_worker_service.settings.WORKER_URL", "https://worker.example"), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", "secret"
        ), patch("app.services.remote_worker_service._request", return_value=self._health(
            provider_status={"error_code": "NO_PROVIDER_CONFIGURED", "model_loaded": False, "inference_passed": False},
            provider="",
        )):
            result = status()
        self.assertEqual(result["worker_state"], "reachable-empty")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["inference_available"])
        self.assertFalse(result["ready"])

    def test_inference_ready_worker_stays_unverified(self):
        with patch("app.services.remote_worker_service.settings.WORKER_URL", "https://worker.example"), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", "secret"
        ), patch("app.services.remote_worker_service._request", return_value=self._health(
            models=[{"id": "flux-schnell", "loaded": True}],
            inference_available=True,
            inference_passed=True,
            provider_status={"provider": "flux-schnell", "model_loaded": True, "inference_passed": True},
            worker_state="inference-ready-unverified",
        )):
            result = status()
        self.assertEqual(result["worker_state"], "inference-ready-unverified")
        self.assertTrue(result["model_loaded"])
        self.assertTrue(result["inference_available"])
        self.assertFalse(result["ready"])

    def test_verified_worker_is_ready_only_after_product_verification(self):
        with patch("app.services.remote_worker_service.settings.WORKER_URL", "https://worker.example"), patch(
            "app.services.remote_worker_service.settings.WORKER_TOKEN", "secret"
        ), patch("app.services.remote_worker_service._request", return_value=self._health(
            models=[{"id": "flux-schnell", "loaded": True}],
            inference_available=True,
            inference_passed=True,
            verification_state="verified",
            human_product_verified=True,
            provider_status={"provider": "flux-schnell", "model_loaded": True, "inference_passed": True},
            worker_state="verified",
        )):
            result = status()
        self.assertEqual(result["worker_state"], "verified")
        self.assertTrue(result["ready"])


if __name__ == "__main__":
    unittest.main()