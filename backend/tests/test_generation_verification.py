from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import settings
from app.api.health import available_generation_engines
from app.api.shoots import _remote_model_matches_status
from app.services.generation_verification_service import (
    verified_status,
    write_verification,
)
from app.services.model_registry import decorate, reference_preview_status


class GenerationVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_path = settings.GENERATION_VERIFICATION_REPORT_PATH
        self.original_ttl = settings.GENERATION_VERIFICATION_TTL_HOURS
        self.original_sdxl_python = settings.SDXL_PYTHON
        self.temporary = tempfile.TemporaryDirectory()
        settings.GENERATION_VERIFICATION_REPORT_PATH = str(
            Path(self.temporary.name) / "report.json"
        )
        settings.GENERATION_VERIFICATION_TTL_HOURS = 168
        self.base = {
            "id": "test",
            "name": "Test engine",
            "mode": "local",
            "ready": True,
            "model_path": str(Path(self.temporary.name) / "model"),
        }
        self.model = Path(self.base["model_path"])
        self.model.mkdir()
        (self.model / "config.json").write_text('{"version": 1}')

    def tearDown(self) -> None:
        settings.GENERATION_VERIFICATION_REPORT_PATH = self.original_path
        settings.GENERATION_VERIFICATION_TTL_HOURS = self.original_ttl
        settings.SDXL_PYTHON = self.original_sdxl_python
        self.temporary.cleanup()

    def test_runtime_ready_engine_fails_closed_before_smoke_test(self) -> None:
        status = verified_status("test", self.base)
        self.assertTrue(status["runtime_ready"])
        self.assertFalse(status["ready"])
        self.assertEqual(status["verification_state"], "unverified")

    def test_passing_smoke_test_enables_same_runtime(self) -> None:
        write_verification(
            "test",
            self.base,
            status="passed",
            reason="Human and product validation passed.",
        )
        status = verified_status("test", self.base)
        self.assertTrue(status["ready"])
        self.assertTrue(status["human_product_verified"])
        self.assertEqual(status["verification_state"], "verified")

    def test_runtime_change_invalidates_passing_smoke_test(self) -> None:
        write_verification(
            "test",
            self.base,
            status="passed",
            reason="Human and product validation passed.",
        )
        changed = {**self.base, "device": "Different GPU"}
        status = verified_status("test", changed)
        self.assertFalse(status["ready"])
        self.assertEqual(status["verification"]["status"], "stale")

    def test_nested_model_change_invalidates_passing_smoke_test(self) -> None:
        write_verification(
            "test",
            self.base,
            status="passed",
            reason="Human and product validation passed.",
        )
        (self.model / "config.json").write_text('{"version": 2}')
        status = verified_status("test", self.base)
        self.assertFalse(status["ready"])
        self.assertEqual(status["verification"]["status"], "stale")

    def test_validator_configuration_change_invalidates_pass(self) -> None:
        original = settings.HUMAN_PRODUCT_IDENTITY_SCORE
        try:
            write_verification(
                "test",
                self.base,
                status="passed",
                reason="Human and product validation passed.",
            )
            settings.HUMAN_PRODUCT_IDENTITY_SCORE = original + 0.01
            status = verified_status("test", self.base)
            self.assertFalse(status["ready"])
            self.assertEqual(status["verification"]["status"], "stale")
        finally:
            settings.HUMAN_PRODUCT_IDENTITY_SCORE = original

    def test_isolated_package_metadata_change_invalidates_pass(self) -> None:
        environment = Path(self.temporary.name) / "runtime"
        python = environment / "bin" / "python"
        metadata = (
            environment
            / "lib"
            / "python3.11"
            / "site-packages"
            / "demo-1.0.dist-info"
            / "METADATA"
        )
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\nexit 1\n")
        python.chmod(0o755)
        metadata.parent.mkdir(parents=True)
        metadata.write_text("Name: demo\nVersion: 1.0\n")
        settings.SDXL_PYTHON = str(python)
        write_verification(
            "sdxl",
            self.base,
            status="passed",
            reason="Human and product validation passed.",
        )
        metadata.write_text("Name: demo\nVersion: 2.0\n")
        status = verified_status("sdxl", self.base)
        self.assertFalse(status["ready"])
        self.assertEqual(status["verification"]["status"], "stale")

    def test_missing_runtime_stays_unavailable_even_with_old_pass(self) -> None:
        write_verification(
            "test",
            self.base,
            status="passed",
            reason="Human and product validation passed.",
        )
        unavailable = {**self.base, "ready": False, "reason": "CUDA missing."}
        status = verified_status("test", unavailable)
        self.assertFalse(status["ready"])
        self.assertEqual(status["verification_state"], "unavailable")

    def test_registry_exposes_independent_audit_fields(self) -> None:
        status = decorate("qwen", {
            **self.base,
            "ready": False,
            "runtime_ready": True,
            "model_present": True,
            "reason": "Run a fresh capability-specific audit.",
        })
        self.assertEqual(status["registry_status"], "unverified")
        self.assertTrue(status["installed"])
        self.assertTrue(status["runtime_reachable"])
        self.assertTrue(status["model_loaded"])
        self.assertFalse(status["inference_passed"])
        self.assertIn("reference_edit", status["capabilities"])
        self.assertTrue(status["requires_human_product_test"])

    def test_registry_marks_larger_gpu_requirement(self) -> None:
        status = decorate("flux2", {
            "id": "flux2",
            "name": "FLUX.2 Dev",
            "configured": True,
            "runtime_ready": False,
            "model_present": True,
            "reason": "Use a larger GPU for this checkpoint.",
        })
        self.assertEqual(status["registry_status"], "requires_larger_gpu")
        self.assertIn("larger GPU", status["next_action"])

    def test_reference_preview_is_not_ai_human_verification(self) -> None:
        status = reference_preview_status()
        self.assertTrue(status["verified"])
        self.assertEqual(status["registry_status"], "verified")
        self.assertNotIn("human_generation", status["capabilities"])
        self.assertFalse(status["requires_human_product_test"])

    def test_available_engines_excludes_unverified_models(self) -> None:
        available = available_generation_engines(
            {
                "flux2_pro": {"ready": True, "provider": "fal.ai"},
                "remote_worker": {"ready": False, "provider": "qwen-edit"},
                "qwen": {"ready": False},
            }
        )
        self.assertEqual([engine["id"] for engine in available], ["flux2-pro"])
        self.assertEqual(available[0]["label"], "FLUX.2 Pro · fal.ai")

    def test_available_engines_includes_only_verified_colab_provider(self) -> None:
        available = available_generation_engines(
            {
                "flux2_pro": {"ready": False},
                "remote_worker": {"ready": True, "provider": "flux-schnell"},
            }
        )
        self.assertEqual([engine["id"] for engine in available], ["colab"])
        self.assertIn("flux-schnell", available[0]["label"])

    def test_available_engines_includes_ready_black_forest_separately(self) -> None:
        available = available_generation_engines(
            {
                "flux2_pro": {"ready": False},
                "black_forest_flux2": {
                    "ready": True,
                    "provider": "Black Forest Labs",
                },
                "remote_worker": {"ready": False},
            }
        )
        self.assertEqual([engine["id"] for engine in available], ["bfl-flux2"])
        self.assertEqual(available[0]["label"], "FLUX.2 Pro · Black Forest")

    def test_explicit_colab_model_must_match_verified_provider(self) -> None:
        worker = {"ready": True, "provider": "qwen-edit"}
        self.assertTrue(_remote_model_matches_status("qwen-edit", worker))
        self.assertTrue(_remote_model_matches_status("auto", worker))
        self.assertFalse(_remote_model_matches_status("flux-schnell", worker))


if __name__ == "__main__":
    unittest.main()