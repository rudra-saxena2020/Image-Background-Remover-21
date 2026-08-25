import unittest
import io

from PIL import Image

from app.api.shoots import (
    CAMPAIGN_FORMAT_FLEXIBLE,
    CAMPAIGN_FORMAT_FRONT_BACK,
    GENERATION_MODE_HUMAN_MODEL,
    GENERATION_MODE_PRODUCT_ONLY,
    FRONT_BACK_SHOT_PLAN,
    SHOT_PLAN,
    Shoot,
    _product_prompt,
    _campaign_plan,
    _estimate_provider_cost,
    _resolve_generation_backend,
    _validate_campaign_request,
    validate_product_consistency,
)


class ShootRoutingTests(unittest.TestCase):
    def test_explicit_ai_model_selections_are_colab_requests(self):
        expected_models = {
            "qwen": "qwen-edit",
            "flux-schnell": "flux-schnell",
            "fooocus": "fooocus",
            "hidream": "hidream",
            "flux2": "flux2",
            "flux2-klein": "flux2-klein",
            "sdxl": "sdxl",
        }
        for selection, requested_model in expected_models.items():
            with self.subTest(selection=selection):
                self.assertEqual(
                    _resolve_generation_backend(selection),
                    ("colab", requested_model),
                )

    def test_flux2_pro_uses_the_paid_fal_provider(self):
        self.assertEqual(
            _resolve_generation_backend("flux2-pro"),
            ("fal", "fal-ai/flux-2-pro/edit"),
        )

    def test_qwen_runpod_uses_its_own_provider_route(self):
        self.assertEqual(
            _resolve_generation_backend("qwen-runpod"),
            ("qwen-runpod", "qwen-image-edit-2511"),
        )

    def test_flux1_runpod_uses_its_own_provider_route(self):
        self.assertEqual(
            _resolve_generation_backend("flux1-runpod"),
            ("flux1-runpod", "black-forest-labs-flux-1-dev"),
        )

    def test_gemini_image_uses_native_image_route(self):
        self.assertEqual(
            _resolve_generation_backend("gemini-image"),
            ("gemini-image", "gemini-2.5-flash-image"),
        )

    def test_runpod_cost_estimate_uses_frame_count_and_published_rates(self):
        self.assertEqual(_estimate_provider_cost("qwen-runpod", 8), 0.16)
        self.assertEqual(_estimate_provider_cost("flux1-runpod", 8), 0.16777216)
        self.assertIsNone(_estimate_provider_cost("cpu", 8))

    def test_black_forest_flux2_uses_its_own_provider_route(self):
        self.assertEqual(
            _resolve_generation_backend("bfl-flux2"),
            ("bfl", "flux-2-pro"),
        )

    def test_automatic_choices_request_colab_auto_model(self):
        for selection in ("rudras", "auto", "colab"):
            with self.subTest(selection=selection):
                self.assertEqual(
                    _resolve_generation_backend(selection),
                    ("colab", "auto"),
                )

    def test_cpu_remains_the_only_local_preview_path(self):
        self.assertEqual(_resolve_generation_backend("cpu"), ("cpu", None))

    def test_campaign_plans_keep_the_legacy_eight_frame_contract(self):
        self.assertEqual(_campaign_plan(CAMPAIGN_FORMAT_FLEXIBLE), SHOT_PLAN)
        self.assertEqual(len(_campaign_plan(CAMPAIGN_FORMAT_FLEXIBLE)), 8)

    def test_front_back_plan_has_exact_required_frames_and_model_positions(self):
        self.assertEqual(_campaign_plan(CAMPAIGN_FORMAT_FRONT_BACK), FRONT_BACK_SHOT_PLAN)
        self.assertEqual(len(FRONT_BACK_SHOT_PLAN), 7)
        self.assertEqual(
            [kind for kind, _title, _purpose in FRONT_BACK_SHOT_PLAN],
            ["studio", "model", "model-angle", "detail", "angle", "lifestyle", "hero"],
        )

    def test_front_back_campaign_requires_exactly_two_references(self):
        _validate_campaign_request(CAMPAIGN_FORMAT_FRONT_BACK, "campaign", 2)
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            _validate_campaign_request(CAMPAIGN_FORMAT_FRONT_BACK, "campaign", 1)
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            _validate_campaign_request(CAMPAIGN_FORMAT_FRONT_BACK, "campaign", 3)

    def test_front_back_campaign_cannot_use_fast_preview(self):
        with self.assertRaisesRegex(ValueError, "exactly 7"):
            _validate_campaign_request(CAMPAIGN_FORMAT_FRONT_BACK, "fast", 2)

    def test_flexible_campaign_retains_one_to_six_reference_boundary(self):
        for count in range(1, 7):
            _validate_campaign_request(CAMPAIGN_FORMAT_FLEXIBLE, "campaign", count)
        with self.assertRaisesRegex(ValueError, "between 1 and 6"):
            _validate_campaign_request(CAMPAIGN_FORMAT_FLEXIBLE, "campaign", 7)

    def test_generation_mode_defaults_to_product_only(self):
        _validate_campaign_request(CAMPAIGN_FORMAT_FLEXIBLE, "fast", 1)
        _validate_campaign_request(
            CAMPAIGN_FORMAT_FLEXIBLE,
            "campaign",
            3,
            GENERATION_MODE_PRODUCT_ONLY,
        )

    def test_human_model_mode_is_valid_for_both_campaign_contracts(self):
        _validate_campaign_request(
            CAMPAIGN_FORMAT_FLEXIBLE,
            "campaign",
            3,
            GENERATION_MODE_HUMAN_MODEL,
        )
        _validate_campaign_request(
            CAMPAIGN_FORMAT_FRONT_BACK,
            "campaign",
            2,
            GENERATION_MODE_HUMAN_MODEL,
        )

    def test_unknown_generation_mode_fails_before_provider_work(self):
        with self.assertRaisesRegex(ValueError, "Unsupported generation mode"):
            _validate_campaign_request(
                CAMPAIGN_FORMAT_FLEXIBLE,
                "fast",
                1,
                "cinematic-model",
            )

    def test_product_only_prompt_forbids_human_context(self):
        shoot = Shoot(
            id="test",
            product_name="Test bag",
            category="bags",
            atmosphere="soft daylight",
            background="white studio",
            output_format="png",
            engine="colab",
            speed_mode="campaign",
            campaign_format=CAMPAIGN_FORMAT_FLEXIBLE,
            generation_mode=GENERATION_MODE_PRODUCT_ONLY,
            frame_count=8,
            reference_count=1,
            created_at="2026-01-01T00:00:00Z",
        )
        prompt = _product_prompt(shoot, "model")
        self.assertIn("PRODUCT-ONLY CATALOG MODE", prompt)
        self.assertIn("never add a person, hand, model", prompt)
        self.assertNotIn("MANDATORY HUMAN MODEL REQUIREMENT", prompt)

    def test_product_only_prompt_uses_permanent_consistency_lock(self):
        shoot = Shoot(
            id="test",
            product_name="Test bag",
            category="bags",
            atmosphere="soft daylight",
            background="white studio",
            output_format="png",
            engine="colab",
            speed_mode="campaign",
            campaign_format=CAMPAIGN_FORMAT_FLEXIBLE,
            generation_mode=GENERATION_MODE_PRODUCT_ONLY,
            frame_count=8,
            reference_count=3,
            created_at="2026-01-01T00:00:00Z",
            identity_profile={
                "consistency_lock": {
                    "known_view_states": ["angle", "front"],
                    "supported_opening": False,
                }
            },
        )
        prompt = _product_prompt(shoot, "angle")
        self.assertIn("PERMANENT PRODUCT CONSISTENCY LOCK", prompt)
        self.assertIn("one fixed physical object", prompt)
        self.assertIn("never invent a lining", prompt)

    def test_consistency_validator_fails_closed_for_unsupported_opening(self):
        image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        for x in range(96, 416):
            for y in range(96, 416):
                image.putpixel((x, y), (165, 105, 55, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        validation = validate_product_consistency(
            buffer.getvalue(),
            {
                "consistency_lock": {
                    "supported_opening": False,
                    "locked_dimensions": {
                        "aspect_ratio": 1.0,
                        "foreground_ratio_min": 0.39,
                        "foreground_ratio_max": 0.39,
                    },
                    "locked_colors": [[165, 105, 55]],
                }
            },
            shot_kind="open",
        )
        self.assertFalse(validation["passed"])
        self.assertIn("unsupported", validation["reason"])

    def test_human_model_prompt_keeps_category_interaction_requirement(self):
        shoot = Shoot(
            id="test",
            product_name="Test bag",
            category="bags",
            atmosphere="soft daylight",
            background="white studio",
            output_format="png",
            engine="colab",
            speed_mode="campaign",
            campaign_format=CAMPAIGN_FORMAT_FLEXIBLE,
            generation_mode=GENERATION_MODE_HUMAN_MODEL,
            frame_count=8,
            reference_count=1,
            created_at="2026-01-01T00:00:00Z",
        )
        prompt = _product_prompt(shoot, "model")
        self.assertIn("MANDATORY HUMAN MODEL REQUIREMENT", prompt)
        self.assertIn("believable hand placement", prompt)


if __name__ == "__main__":
    unittest.main()