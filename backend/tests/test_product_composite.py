from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.services.controlled_composite_service import (
    composite_product_layer,
    validate_immutable_product_composite,
)
from app.services.product_identity_service import build_product_identity_profile


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ProductCompositeTests(unittest.TestCase):
    def test_profile_and_composite_keep_same_source_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutout = Image.new("RGBA", (120, 100), (0, 0, 0, 0))
            draw = ImageDraw.Draw(cutout)
            draw.rounded_rectangle((20, 20, 100, 92), radius=12, fill=(120, 20, 35, 255))
            cutout_path = root / "product.png"
            cutout.save(cutout_path)
            profile = build_product_identity_profile(
                source_paths=[str(cutout_path)],
                source_bytes=[b"original-reference"],
                product_name="Test bag",
                output_path=str(root / "profile.json"),
            )
            plate = Image.new("RGB", (400, 400), (220, 210, 205))
            output, metadata = composite_product_layer(
                plate_bytes=_png(plate),
                product_cutout_path=str(cutout_path),
                shot_kind="model",
                output_path=str(root / "frame.png"),
            )
            validation = validate_immutable_product_composite(
                output,
                metadata=metadata,
                profile=profile,
            )
            self.assertTrue(validation["passed"])
            self.assertTrue(validation["source_layer_applied"])
            self.assertEqual(
                metadata["product_source_sha256"],
                profile["primary_view"]["sha256"],
            )
            self.assertEqual(json.loads((root / "profile.json").read_text())["version"], 1)

    def test_composite_rejects_unprofiled_or_missing_product_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutout = Image.new("RGBA", (20, 20), (50, 20, 10, 255))
            cutout_path = root / "product.png"
            cutout.save(cutout_path)
            output, metadata = composite_product_layer(
                plate_bytes=_png(Image.new("RGB", (200, 200), "white")),
                product_cutout_path=str(cutout_path),
                shot_kind="model",
                output_path=str(root / "frame.png"),
            )
            metadata["product_source_sha256"] = "altered"
            result = validate_immutable_product_composite(
                output,
                metadata=metadata,
                profile={"version": 1, "primary_view": {"sha256": "different"}},
            )
            self.assertFalse(result["passed"])
            self.assertIn("immutable product source", result["reason"])


if __name__ == "__main__":
    unittest.main()