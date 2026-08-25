import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import shoot_history_service


class ShootHistoryServiceTests(unittest.TestCase):
    def test_shoot_and_frame_storage_survive_in_memory_cache_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(shoot_history_service, "SHOOTS_ROOT", root / "shoots"), patch.object(
                shoot_history_service, "FRAMES_ROOT", root / "frames"
            ):
                payload = {"id": "shoot-1", "status": "completed", "provider_cost_usd": 0.02}
                shoot_history_service.save_shoot(payload)
                shoot_history_service.save_frame("shoot-1-1", b"image", "image/png")

                self.assertEqual(shoot_history_service.load_shoots(), [payload])
                self.assertEqual(
                    shoot_history_service.load_frame("shoot-1-1"),
                    (b"image", "image/png"),
                )

    def test_missing_frame_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                shoot_history_service, "FRAMES_ROOT", Path(directory) / "frames"
            ):
                self.assertIsNone(shoot_history_service.load_frame("missing"))


if __name__ == "__main__":
    unittest.main()