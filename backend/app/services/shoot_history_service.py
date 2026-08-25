"""Durable local storage for paid shoot metadata and generated frames."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


HISTORY_ROOT = Path(
    os.environ.get(
        "SHOOT_HISTORY_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "shoot-history"),
    )
)
SHOOTS_ROOT = HISTORY_ROOT / "shoots"
FRAMES_ROOT = HISTORY_ROOT / "frames"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def save_shoot(shoot: dict[str, Any]) -> None:
    shoot_id = str(shoot.get("id") or "")
    if not shoot_id:
        return
    payload = json.dumps(shoot, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    _atomic_write(SHOOTS_ROOT / f"{_safe_name(shoot_id)}.json", payload)


def load_shoots() -> list[dict[str, Any]]:
    if not SHOOTS_ROOT.is_dir():
        return []
    loaded: list[dict[str, Any]] = []
    for path in sorted(SHOOTS_ROOT.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("id"):
                loaded.append(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return loaded


def save_frame(frame_id: str, content: bytes, media_type: str) -> None:
    safe_id = _safe_name(frame_id)
    _atomic_write(FRAMES_ROOT / f"{safe_id}.bin", content)
    _atomic_write(
        FRAMES_ROOT / f"{safe_id}.json",
        json.dumps({"media_type": media_type}).encode("utf-8"),
    )


def load_frame(frame_id: str) -> tuple[bytes, str] | None:
    safe_id = _safe_name(frame_id)
    content_path = FRAMES_ROOT / f"{safe_id}.bin"
    metadata_path = FRAMES_ROOT / f"{safe_id}.json"
    try:
        content = content_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        media_type = str(metadata.get("media_type") or "image/png")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return content, media_type