from __future__ import annotations

import os
from datetime import datetime


def safe_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def ensure_photo_dir() -> str:
    path = os.path.join("img", "uav")
    os.makedirs(path, exist_ok=True)
    return path


def build_photo_path(prefix: str = "photo") -> str:
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    return os.path.join(ensure_photo_dir(), filename)
