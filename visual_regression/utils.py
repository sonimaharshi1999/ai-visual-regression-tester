# Author: Maharshi Soni | License: MIT
"""
Utility functions for the AI Visual Regression Tester.

Provides common helpers for image loading, saving, path management,
and data conversion used across all modules.
"""

import os
import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Viewport presets for responsive testing
# ---------------------------------------------------------------------------

VIEWPORT_PRESETS: Dict[str, Tuple[int, int]] = {
    "desktop": (1920, 1080),
    "laptop": (1366, 768),
    "tablet": (768, 1024),
    "tablet_landscape": (1024, 768),
    "mobile": (375, 812),
    "mobile_landscape": (812, 375),
    "mobile_small": (320, 568),
}

# ---------------------------------------------------------------------------
# Change category labels
# ---------------------------------------------------------------------------

CHANGE_CATEGORIES = [
    "layout_shift",
    "color_change",
    "missing_element",
    "new_element",
    "font_change",
    "no_change",
]

# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def load_image(path: str) -> Image.Image:
    """Load an image from *path* and convert to RGB."""
    img = Image.open(path).convert("RGB")
    return img


def save_image(img: Image.Image, path: str) -> str:
    """Save a PIL Image to *path*, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    return path


def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy uint8 array (H, W, 3)."""
    return np.array(img, dtype=np.uint8)


def array_to_image(arr: np.ndarray) -> Image.Image:
    """Convert a NumPy array back to a PIL Image."""
    return Image.fromarray(arr.astype(np.uint8))


def resize_to_match(
    img1: Image.Image, img2: Image.Image
) -> Tuple[Image.Image, Image.Image]:
    """Resize *img2* to match the dimensions of *img1* (width, height)."""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
    return img1, img2


def resize_to_viewport(
    img: Image.Image, viewport: str
) -> Image.Image:
    """Resize an image to a named viewport preset."""
    if viewport not in VIEWPORT_PRESETS:
        raise ValueError(
            f"Unknown viewport '{viewport}'. "
            f"Choose from: {list(VIEWPORT_PRESETS.keys())}"
        )
    w, h = VIEWPORT_PRESETS[viewport]
    return img.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Encoding helpers (for HTML reports)
# ---------------------------------------------------------------------------


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data-URI string."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def file_hash(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist. Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path


def project_root() -> Path:
    """Return the root directory of this project."""
    return Path(__file__).resolve().parent.parent


def default_baselines_dir() -> str:
    return str(project_root() / "baselines")


def default_current_dir() -> str:
    return str(project_root() / "current")


def default_reports_dir() -> str:
    return str(project_root() / "reports")
