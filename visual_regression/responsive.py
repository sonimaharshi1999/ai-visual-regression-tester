# Author: Maharshi Soni | License: MIT
"""
Responsive testing module for the AI Visual Regression Tester.

Handles viewport-based image resizing and orchestrates comparison across
multiple device presets (desktop, tablet, mobile, etc.).
"""

from typing import List, Dict, Any, Optional

from PIL import Image

from visual_regression.utils import (
    VIEWPORT_PRESETS,
    resize_to_viewport,
    load_image,
)
from visual_regression.comparator import compare_images
from visual_regression.classifier import classify_comparison


def available_viewports() -> Dict[str, tuple]:
    """Return the dict of built-in viewport presets."""
    return dict(VIEWPORT_PRESETS)


def compare_at_viewport(
    baseline: Image.Image,
    current: Image.Image,
    viewport: str,
    ssim_win_size: int = 7,
    pixel_threshold: int = 30,
) -> Dict[str, Any]:
    """
    Resize both images to *viewport* and run the full comparison pipeline.

    Returns a dict ready for the reporter:
        comparison, classification, viewport
    """
    b_resized = resize_to_viewport(baseline, viewport)
    c_resized = resize_to_viewport(current, viewport)

    comparison = compare_images(
        b_resized,
        c_resized,
        ssim_win_size=ssim_win_size,
        pixel_threshold=pixel_threshold,
    )
    classification = classify_comparison(comparison)

    return {
        "comparison": comparison,
        "classification": classification,
        "viewport": viewport,
        "dimensions": VIEWPORT_PRESETS[viewport],
    }


def compare_responsive(
    baseline: Image.Image,
    current: Image.Image,
    viewports: Optional[List[str]] = None,
    ssim_win_size: int = 7,
    pixel_threshold: int = 30,
) -> List[Dict[str, Any]]:
    """
    Compare a baseline/current pair across multiple viewports.

    Parameters
    ----------
    viewports : list[str] or None
        Viewport names to test. Defaults to ``["desktop", "tablet", "mobile"]``.

    Returns
    -------
    list[dict]  -- one result dict per viewport, each containing
        comparison, classification, viewport, dimensions.
    """
    if viewports is None:
        viewports = ["desktop", "tablet", "mobile"]

    results: List[Dict[str, Any]] = []
    for vp in viewports:
        result = compare_at_viewport(
            baseline,
            current,
            vp,
            ssim_win_size=ssim_win_size,
            pixel_threshold=pixel_threshold,
        )
        results.append(result)

    return results


def compare_responsive_from_paths(
    baseline_path: str,
    current_path: str,
    viewports: Optional[List[str]] = None,
    name: str = "page",
) -> List[Dict[str, Any]]:
    """
    Load images from file paths and run responsive comparison.

    Each result dict includes the *name* key for the reporter.
    """
    baseline = load_image(baseline_path)
    current = load_image(current_path)

    raw_results = compare_responsive(baseline, current, viewports=viewports)
    for r in raw_results:
        r["name"] = f"{name} ({r['viewport']})"
    return raw_results
