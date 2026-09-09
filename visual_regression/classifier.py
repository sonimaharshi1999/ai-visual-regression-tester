# Author: Maharshi Soni | License: MIT
"""
ML-based change classifier for the AI Visual Regression Tester.

Analyses the comparison artefacts (SSIM map, pixel diff, perceptual hash)
produced by :mod:`comparator` and classifies each detected change into one
of five categories:

    - **layout_shift**    -- structural / positional displacement
    - **color_change**    -- hue or saturation shift without layout change
    - **missing_element** -- content present in baseline but absent in current
    - **new_element**     -- content present in current but absent in baseline
    - **font_change**     -- text-area changes with minimal colour/layout delta

The classifier uses a feature-engineering + heuristic-rules approach that
extracts measurable signals from the diff artefacts. A future iteration can
swap the rules for a trained sklearn model by keeping the same feature vector.
"""

from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image, ImageFilter

from visual_regression.utils import image_to_array, CHANGE_CATEGORIES


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _colour_shift_features(
    baseline: Image.Image, current: Image.Image, mask: np.ndarray
) -> Dict[str, float]:
    """Measure colour distance in changed regions using HSV space."""
    b_arr = image_to_array(baseline).astype(np.float64)
    c_arr = image_to_array(current).astype(np.float64)

    if mask.sum() == 0:
        return {"mean_hue_delta": 0.0, "mean_sat_delta": 0.0, "mean_val_delta": 0.0}

    # Simple RGB-to-pseudo-HSV approximation (avoids OpenCV dependency)
    def _channel_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        v = np.maximum(np.maximum(r, g), b)
        delta = v - np.minimum(np.minimum(r, g), b)
        s = np.where(v > 0, delta / (v + 1e-7), 0.0)
        # Rough hue
        h = np.zeros_like(v)
        nonzero = delta > 0
        idx = nonzero & (v == r)
        h[idx] = 60.0 * (((g[idx] - b[idx]) / (delta[idx] + 1e-7)) % 6)
        idx = nonzero & (v == g)
        h[idx] = 60.0 * (((b[idx] - r[idx]) / (delta[idx] + 1e-7)) + 2)
        idx = nonzero & (v == b)
        h[idx] = 60.0 * (((r[idx] - g[idx]) / (delta[idx] + 1e-7)) + 4)
        return h, s, v

    bh, bs, bv = _channel_stats(b_arr)
    ch, cs, cv = _channel_stats(c_arr)

    return {
        "mean_hue_delta": float(np.abs(bh[mask] - ch[mask]).mean()),
        "mean_sat_delta": float(np.abs(bs[mask] - cs[mask]).mean()),
        "mean_val_delta": float(np.abs(bv[mask] - cv[mask]).mean()),
    }


def _spatial_features(mask: np.ndarray) -> Dict[str, float]:
    """Geometric features of the changed-pixel region."""
    if mask.sum() == 0:
        return {
            "change_area_ratio": 0.0,
            "bbox_aspect_ratio": 0.0,
            "compactness": 0.0,
            "centroid_y_ratio": 0.0,
            "centroid_x_ratio": 0.0,
        }

    ys, xs = np.where(mask)
    h, w = mask.shape

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1

    area = float(mask.sum())
    bbox_area = float(bbox_h * bbox_w)

    return {
        "change_area_ratio": area / (h * w),
        "bbox_aspect_ratio": bbox_w / (bbox_h + 1e-7),
        "compactness": area / (bbox_area + 1e-7),
        "centroid_y_ratio": float(ys.mean()) / h,
        "centroid_x_ratio": float(xs.mean()) / w,
    }


def _edge_features(
    baseline: Image.Image, current: Image.Image, mask: np.ndarray
) -> Dict[str, float]:
    """Compare edge density in changed regions (proxy for text / font changes)."""
    b_edges = np.array(
        baseline.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float64
    )
    c_edges = np.array(
        current.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float64
    )

    if mask.sum() == 0:
        return {"edge_delta_mean": 0.0, "edge_density_baseline": 0.0, "edge_density_current": 0.0}

    region_b = b_edges[mask]
    region_c = c_edges[mask]

    return {
        "edge_delta_mean": float(np.abs(region_b - region_c).mean()),
        "edge_density_baseline": float((region_b > 30).mean()),
        "edge_density_current": float((region_c > 30).mean()),
    }


def _content_presence_features(
    baseline: Image.Image, current: Image.Image, mask: np.ndarray
) -> Dict[str, float]:
    """Detect whether content was added or removed by analysing intensity."""
    b_gray = np.array(baseline.convert("L"), dtype=np.float64)
    c_gray = np.array(current.convert("L"), dtype=np.float64)

    if mask.sum() == 0:
        return {"baseline_intensity": 0.0, "current_intensity": 0.0, "intensity_delta": 0.0}

    bi = b_gray[mask].mean()
    ci = c_gray[mask].mean()

    return {
        "baseline_intensity": float(bi),
        "current_intensity": float(ci),
        "intensity_delta": float(ci - bi),
    }


def extract_features(comparison: Dict[str, Any]) -> Dict[str, float]:
    """
    Build the full feature vector from a :func:`comparator.compare_images` result.
    """
    baseline = comparison["baseline"]
    current = comparison["current"]
    mask = comparison["pixel_mask"]

    features: Dict[str, float] = {
        "ssim_score": comparison["ssim_score"],
        "phash_score": comparison["phash_score"],
        "pixel_change_pct": comparison["pixel_change_pct"],
    }
    features.update(_colour_shift_features(baseline, current, mask))
    features.update(_spatial_features(mask))
    features.update(_edge_features(baseline, current, mask))
    features.update(_content_presence_features(baseline, current, mask))

    return features


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------


def classify_change(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Classify the detected change using engineered features.

    Returns
    -------
    dict with keys:
        category : str  -- one of :data:`CHANGE_CATEGORIES`
        confidence : float -- 0-1 confidence estimate
        details : str -- human-readable explanation
    """
    pct = features["pixel_change_pct"]
    ssim_s = features["ssim_score"]

    # No meaningful change
    if pct < 0.005 and ssim_s > 0.98:
        return {
            "category": "no_change",
            "confidence": min(1.0, ssim_s),
            "details": "Images are virtually identical.",
        }

    hue_d = features["mean_hue_delta"]
    sat_d = features["mean_sat_delta"]
    val_d = features["mean_val_delta"]
    edge_d = features["edge_delta_mean"]
    compact = features["compactness"]
    aspect = features["bbox_aspect_ratio"]
    intensity_d = features["intensity_delta"]
    edge_base = features["edge_density_baseline"]
    edge_curr = features["edge_density_current"]
    area_ratio = features["change_area_ratio"]

    scores: Dict[str, float] = {cat: 0.0 for cat in CHANGE_CATEGORIES if cat != "no_change"}

    # --- Layout shift signals ---
    if ssim_s < 0.85:
        scores["layout_shift"] += 0.3
    if area_ratio > 0.15:
        scores["layout_shift"] += 0.2
    if compact < 0.4:
        scores["layout_shift"] += 0.2

    # --- Colour change signals ---
    if hue_d > 15:
        scores["color_change"] += 0.35
    if sat_d > 0.15:
        scores["color_change"] += 0.25
    if val_d > 30 and hue_d > 5:
        scores["color_change"] += 0.15

    # --- Missing element signals ---
    if intensity_d > 40 and edge_base > edge_curr + 0.1:
        scores["missing_element"] += 0.4
    if edge_base > 0.2 and edge_curr < 0.05:
        scores["missing_element"] += 0.3

    # --- New element signals ---
    if intensity_d < -40 and edge_curr > edge_base + 0.1:
        scores["new_element"] += 0.4
    if edge_curr > 0.2 and edge_base < 0.05:
        scores["new_element"] += 0.3

    # --- Font change signals ---
    if edge_d > 15 and hue_d < 10 and area_ratio < 0.25:
        scores["font_change"] += 0.35
    if 0.5 < aspect < 8.0 and compact > 0.5 and edge_d > 8:
        scores["font_change"] += 0.25

    best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_cat]

    if best_score < 0.1:
        best_cat = "layout_shift"  # fallback when no rule fires strongly
        best_score = 0.3

    confidence = min(1.0, best_score + 0.1)

    details_map = {
        "layout_shift": "Structural displacement detected -- elements have moved or resized.",
        "color_change": "Colour or saturation shift detected in the changed region.",
        "missing_element": "Content present in the baseline appears absent in the current screenshot.",
        "new_element": "New content detected in the current screenshot that was not in the baseline.",
        "font_change": "Text-region changes suggest a font family, size, or weight difference.",
    }

    return {
        "category": best_cat,
        "confidence": round(confidence, 3),
        "details": details_map.get(best_cat, ""),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# Convenience: full pipeline from comparison dict
# ---------------------------------------------------------------------------


def classify_comparison(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features from a comparison result and classify."""
    features = extract_features(comparison)
    result = classify_change(features)
    result["features"] = features
    return result
