# Author: Maharshi Soni | License: MIT
"""
Image comparison engine for the AI Visual Regression Tester.

Implements three complementary comparison strategies:
  1. Structural Similarity Index (SSIM) via scikit-image
  2. Perceptual hashing (average hash) via Pillow
  3. Pixel-level difference map

Each method returns a similarity score and/or a difference artefact that
downstream modules (classifier, reporter) consume.
"""

from typing import Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from visual_regression.utils import (
    image_to_array,
    array_to_image,
    resize_to_match,
)


# ---------------------------------------------------------------------------
# Perceptual hashing (average hash)
# ---------------------------------------------------------------------------


def _average_hash(img: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Compute a binary average-hash vector for *img*."""
    resized = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(resized, dtype=np.float64)
    mean_val = pixels.mean()
    return (pixels > mean_val).astype(np.uint8)


def perceptual_hash_similarity(
    img1: Image.Image, img2: Image.Image, hash_size: int = 16
) -> float:
    """Return a 0-1 similarity score based on average-hash Hamming distance."""
    h1 = _average_hash(img1, hash_size).flatten()
    h2 = _average_hash(img2, hash_size).flatten()
    total_bits = h1.size
    matching = np.sum(h1 == h2)
    return float(matching / total_bits)


# ---------------------------------------------------------------------------
# SSIM comparison
# ---------------------------------------------------------------------------


def ssim_compare(
    img1: Image.Image, img2: Image.Image, win_size: int = 7
) -> Tuple[float, np.ndarray]:
    """
    Compute the mean SSIM between two images.

    Returns
    -------
    score : float
        Mean SSIM in [0, 1].
    diff_map : np.ndarray
        Per-pixel SSIM map (same spatial dims as input, single-channel float).
    """
    img1, img2 = resize_to_match(img1, img2)
    arr1 = image_to_array(img1)
    arr2 = image_to_array(img2)

    # Convert to grayscale for SSIM
    gray1 = np.mean(arr1, axis=2).astype(np.float64)
    gray2 = np.mean(arr2, axis=2).astype(np.float64)

    # Ensure win_size is odd and <= smallest image dimension
    min_dim = min(gray1.shape[0], gray1.shape[1])
    if win_size > min_dim:
        win_size = max(3, min_dim if min_dim % 2 == 1 else min_dim - 1)

    score, diff_map = ssim(
        gray1,
        gray2,
        full=True,
        data_range=255.0,
        win_size=win_size,
    )
    return float(score), diff_map


# ---------------------------------------------------------------------------
# Pixel difference
# ---------------------------------------------------------------------------


def pixel_diff(
    img1: Image.Image, img2: Image.Image, threshold: int = 30
) -> Tuple[float, Image.Image, np.ndarray]:
    """
    Compute the absolute pixel difference between two images.

    Parameters
    ----------
    threshold : int
        Per-channel difference below which a pixel is considered unchanged.

    Returns
    -------
    change_pct : float
        Fraction of pixels that changed (0-1).
    diff_image : Image.Image
        Visual heat-map of differences (brighter = larger delta).
    mask : np.ndarray
        Boolean mask where True means the pixel changed.
    """
    img1, img2 = resize_to_match(img1, img2)
    arr1 = image_to_array(img1).astype(np.int16)
    arr2 = image_to_array(img2).astype(np.int16)

    diff = np.abs(arr1 - arr2).astype(np.uint8)
    # Per-pixel max channel diff for the mask
    max_diff = diff.max(axis=2)
    mask = max_diff > threshold

    change_pct = float(mask.sum() / mask.size)

    # Amplify the diff for visualisation
    amplified = np.clip(diff * 3, 0, 255).astype(np.uint8)
    diff_image = array_to_image(amplified)

    return change_pct, diff_image, mask


# ---------------------------------------------------------------------------
# Composite comparison (runs all three)
# ---------------------------------------------------------------------------


def compare_images(
    baseline: Image.Image,
    current: Image.Image,
    ssim_win_size: int = 7,
    pixel_threshold: int = 30,
    hash_size: int = 16,
) -> Dict[str, Any]:
    """
    Run the full comparison suite on a baseline/current image pair.

    Returns a dict with keys:
        ssim_score, ssim_map, phash_score, pixel_change_pct,
        pixel_diff_image, pixel_mask
    """
    baseline, current = resize_to_match(baseline, current)

    ssim_score, ssim_map = ssim_compare(baseline, current, win_size=ssim_win_size)
    phash_score = perceptual_hash_similarity(baseline, current, hash_size=hash_size)
    change_pct, diff_img, mask = pixel_diff(baseline, current, threshold=pixel_threshold)

    return {
        "ssim_score": ssim_score,
        "ssim_map": ssim_map,
        "phash_score": phash_score,
        "pixel_change_pct": change_pct,
        "pixel_diff_image": diff_img,
        "pixel_mask": mask,
        "baseline": baseline,
        "current": current,
    }
