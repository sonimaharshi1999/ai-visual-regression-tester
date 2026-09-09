# Author: Maharshi Soni | License: MIT
"""Tests for visual_regression.comparator"""

import numpy as np
import pytest
from PIL import Image

from visual_regression.comparator import (
    ssim_compare,
    perceptual_hash_similarity,
    pixel_diff,
    compare_images,
)


class TestSSIMCompare:
    """Tests for the SSIM comparison function."""

    def test_identical_images_score_one(self, simple_baseline):
        score, diff_map = ssim_compare(simple_baseline, simple_baseline.copy())
        assert score == pytest.approx(1.0, abs=0.001)

    def test_different_images_score_below_one(self, simple_baseline, simple_current):
        score, _ = ssim_compare(simple_baseline, simple_current)
        assert score < 1.0

    def test_diff_map_shape_matches(self, simple_baseline, simple_current):
        _, diff_map = ssim_compare(simple_baseline, simple_current)
        w, h = simple_baseline.size
        assert diff_map.shape == (h, w)

    def test_completely_different_images(self, blank_white, blank_black):
        score, _ = ssim_compare(blank_white, blank_black)
        assert score < 0.5

    def test_returns_float(self, simple_baseline):
        score, _ = ssim_compare(simple_baseline, simple_baseline)
        assert isinstance(score, float)

    def test_small_images(self):
        """SSIM should work even for very small images."""
        a = Image.new("RGB", (16, 16), (128, 128, 128))
        b = Image.new("RGB", (16, 16), (130, 130, 130))
        score, _ = ssim_compare(a, b, win_size=3)
        assert 0.0 <= score <= 1.0


class TestPerceptualHash:
    """Tests for perceptual hash similarity."""

    def test_identical_images_return_one(self, simple_baseline):
        score = perceptual_hash_similarity(simple_baseline, simple_baseline.copy())
        assert score == pytest.approx(1.0, abs=0.001)

    def test_different_images_below_one(self, blank_white, blank_black):
        score = perceptual_hash_similarity(blank_white, blank_black)
        assert score < 1.0

    def test_score_in_range(self, simple_baseline, simple_current):
        score = perceptual_hash_similarity(simple_baseline, simple_current)
        assert 0.0 <= score <= 1.0

    def test_similar_images_high_score(self, simple_baseline):
        # Slightly modify one pixel -- should still be very similar
        modified = simple_baseline.copy()
        modified.putpixel((0, 0), (244, 244, 249))
        score = perceptual_hash_similarity(simple_baseline, modified)
        assert score > 0.9


class TestPixelDiff:
    """Tests for pixel-level difference."""

    def test_identical_zero_change(self, simple_baseline):
        pct, diff_img, mask = pixel_diff(simple_baseline, simple_baseline.copy())
        assert pct == pytest.approx(0.0, abs=0.001)
        assert mask.sum() == 0

    def test_totally_different(self, blank_white, blank_black):
        pct, diff_img, mask = pixel_diff(blank_white, blank_black, threshold=10)
        assert pct > 0.9

    def test_diff_image_is_pil(self, simple_baseline, simple_current):
        _, diff_img, _ = pixel_diff(simple_baseline, simple_current)
        assert isinstance(diff_img, Image.Image)

    def test_mask_is_boolean(self, simple_baseline, simple_current):
        _, _, mask = pixel_diff(simple_baseline, simple_current)
        assert mask.dtype == bool

    def test_threshold_affects_sensitivity(self, simple_baseline, simple_current):
        pct_low, _, _ = pixel_diff(simple_baseline, simple_current, threshold=5)
        pct_high, _, _ = pixel_diff(simple_baseline, simple_current, threshold=100)
        assert pct_low >= pct_high


class TestCompareImages:
    """Tests for the composite comparison function."""

    def test_returns_all_keys(self, simple_baseline, simple_current):
        result = compare_images(simple_baseline, simple_current)
        expected_keys = {
            "ssim_score", "ssim_map", "phash_score",
            "pixel_change_pct", "pixel_diff_image", "pixel_mask",
            "baseline", "current",
        }
        assert expected_keys.issubset(result.keys())

    def test_identical_images(self, simple_baseline):
        result = compare_images(simple_baseline, simple_baseline.copy())
        assert result["ssim_score"] == pytest.approx(1.0, abs=0.001)
        assert result["phash_score"] == pytest.approx(1.0, abs=0.001)
        assert result["pixel_change_pct"] == pytest.approx(0.0, abs=0.001)

    def test_different_sizes_handled(self):
        """compare_images should resize the second image to match."""
        a = Image.new("RGB", (200, 150), (100, 100, 100))
        b = Image.new("RGB", (300, 200), (110, 110, 110))
        result = compare_images(a, b)
        assert result["baseline"].size == result["current"].size
