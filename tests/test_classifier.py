# Author: Maharshi Soni | License: MIT
"""Tests for visual_regression.classifier"""

import pytest

from visual_regression.comparator import compare_images
from visual_regression.classifier import (
    extract_features,
    classify_change,
    classify_comparison,
)
from visual_regression.utils import CHANGE_CATEGORIES


class TestExtractFeatures:
    """Tests for feature extraction."""

    def test_returns_dict(self, simple_baseline, simple_current):
        comp = compare_images(simple_baseline, simple_current)
        features = extract_features(comp)
        assert isinstance(features, dict)

    def test_contains_core_keys(self, simple_baseline, simple_current):
        comp = compare_images(simple_baseline, simple_current)
        features = extract_features(comp)
        for key in ["ssim_score", "phash_score", "pixel_change_pct",
                     "mean_hue_delta", "mean_sat_delta", "change_area_ratio"]:
            assert key in features, f"Missing feature key: {key}"

    def test_values_are_numeric(self, simple_baseline, simple_current):
        comp = compare_images(simple_baseline, simple_current)
        features = extract_features(comp)
        for k, v in features.items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}, expected numeric"

    def test_identical_images_low_deltas(self, simple_baseline):
        comp = compare_images(simple_baseline, simple_baseline.copy())
        features = extract_features(comp)
        assert features["pixel_change_pct"] < 0.01
        assert features["change_area_ratio"] < 0.01


class TestClassifyChange:
    """Tests for the rule-based classifier."""

    def test_no_change_detected(self, no_change_pair):
        baseline, current = no_change_pair
        comp = compare_images(baseline, current)
        features = extract_features(comp)
        result = classify_change(features)
        assert result["category"] == "no_change"
        assert result["confidence"] > 0.5

    def test_result_has_required_keys(self, simple_baseline, simple_current):
        comp = compare_images(simple_baseline, simple_current)
        features = extract_features(comp)
        result = classify_change(features)
        assert "category" in result
        assert "confidence" in result
        assert "details" in result

    def test_category_is_valid(self, layout_shift_pair):
        baseline, current = layout_shift_pair
        comp = compare_images(baseline, current)
        features = extract_features(comp)
        result = classify_change(features)
        assert result["category"] in CHANGE_CATEGORIES

    def test_confidence_in_range(self, color_change_pair):
        baseline, current = color_change_pair
        comp = compare_images(baseline, current)
        features = extract_features(comp)
        result = classify_change(features)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_color_change_detected(self, color_change_pair):
        baseline, current = color_change_pair
        comp = compare_images(baseline, current)
        features = extract_features(comp)
        result = classify_change(features)
        # Should detect a change (not no_change)
        assert result["category"] != "no_change"

    def test_missing_element_detected(self, missing_element_pair):
        baseline, current = missing_element_pair
        comp = compare_images(baseline, current)
        features = extract_features(comp)
        result = classify_change(features)
        assert result["category"] != "no_change"


class TestClassifyComparison:
    """Tests for the convenience classify_comparison pipeline."""

    def test_returns_full_result(self, simple_baseline, simple_current):
        comp = compare_images(simple_baseline, simple_current)
        result = classify_comparison(comp)
        assert "category" in result
        assert "confidence" in result
        assert "features" in result

    def test_features_included(self, simple_baseline, simple_current):
        comp = compare_images(simple_baseline, simple_current)
        result = classify_comparison(comp)
        assert isinstance(result["features"], dict)
        assert "ssim_score" in result["features"]

    def test_all_scenarios_classify(
        self,
        layout_shift_pair,
        color_change_pair,
        missing_element_pair,
        new_element_pair,
        font_change_pair,
        no_change_pair,
    ):
        """Every generated scenario should produce a valid classification."""
        pairs = [
            layout_shift_pair, color_change_pair, missing_element_pair,
            new_element_pair, font_change_pair, no_change_pair,
        ]
        for baseline, current in pairs:
            comp = compare_images(baseline, current)
            result = classify_comparison(comp)
            assert result["category"] in CHANGE_CATEGORIES
            assert 0.0 <= result["confidence"] <= 1.0
