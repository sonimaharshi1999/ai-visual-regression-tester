# Author: Maharshi Soni | License: MIT
"""Tests for visual_regression.responsive"""

import pytest
from PIL import Image

from visual_regression.responsive import (
    available_viewports,
    compare_at_viewport,
    compare_responsive,
    compare_responsive_from_paths,
)
from visual_regression.utils import VIEWPORT_PRESETS, save_image, CHANGE_CATEGORIES


class TestAvailableViewports:
    def test_returns_dict(self):
        vps = available_viewports()
        assert isinstance(vps, dict)

    def test_includes_desktop(self):
        assert "desktop" in available_viewports()

    def test_includes_mobile(self):
        assert "mobile" in available_viewports()

    def test_values_are_tuples(self):
        for name, dims in available_viewports().items():
            assert isinstance(dims, tuple) and len(dims) == 2


class TestCompareAtViewport:
    def test_returns_comparison_and_classification(self, simple_baseline, simple_current):
        result = compare_at_viewport(simple_baseline, simple_current, "mobile")
        assert "comparison" in result
        assert "classification" in result
        assert "viewport" in result
        assert result["viewport"] == "mobile"

    def test_dimensions_match_preset(self, simple_baseline, simple_current):
        result = compare_at_viewport(simple_baseline, simple_current, "tablet")
        expected = VIEWPORT_PRESETS["tablet"]
        assert result["dimensions"] == expected

    def test_invalid_viewport_raises(self, simple_baseline, simple_current):
        with pytest.raises(ValueError, match="Unknown viewport"):
            compare_at_viewport(simple_baseline, simple_current, "nonexistent_device")

    def test_classification_valid(self, simple_baseline, simple_current):
        result = compare_at_viewport(simple_baseline, simple_current, "desktop")
        assert result["classification"]["category"] in CHANGE_CATEGORIES


class TestCompareResponsive:
    def test_default_viewports(self, simple_baseline, simple_current):
        results = compare_responsive(simple_baseline, simple_current)
        assert len(results) == 3  # desktop, tablet, mobile

    def test_custom_viewports(self, simple_baseline, simple_current):
        results = compare_responsive(
            simple_baseline, simple_current,
            viewports=["mobile", "laptop"],
        )
        assert len(results) == 2
        assert results[0]["viewport"] == "mobile"
        assert results[1]["viewport"] == "laptop"

    def test_each_result_valid(self, simple_baseline, simple_current):
        results = compare_responsive(simple_baseline, simple_current)
        for r in results:
            assert "comparison" in r
            assert "classification" in r
            assert r["classification"]["category"] in CHANGE_CATEGORIES


class TestCompareResponsiveFromPaths:
    def test_loads_and_compares(self, simple_baseline, simple_current, tmp_dir):
        bp = save_image(simple_baseline, f"{tmp_dir}/base.png")
        cp = save_image(simple_current, f"{tmp_dir}/curr.png")
        results = compare_responsive_from_paths(bp, cp, name="test_page")
        assert len(results) == 3
        for r in results:
            assert "test_page" in r["name"]

    def test_custom_viewports_from_paths(self, simple_baseline, simple_current, tmp_dir):
        bp = save_image(simple_baseline, f"{tmp_dir}/base.png")
        cp = save_image(simple_current, f"{tmp_dir}/curr.png")
        results = compare_responsive_from_paths(
            bp, cp, viewports=["mobile"], name="page"
        )
        assert len(results) == 1
        assert "mobile" in results[0]["name"]
