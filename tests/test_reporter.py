# Author: Maharshi Soni | License: MIT
"""Tests for visual_regression.reporter"""

import os
import pytest

from visual_regression.comparator import compare_images
from visual_regression.classifier import classify_comparison
from visual_regression.reporter import generate_report


def _make_result(baseline, current, name="test"):
    comp = compare_images(baseline, current)
    cls = classify_comparison(comp)
    return {
        "name": name,
        "comparison": comp,
        "classification": cls,
        "viewport": "desktop",
    }


class TestGenerateReport:
    """Tests for HTML report generation."""

    def test_creates_file(self, simple_baseline, simple_current, tmp_dir):
        result = _make_result(simple_baseline, simple_current)
        path = os.path.join(tmp_dir, "report.html")
        out = generate_report([result], path)
        assert os.path.isfile(out)

    def test_file_is_html(self, simple_baseline, simple_current, tmp_dir):
        result = _make_result(simple_baseline, simple_current)
        path = os.path.join(tmp_dir, "report.html")
        generate_report([result], path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_contains_title(self, simple_baseline, simple_current, tmp_dir):
        result = _make_result(simple_baseline, simple_current)
        path = os.path.join(tmp_dir, "report.html")
        generate_report([result], path, title="My Test Report")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "My Test Report" in content

    def test_contains_comparison_name(self, simple_baseline, simple_current, tmp_dir):
        result = _make_result(simple_baseline, simple_current, name="Homepage")
        path = os.path.join(tmp_dir, "report.html")
        generate_report([result], path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "Homepage" in content

    def test_contains_base64_images(self, simple_baseline, simple_current, tmp_dir):
        result = _make_result(simple_baseline, simple_current)
        path = os.path.join(tmp_dir, "report.html")
        generate_report([result], path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "data:image/png;base64," in content

    def test_multiple_results(self, simple_baseline, simple_current, tmp_dir):
        r1 = _make_result(simple_baseline, simple_current, "Page A")
        r2 = _make_result(simple_baseline, simple_baseline.copy(), "Page B")
        path = os.path.join(tmp_dir, "report.html")
        generate_report([r1, r2], path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "Page A" in content
        assert "Page B" in content

    def test_empty_results(self, tmp_dir):
        path = os.path.join(tmp_dir, "report.html")
        generate_report([], path)
        assert os.path.isfile(path)

    def test_returns_absolute_path(self, simple_baseline, simple_current, tmp_dir):
        result = _make_result(simple_baseline, simple_current)
        path = os.path.join(tmp_dir, "report.html")
        out = generate_report([result], path)
        assert os.path.isabs(out)

    def test_creates_parent_directories(self, simple_baseline, simple_current, tmp_dir):
        result = _make_result(simple_baseline, simple_current)
        path = os.path.join(tmp_dir, "nested", "deep", "report.html")
        out = generate_report([result], path)
        assert os.path.isfile(out)
