# Author: Maharshi Soni | License: MIT
"""Tests for visual_regression.image_generator"""

import os
import pytest
from PIL import Image

from visual_regression.image_generator import (
    generate_layout_shift,
    generate_color_change,
    generate_missing_element,
    generate_new_element,
    generate_font_change,
    generate_no_change,
    generate_all_scenarios,
    SCENARIOS,
)


class TestIndividualGenerators:
    """Each generator should return a (baseline, current) pair of PIL Images."""

    @pytest.mark.parametrize("gen_fn", list(SCENARIOS.values()), ids=list(SCENARIOS.keys()))
    def test_returns_two_images(self, gen_fn):
        baseline, current = gen_fn(400, 300)
        assert isinstance(baseline, Image.Image)
        assert isinstance(current, Image.Image)

    @pytest.mark.parametrize("gen_fn", list(SCENARIOS.values()), ids=list(SCENARIOS.keys()))
    def test_correct_dimensions(self, gen_fn):
        w, h = 600, 400
        baseline, current = gen_fn(w, h)
        assert baseline.size == (w, h)
        assert current.size == (w, h)

    @pytest.mark.parametrize("gen_fn", list(SCENARIOS.values()), ids=list(SCENARIOS.keys()))
    def test_rgb_mode(self, gen_fn):
        baseline, current = gen_fn(400, 300)
        assert baseline.mode == "RGB"
        assert current.mode == "RGB"


class TestNoChangeGenerator:
    def test_images_identical(self):
        baseline, current = generate_no_change(200, 150)
        import numpy as np
        assert np.array_equal(np.array(baseline), np.array(current))


class TestLayoutShiftGenerator:
    def test_images_differ(self):
        baseline, current = generate_layout_shift(400, 300)
        import numpy as np
        assert not np.array_equal(np.array(baseline), np.array(current))


class TestColorChangeGenerator:
    def test_images_differ(self):
        baseline, current = generate_color_change(400, 300)
        import numpy as np
        assert not np.array_equal(np.array(baseline), np.array(current))


class TestMissingElementGenerator:
    def test_images_differ(self):
        baseline, current = generate_missing_element(400, 300)
        import numpy as np
        assert not np.array_equal(np.array(baseline), np.array(current))


class TestGenerateAllScenarios:
    def test_creates_all_files(self, tmp_dir):
        baselines_dir = os.path.join(tmp_dir, "baselines")
        current_dir = os.path.join(tmp_dir, "current")
        paths = generate_all_scenarios(baselines_dir, current_dir, 400, 300)

        assert len(paths) == len(SCENARIOS)

        for name, (bp, cp) in paths.items():
            assert os.path.isfile(bp), f"Missing baseline: {bp}"
            assert os.path.isfile(cp), f"Missing current: {cp}"
            # Verify they are valid images
            img = Image.open(bp)
            assert img.size == (400, 300)

    def test_returns_correct_keys(self, tmp_dir):
        baselines_dir = os.path.join(tmp_dir, "baselines")
        current_dir = os.path.join(tmp_dir, "current")
        paths = generate_all_scenarios(baselines_dir, current_dir)
        for name in SCENARIOS:
            assert name in paths
