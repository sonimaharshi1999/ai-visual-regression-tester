# Author: Maharshi Soni | License: MIT
"""
Shared pytest fixtures for the AI Visual Regression Tester test suite.
"""

import os
import pytest
from PIL import Image, ImageDraw

from visual_regression.image_generator import (
    generate_layout_shift,
    generate_color_change,
    generate_missing_element,
    generate_new_element,
    generate_font_change,
    generate_no_change,
)


@pytest.fixture
def img_size():
    """Default test image dimensions (small for speed)."""
    return (400, 300)


@pytest.fixture
def blank_white(img_size):
    """A plain white image."""
    return Image.new("RGB", img_size, (255, 255, 255))


@pytest.fixture
def blank_black(img_size):
    """A plain black image."""
    return Image.new("RGB", img_size, (0, 0, 0))


@pytest.fixture
def simple_baseline(img_size):
    """Baseline with a blue rectangle in the centre."""
    img = Image.new("RGB", img_size, (245, 245, 250))
    d = ImageDraw.Draw(img)
    w, h = img_size
    d.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(41, 98, 255))
    return img


@pytest.fixture
def simple_current(img_size):
    """Current with a red rectangle (colour change vs baseline)."""
    img = Image.new("RGB", img_size, (245, 245, 250))
    d = ImageDraw.Draw(img)
    w, h = img_size
    d.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(220, 53, 69))
    return img


@pytest.fixture
def layout_shift_pair():
    return generate_layout_shift(600, 400)


@pytest.fixture
def color_change_pair():
    return generate_color_change(600, 400)


@pytest.fixture
def missing_element_pair():
    return generate_missing_element(600, 400)


@pytest.fixture
def new_element_pair():
    return generate_new_element(600, 400)


@pytest.fixture
def font_change_pair():
    return generate_font_change(600, 400)


@pytest.fixture
def no_change_pair():
    return generate_no_change(600, 400)


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory and return its string path."""
    return str(tmp_path)
