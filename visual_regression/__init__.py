# Author: Maharshi Soni | License: MIT
"""
AI Visual Regression Tester
============================

A computer-vision toolkit that detects and classifies visual differences
between baseline and current web-page screenshots.

Submodules
----------
comparator      -- SSIM, perceptual hash, and pixel-diff comparison
classifier      -- ML-based change classification
reporter        -- HTML report generation
responsive      -- multi-viewport responsive testing
image_generator -- synthetic test-image generation
utils           -- shared helpers and constants
"""

__version__ = "1.0.0"
__author__ = "Maharshi Soni"

from visual_regression.comparator import compare_images
from visual_regression.classifier import classify_comparison
from visual_regression.reporter import generate_report
from visual_regression.responsive import compare_responsive
from visual_regression.image_generator import generate_all_scenarios

__all__ = [
    "compare_images",
    "classify_comparison",
    "generate_report",
    "compare_responsive",
    "generate_all_scenarios",
]
