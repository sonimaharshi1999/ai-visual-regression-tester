# AI Visual Regression Tester

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A computer-vision toolkit that detects and **classifies** visual differences between baseline and current web-page screenshots. It goes beyond simple pixel diffing by combining three comparison strategies (SSIM, perceptual hashing, pixel difference) with a feature-engineered ML classifier that labels every detected change as one of five meaningful categories.

---

## Overview

Traditional visual regression tools flag _that_ something changed but rarely explain _what_ changed. This project closes that gap. Given a pair of screenshots (baseline vs. current), the pipeline:

1. **Compares** the images using SSIM, perceptual hash, and absolute pixel diff.
2. **Extracts** a rich feature vector (colour shifts in HSV space, spatial geometry of the changed region, edge-density deltas, intensity analysis).
3. **Classifies** the change into one of six categories with a confidence score.
4. **Generates** a self-contained HTML report with side-by-side thumbnails, diff heat-maps, and per-comparison detail tables.
5. **Supports responsive testing** across desktop, tablet, and mobile viewports out of the box.

A built-in synthetic image generator creates realistic baseline/current pairs for every change category so the entire pipeline can be demoed without a real browser.

---

## Features

- **Three-strategy comparison engine** -- Structural Similarity Index (SSIM), average perceptual hash, and pixel-level absolute difference with configurable threshold.
- **Intelligent change classification** -- layout shift, colour change, missing element, new element, font change, or no change -- each with a confidence score and human-readable explanation.
- **Feature engineering pipeline** -- HSV colour deltas, bounding-box geometry, edge-density analysis, and intensity profiling feed the classifier.
- **Responsive viewport testing** -- compare at seven built-in presets (desktop 1920x1080, laptop 1366x768, tablet 768x1024, tablet landscape, mobile 375x812, mobile landscape, mobile small) or define your own.
- **Self-contained HTML reports** -- summary statistics, side-by-side baseline/current/diff images (base64-embedded), classification badges, and metric tables in a single file you can open in any browser.
- **Synthetic demo generator** -- produces labelled image pairs for every change category, ideal for demos, testing, and CI.
- **CLI interface** -- compare single pairs, entire directories, choose viewports, set thresholds, and control report output from the command line.
- **Pure Python / Pillow** -- no OpenCV dependency; uses Pillow for all image operations and scikit-image for SSIM.
- **Comprehensive test suite** -- 40+ pytest tests covering every module.

---

## Architecture

```
main.py                          CLI entry point & orchestration
 |
 +-- visual_regression/
 |    |-- __init__.py             Package facade
 |    |-- utils.py                Image I/O, viewport presets, base64 encoding
 |    |-- comparator.py           SSIM, perceptual hash, pixel diff
 |    |-- classifier.py           Feature extraction + rule-based classifier
 |    |-- reporter.py             HTML report generation
 |    |-- responsive.py           Multi-viewport comparison orchestrator
 |    +-- image_generator.py      Synthetic test-image factory
 |
 +-- tests/
 |    |-- conftest.py             Shared fixtures
 |    |-- test_comparator.py      Comparison engine tests
 |    |-- test_classifier.py      Classifier tests
 |    |-- test_reporter.py        Report generation tests
 |    |-- test_responsive.py      Responsive module tests
 |    +-- test_image_generator.py Image generator tests
 |
 +-- baselines/                   Generated/user-supplied baseline PNGs
 +-- current/                     Generated/user-supplied current PNGs
 +-- reports/                     Generated HTML reports
```

### Data flow

```
Baseline PNG ──┐
               ├─> comparator.compare_images()
Current  PNG ──┘         |
                         v
                  { ssim_score, phash_score, pixel_mask, diff_image, ... }
                         |
                         v
               classifier.extract_features()
                         |
                         v
               classifier.classify_change()
                         |
                         v
                  { category, confidence, details }
                         |
                         v
                reporter.generate_report()  ──>  report.html
```

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Image processing | **Pillow** >= 10.0 | Loading, resizing, drawing, base64 encoding |
| Structural similarity | **scikit-image** >= 0.21 | SSIM computation |
| Numerical operations | **NumPy** >= 1.24 | Array manipulation, feature math |
| Testing | **pytest** >= 7.4 | Unit and integration tests |
| Report output | Vanilla HTML + CSS | Zero-dependency browser reports |

---

## Getting Started

### Prerequisites

- Python 3.9 or later

### Installation

```bash
# Clone the repository
git clone https://github.com/sonimaharshi1999/ai-visual-regression-tester.git
cd ai-visual-regression-tester

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick start

```bash
# Generate demo images, run all comparisons, and produce an HTML report
python main.py --generate-demo
```

This will:
1. Create synthetic baseline and current screenshots in `baselines/` and `current/`.
2. Compare every matching pair across the default viewports.
3. Write `reports/regression_report.html` -- open it in your browser.

---

## Usage Examples

### Compare two specific images

```bash
python main.py --compare baselines/homepage.png current/homepage.png
```

### Compare entire directories

```bash
python main.py --baseline-dir ./baselines --current-dir ./current
```

### Responsive testing (multiple viewports)

```bash
python main.py --generate-demo --viewports desktop tablet mobile
```

### Custom pixel-diff threshold

```bash
python main.py --generate-demo --threshold 15
```

### Custom report title

```bash
python main.py --generate-demo --title "Sprint 42 Regression Report"
```

### Programmatic usage

```python
from visual_regression import (
    compare_images,
    classify_comparison,
    generate_report,
    compare_responsive,
    generate_all_scenarios,
)
from visual_regression.utils import load_image

# Load images
baseline = load_image("baselines/homepage.png")
current  = load_image("current/homepage.png")

# Compare
comparison = compare_images(baseline, current)
print(f"SSIM: {comparison['ssim_score']:.4f}")
print(f"Pixel change: {comparison['pixel_change_pct']:.2%}")

# Classify
result = classify_comparison(comparison)
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.0%}")

# Responsive comparison (multiple viewports)
results = compare_responsive(baseline, current, viewports=["desktop", "mobile"])

# Generate HTML report
for r in results:
    r["name"] = "Homepage"
generate_report(results, "reports/homepage_report.html")
```

---

## Sample I/O

### CLI output

```
================================================================
  AI Visual Regression Tester
================================================================

[1/3] Generating synthetic demo images ...
  layout_shift          baseline -> baselines/layout_shift.png
                        current  -> current/layout_shift.png
  color_change          baseline -> baselines/color_change.png
                        current  -> current/color_change.png
  ...

[2/3] Running visual comparisons ...
  Completed 6 comparison(s) in 1.24s
  [FAIL] layout_shift                    category=layout_shift       confidence=0.70  ssim=0.8231  pixel_change=8.42%
  [FAIL] color_change                    category=color_change       confidence=0.60  ssim=0.9012  pixel_change=5.17%
  [FAIL] missing_element                 category=missing_element    confidence=0.70  ssim=0.8756  pixel_change=11.23%
  [FAIL] new_element                     category=new_element        confidence=0.70  ssim=0.8543  pixel_change=9.81%
  [FAIL] font_change                     category=font_change        confidence=0.60  ssim=0.9234  pixel_change=3.44%
  [PASS] no_change                       category=no_change          confidence=1.00  ssim=1.0000  pixel_change=0.00%

[3/3] Generating HTML report ...
  Report saved to: reports/regression_report.html

Summary: 1 passed, 5 failed, 6 total
================================================================
```

### HTML report

The generated report includes:

- **Summary bar** -- total comparisons, passed, failed, average SSIM.
- **Comparison cards** -- for each pair:
  - Side-by-side baseline / current / diff thumbnails
  - Classification badge (colour-coded by category)
  - Confidence percentage
  - Detail table with SSIM, perceptual hash, and pixel-change metrics
  - Human-readable analysis text

---

## Project Structure

```
ai-visual-regression-tester/
|-- main.py                       # CLI entry point
|-- requirements.txt              # Python dependencies
|-- .gitignore                    # Git ignore rules
|-- LICENSE                       # MIT license
|-- README.md                     # This file
|-- visual_regression/
|   |-- __init__.py               # Package init and public API
|   |-- utils.py                  # Shared utilities and constants
|   |-- comparator.py             # Image comparison engine
|   |-- classifier.py             # Change classification pipeline
|   |-- reporter.py               # HTML report generator
|   |-- responsive.py             # Multi-viewport testing
|   +-- image_generator.py        # Synthetic image factory
|-- tests/
|   |-- __init__.py
|   |-- conftest.py               # Shared pytest fixtures
|   |-- test_comparator.py        # 16 tests
|   |-- test_classifier.py        # 12 tests
|   |-- test_reporter.py          # 9 tests
|   |-- test_responsive.py        # 11 tests
|   +-- test_image_generator.py   # 9 tests
|-- baselines/                    # Baseline screenshots
|-- current/                      # Current screenshots
+-- reports/                      # Generated HTML reports
```

---

## Tests

Run the full test suite:

```bash
pytest tests/ -v
```

Run a specific test module:

```bash
pytest tests/test_comparator.py -v
```

Run with coverage:

```bash
pip install pytest-cov
pytest tests/ --cov=visual_regression --cov-report=term-missing
```

---

## Contributing

Contributions are welcome! To get started:

1. **Fork** the repository.
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Commit your changes**: `git commit -m "Add my feature"`
4. **Push to the branch**: `git push origin feature/my-feature`
5. **Open a Pull Request**.

### Guidelines

- Follow PEP 8 style conventions.
- Add tests for any new functionality.
- Update documentation if the public API changes.
- Keep commits focused and well-described.

---

## Roadmap

- [ ] **Trained ML classifier** -- replace rule-based heuristics with a scikit-learn model trained on labelled screenshot pairs.
- [ ] **Browser integration** -- use Playwright or Selenium to capture live screenshots automatically.
- [ ] **Region-of-interest masking** -- allow users to exclude dynamic content areas (ads, timestamps) from comparison.
- [ ] **CI/CD integration** -- GitHub Actions workflow that runs visual regression on every PR.
- [ ] **Threshold configuration file** -- YAML/JSON config for per-page sensitivity tuning.
- [ ] **Animated diff overlay** -- slider or fade-toggle in the HTML report for easier side-by-side inspection.
- [ ] **PDF report export** -- generate PDF reports alongside HTML.
- [ ] **Historical trend tracking** -- store comparison results over time and chart regression trends.

---

## Author

**Maharshi Soni**

- GitHub: [github.com/sonimaharshi1999](https://github.com/sonimaharshi1999)
- LinkedIn: [linkedin.com/in/maharshi-soni-b56736170](https://linkedin.com/in/maharshi-soni-b56736170)

---

## License

This project is licensed under the **MIT License** -- see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Pillow](https://python-pillow.org/) -- the friendly PIL fork for Python image processing.
- [scikit-image](https://scikit-image.org/) -- image processing algorithms, especially SSIM.
- [NumPy](https://numpy.org/) -- the foundation of numerical Python.
- [pytest](https://pytest.org/) -- simple and powerful testing framework.
- Perceptual hashing research by Christoph Zauner and others.
