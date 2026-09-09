# Author: Maharshi Soni | License: MIT
"""
AI Visual Regression Tester -- main entry point.

Usage
-----
    python main.py                  # generate demo images, run comparison, produce report
    python main.py --help           # show all options
    python main.py --viewports desktop tablet mobile
    python main.py --baseline-dir ./baselines --current-dir ./current
    python main.py --compare baseline.png current.png
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from visual_regression.utils import (
    load_image,
    default_baselines_dir,
    default_current_dir,
    default_reports_dir,
    ensure_dir,
    VIEWPORT_PRESETS,
)
from visual_regression.comparator import compare_images
from visual_regression.classifier import classify_comparison
from visual_regression.reporter import generate_report
from visual_regression.responsive import compare_responsive, compare_at_viewport
from visual_regression.image_generator import generate_all_scenarios, SCENARIOS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-visual-regression-tester",
        description="Detect and classify visual differences between web-page screenshots.",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CURRENT"),
        help="Compare two specific image files.",
    )
    p.add_argument(
        "--baseline-dir",
        default=None,
        help="Directory containing baseline images (default: ./baselines).",
    )
    p.add_argument(
        "--current-dir",
        default=None,
        help="Directory containing current images (default: ./current).",
    )
    p.add_argument(
        "--report-dir",
        default=None,
        help="Directory for HTML reports (default: ./reports).",
    )
    p.add_argument(
        "--viewports",
        nargs="*",
        default=None,
        help=f"Viewport presets to test. Choices: {list(VIEWPORT_PRESETS.keys())}",
    )
    p.add_argument(
        "--generate-demo",
        action="store_true",
        default=False,
        help="Generate synthetic demo images before comparison.",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Pixel-diff threshold (0-255, default 30).",
    )
    p.add_argument(
        "--title",
        default="Visual Regression Report",
        help="Title for the HTML report.",
    )
    return p


# ---------------------------------------------------------------------------
# Core workflows
# ---------------------------------------------------------------------------


def compare_single_pair(
    baseline_path: str,
    current_path: str,
    viewports: Optional[List[str]] = None,
    threshold: int = 30,
    name: str = "comparison",
) -> List[dict]:
    """Compare one pair of images, optionally across viewports."""
    baseline = load_image(baseline_path)
    current = load_image(current_path)

    results: List[dict] = []

    if viewports:
        for vp in viewports:
            result = compare_at_viewport(
                baseline, current, vp, pixel_threshold=threshold
            )
            result["name"] = f"{name} ({vp})"
            results.append(result)
    else:
        comp = compare_images(baseline, current, pixel_threshold=threshold)
        cls = classify_comparison(comp)
        results.append(
            {
                "name": name,
                "comparison": comp,
                "classification": cls,
                "viewport": "original",
            }
        )
    return results


def compare_directories(
    baseline_dir: str,
    current_dir: str,
    viewports: Optional[List[str]] = None,
    threshold: int = 30,
) -> List[dict]:
    """Compare all matching PNG pairs in two directories."""
    baseline_files = {
        f for f in os.listdir(baseline_dir) if f.lower().endswith(".png")
    }
    current_files = {
        f for f in os.listdir(current_dir) if f.lower().endswith(".png")
    }
    common = sorted(baseline_files & current_files)

    if not common:
        print("  No matching PNG files found in baseline and current directories.")
        return []

    all_results: List[dict] = []
    for fname in common:
        bp = os.path.join(baseline_dir, fname)
        cp = os.path.join(current_dir, fname)
        name = Path(fname).stem
        results = compare_single_pair(bp, cp, viewports=viewports, threshold=threshold, name=name)
        all_results.extend(results)
    return all_results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    baseline_dir = args.baseline_dir or default_baselines_dir()
    current_dir = args.current_dir or default_current_dir()
    report_dir = args.report_dir or default_reports_dir()

    print("=" * 64)
    print("  AI Visual Regression Tester")
    print("=" * 64)

    # Step 1 -- optionally generate demo images
    if args.generate_demo or (args.compare is None and not os.listdir(baseline_dir if os.path.isdir(baseline_dir) else ".")):
        print("\n[1/3] Generating synthetic demo images ...")
        paths = generate_all_scenarios(baseline_dir, current_dir)
        for scenario, (bp, cp) in paths.items():
            print(f"  {scenario:20s}  baseline -> {bp}")
            print(f"  {'':20s}  current  -> {cp}")
        args.generate_demo = True  # flag so we know images exist
    else:
        print("\n[1/3] Skipping demo generation (images already present).")

    # Step 2 -- run comparisons
    print("\n[2/3] Running visual comparisons ...")
    t0 = time.time()

    if args.compare:
        bp, cp = args.compare
        all_results = compare_single_pair(
            bp, cp, viewports=args.viewports, threshold=args.threshold, name=Path(bp).stem
        )
    else:
        all_results = compare_directories(
            baseline_dir, current_dir, viewports=args.viewports, threshold=args.threshold
        )

    elapsed = time.time() - t0
    print(f"  Completed {len(all_results)} comparison(s) in {elapsed:.2f}s")

    for r in all_results:
        cat = r["classification"]["category"]
        conf = r["classification"]["confidence"]
        ssim_s = r["comparison"]["ssim_score"]
        pct = r["comparison"]["pixel_change_pct"]
        status = "PASS" if cat == "no_change" else "FAIL"
        print(
            f"  [{status}] {r['name']:30s}  "
            f"category={cat:18s}  confidence={conf:.2f}  "
            f"ssim={ssim_s:.4f}  pixel_change={pct:.2%}"
        )

    # Step 3 -- generate report
    if all_results:
        print("\n[3/3] Generating HTML report ...")
        ensure_dir(report_dir)
        report_path = os.path.join(report_dir, "regression_report.html")
        out = generate_report(all_results, report_path, title=args.title)
        print(f"  Report saved to: {out}")
    else:
        print("\n[3/3] No results to report.")

    passed = sum(1 for r in all_results if r["classification"]["category"] == "no_change")
    failed = len(all_results) - passed
    print(f"\nSummary: {passed} passed, {failed} failed, {len(all_results)} total")
    print("=" * 64)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
