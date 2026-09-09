# Author: Maharshi Soni | License: MIT
"""
HTML report generator for the AI Visual Regression Tester.

Produces a self-contained HTML file with:
  - Summary statistics (overall pass/fail, score breakdown)
  - Side-by-side baseline / current / diff thumbnails
  - Classification badge for each comparison
  - Expandable detail sections
"""

import os
import datetime
from typing import List, Dict, Any

from PIL import Image

from visual_regression.utils import image_to_base64, ensure_dir


# ---------------------------------------------------------------------------
# CSS & HTML templates
# ---------------------------------------------------------------------------

_CSS = """\
:root {
    --bg: #f8f9fa; --card-bg: #ffffff; --text: #212529;
    --border: #dee2e6; --accent: #0d6efd; --danger: #dc3545;
    --success: #198754; --warning: #ffc107; --info: #0dcaf0;
    --muted: #6c757d;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.6; }
.container { max-width: 1300px; margin: 0 auto; padding: 24px; }
h1 { font-size: 1.75rem; margin-bottom: 8px; }
.subtitle { color: var(--muted); margin-bottom: 24px; }
/* Summary bar */
.summary { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 32px; }
.stat-card { background: var(--card-bg); border: 1px solid var(--border);
             border-radius: 8px; padding: 16px 24px; flex: 1; min-width: 160px;
             text-align: center; }
.stat-card .value { font-size: 2rem; font-weight: 700; }
.stat-card .label { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; }
/* Comparison cards */
.comp-card { background: var(--card-bg); border: 1px solid var(--border);
             border-radius: 8px; margin-bottom: 24px; overflow: hidden; }
.comp-header { padding: 14px 20px; display: flex; align-items: center;
               justify-content: space-between; border-bottom: 1px solid var(--border); }
.comp-header h3 { font-size: 1.1rem; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
         font-size: 0.8rem; font-weight: 600; color: #fff; }
.badge-layout_shift  { background: var(--danger); }
.badge-color_change  { background: var(--warning); color: #212529; }
.badge-missing_element { background: #6f42c1; }
.badge-new_element   { background: var(--info); color: #212529; }
.badge-font_change   { background: #fd7e14; }
.badge-no_change     { background: var(--success); }
.images { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; }
.img-col { padding: 12px; text-align: center; }
.img-col img { max-width: 100%; border: 1px solid var(--border); border-radius: 4px; }
.img-col .cap { font-size: 0.8rem; color: var(--muted); margin-top: 6px; }
.details { padding: 14px 20px; font-size: 0.9rem; border-top: 1px solid var(--border); }
.details table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.details td, .details th { text-align: left; padding: 4px 8px;
                            border-bottom: 1px solid var(--border); }
.details th { width: 200px; color: var(--muted); font-weight: 500; }
.footer { text-align: center; color: var(--muted); font-size: 0.8rem;
          margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); }
@media (max-width: 768px) {
    .images { grid-template-columns: 1fr; }
    .summary { flex-direction: column; }
}
"""


def _stat_card(value: str, label: str) -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    )


def _badge(category: str) -> str:
    nice = category.replace("_", " ").title()
    return f'<span class="badge badge-{category}">{nice}</span>'


def _comparison_card(entry: Dict[str, Any], index: int) -> str:
    """Render one comparison entry as an HTML card."""
    name = entry.get("name", f"Comparison {index + 1}")
    cat = entry["classification"]["category"]
    conf = entry["classification"]["confidence"]
    ssim_s = entry["comparison"]["ssim_score"]
    phash_s = entry["comparison"]["phash_score"]
    pct = entry["comparison"]["pixel_change_pct"]

    baseline_b64 = image_to_base64(entry["comparison"]["baseline"])
    current_b64 = image_to_base64(entry["comparison"]["current"])
    diff_b64 = image_to_base64(entry["comparison"]["pixel_diff_image"])

    viewport = entry.get("viewport", "original")
    details_text = entry["classification"].get("details", "")

    return f"""
    <div class="comp-card">
      <div class="comp-header">
        <h3>{name} <small style="color:var(--muted)">({viewport})</small></h3>
        <div>{_badge(cat)} <span style="margin-left:8px;color:var(--muted)">{conf:.0%} confidence</span></div>
      </div>
      <div class="images">
        <div class="img-col">
          <img src="{baseline_b64}" alt="baseline"/>
          <div class="cap">Baseline</div>
        </div>
        <div class="img-col">
          <img src="{current_b64}" alt="current"/>
          <div class="cap">Current</div>
        </div>
        <div class="img-col">
          <img src="{diff_b64}" alt="diff"/>
          <div class="cap">Difference</div>
        </div>
      </div>
      <div class="details">
        <p><strong>Analysis:</strong> {details_text}</p>
        <table>
          <tr><th>SSIM Score</th><td>{ssim_s:.4f}</td></tr>
          <tr><th>Perceptual Hash Similarity</th><td>{phash_s:.4f}</td></tr>
          <tr><th>Pixel Change</th><td>{pct:.2%}</td></tr>
          <tr><th>Classification</th><td>{cat.replace('_',' ').title()}</td></tr>
        </table>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Visual Regression Report",
) -> str:
    """
    Generate a self-contained HTML report.

    Parameters
    ----------
    results : list[dict]
        Each dict must contain:
            - name : str
            - comparison : dict (from :func:`comparator.compare_images`)
            - classification : dict (from :func:`classifier.classify_comparison`)
            - viewport : str (optional)
    output_path : str
        Where to write the HTML file.
    title : str
        Report heading.

    Returns
    -------
    str  -- the absolute path to the generated report.
    """
    ensure_dir(os.path.dirname(output_path) or ".")

    total = len(results)
    passed = sum(
        1 for r in results if r["classification"]["category"] == "no_change"
    )
    failed = total - passed
    avg_ssim = (
        sum(r["comparison"]["ssim_score"] for r in results) / total if total else 0
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cards_html = "\n".join(
        _comparison_card(r, i) for i, r in enumerate(results)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p class="subtitle">Generated on {now}</p>

  <div class="summary">
    {_stat_card(str(total), "Total Comparisons")}
    {_stat_card(str(passed), "Passed")}
    {_stat_card(str(failed), "Failed")}
    {_stat_card(f"{avg_ssim:.3f}", "Avg SSIM")}
  </div>

  {cards_html}

  <div class="footer">
    AI Visual Regression Tester &middot; Maharshi Soni &middot; MIT License
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(output_path)
