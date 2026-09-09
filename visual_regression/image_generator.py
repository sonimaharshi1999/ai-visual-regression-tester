# Author: Maharshi Soni | License: MIT
"""
Synthetic image generator for demo and testing purposes.

Creates pairs of baseline / current screenshots that exhibit each of the
five change categories so the full pipeline can be exercised without a
real browser.

Each generator returns ``(baseline, current)`` PIL Images at the
requested resolution.
"""

import os
from typing import Tuple, Optional, Dict

from PIL import Image, ImageDraw, ImageFont

from visual_regression.utils import ensure_dir, save_image


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

_BG = (245, 245, 250)
_HEADER_BG = (41, 98, 255)
_HEADER_BG_ALT = (41, 98, 255)
_SIDEBAR_BG = (230, 232, 240)
_CARD_BG = (255, 255, 255)
_TEXT = (33, 37, 41)
_ACCENT = (13, 110, 253)
_BORDER = (200, 204, 210)
_BUTTON = (25, 135, 84)


def _try_font(size: int = 14) -> ImageFont.ImageFont:
    """Try to load a TTF font; fall back to the default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Primitive drawing helpers
# ---------------------------------------------------------------------------


def _draw_header(draw: ImageDraw.ImageDraw, w: int, bg: tuple = _HEADER_BG, text: str = "My Application"):
    """Draw a top navigation bar."""
    draw.rectangle([0, 0, w, 60], fill=bg)
    font = _try_font(20)
    draw.text((20, 16), text, fill=(255, 255, 255), font=font)
    # nav links
    small = _try_font(13)
    x = w - 300
    for label in ["Home", "About", "Products", "Contact"]:
        draw.text((x, 22), label, fill=(220, 225, 255), font=small)
        x += 70


def _draw_sidebar(draw: ImageDraw.ImageDraw, h: int, x0: int = 0, w: int = 200):
    """Draw a left sidebar."""
    draw.rectangle([x0, 60, x0 + w, h], fill=_SIDEBAR_BG)
    font = _try_font(13)
    y = 80
    for item in ["Dashboard", "Analytics", "Settings", "Users", "Reports"]:
        draw.text((x0 + 16, y), item, fill=_TEXT, font=font)
        y += 32


def _draw_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
               title: str = "Card", body: str = "Some content here.",
               bg: tuple = _CARD_BG):
    """Draw a rounded-corner-ish card."""
    draw.rectangle([x, y, x + w, y + h], fill=bg, outline=_BORDER, width=1)
    font_t = _try_font(14)
    font_b = _try_font(12)
    draw.text((x + 12, y + 10), title, fill=_TEXT, font=font_t)
    draw.line([(x + 12, y + 30), (x + w - 12, y + 30)], fill=_BORDER, width=1)
    draw.text((x + 12, y + 38), body, fill=(100, 100, 100), font=font_b)


def _draw_button(draw: ImageDraw.ImageDraw, x: int, y: int, label: str = "Submit",
                 bg: tuple = _BUTTON):
    font = _try_font(13)
    tw = len(label) * 8 + 24
    draw.rectangle([x, y, x + tw, y + 32], fill=bg)
    draw.text((x + 12, y + 7), label, fill=(255, 255, 255), font=font)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------


def generate_layout_shift(
    width: int = 1200, height: int = 800
) -> Tuple[Image.Image, Image.Image]:
    """Generate a pair where the current has a layout shift (sidebar wider, cards moved)."""
    # --- Baseline ---
    base = Image.new("RGB", (width, height), _BG)
    d = ImageDraw.Draw(base)
    _draw_header(d, width)
    _draw_sidebar(d, height, x0=0, w=200)
    _draw_card(d, 220, 80, 280, 120, "Revenue", "$12,400 this month")
    _draw_card(d, 520, 80, 280, 120, "Users", "3,241 active")
    _draw_card(d, 220, 220, 580, 200, "Recent Activity", "User signups are up 12%...")
    _draw_button(d, 220, 440, "View Details")

    # --- Current (shifted layout) ---
    curr = Image.new("RGB", (width, height), _BG)
    d2 = ImageDraw.Draw(curr)
    _draw_header(d2, width)
    _draw_sidebar(d2, height, x0=0, w=260)  # wider sidebar
    _draw_card(d2, 280, 80, 280, 120, "Revenue", "$12,400 this month")  # shifted right
    _draw_card(d2, 580, 80, 280, 120, "Users", "3,241 active")
    _draw_card(d2, 280, 220, 560, 200, "Recent Activity", "User signups are up 12%...")
    _draw_button(d2, 280, 440, "View Details")

    return base, curr


def generate_color_change(
    width: int = 1200, height: int = 800
) -> Tuple[Image.Image, Image.Image]:
    """Generate a pair where the current has different header/button colours."""
    base = Image.new("RGB", (width, height), _BG)
    d = ImageDraw.Draw(base)
    _draw_header(d, width, bg=(41, 98, 255))
    _draw_sidebar(d, height)
    _draw_card(d, 220, 80, 280, 120, "Sales", "Quarterly target met")
    _draw_button(d, 220, 220, "Primary Action", bg=(25, 135, 84))

    curr = Image.new("RGB", (width, height), _BG)
    d2 = ImageDraw.Draw(curr)
    _draw_header(d2, width, bg=(220, 53, 69))  # red header
    _draw_sidebar(d2, height)
    _draw_card(d2, 220, 80, 280, 120, "Sales", "Quarterly target met")
    _draw_button(d2, 220, 220, "Primary Action", bg=(255, 193, 7))  # yellow button

    return base, curr


def generate_missing_element(
    width: int = 1200, height: int = 800
) -> Tuple[Image.Image, Image.Image]:
    """Generate a pair where a card is removed in the current version."""
    base = Image.new("RGB", (width, height), _BG)
    d = ImageDraw.Draw(base)
    _draw_header(d, width)
    _draw_sidebar(d, height)
    _draw_card(d, 220, 80, 260, 120, "Notifications", "5 new alerts")
    _draw_card(d, 500, 80, 260, 120, "Messages", "12 unread")
    _draw_card(d, 220, 220, 540, 160, "System Health", "All systems operational")

    curr = Image.new("RGB", (width, height), _BG)
    d2 = ImageDraw.Draw(curr)
    _draw_header(d2, width)
    _draw_sidebar(d2, height)
    _draw_card(d2, 220, 80, 260, 120, "Notifications", "5 new alerts")
    # "Messages" card removed
    # "System Health" card removed -- now blank area

    return base, curr


def generate_new_element(
    width: int = 1200, height: int = 800
) -> Tuple[Image.Image, Image.Image]:
    """Generate a pair where a banner and extra card appear in the current."""
    base = Image.new("RGB", (width, height), _BG)
    d = ImageDraw.Draw(base)
    _draw_header(d, width)
    _draw_sidebar(d, height)
    _draw_card(d, 220, 80, 540, 120, "Overview", "Everything looks good.")

    curr = Image.new("RGB", (width, height), _BG)
    d2 = ImageDraw.Draw(curr)
    _draw_header(d2, width)
    _draw_sidebar(d2, height)
    # New promotional banner
    d2.rectangle([220, 70, 780, 110], fill=(255, 243, 205), outline=(255, 193, 7))
    font = _try_font(13)
    d2.text((230, 80), "NEW: Try our premium plan -- 50% off this month!", fill=(133, 100, 4), font=font)
    _draw_card(d2, 220, 130, 540, 120, "Overview", "Everything looks good.")
    # New card
    _draw_card(d2, 220, 270, 260, 100, "Quick Stats", "42 tasks completed")
    _draw_button(d2, 500, 290, "Upgrade Now", bg=_ACCENT)

    return base, curr


def generate_font_change(
    width: int = 1200, height: int = 800
) -> Tuple[Image.Image, Image.Image]:
    """
    Generate a pair where the current has different font rendering.

    Since we cannot guarantee arbitrary TTF fonts, we simulate a font
    change by drawing text at a different size and slightly different
    position, which produces edge-density differences typical of real
    font swaps.
    """
    base = Image.new("RGB", (width, height), _BG)
    d = ImageDraw.Draw(base)
    _draw_header(d, width)
    _draw_sidebar(d, height)
    font_normal = _try_font(14)
    y = 90
    for line in [
        "Welcome back, Maharshi!",
        "Here is your daily summary.",
        "Revenue is up 8% compared to last week.",
        "3 new team members joined today.",
        "Next meeting: Project Review at 3 PM.",
    ]:
        d.text((230, y), line, fill=_TEXT, font=font_normal)
        y += 28

    curr = Image.new("RGB", (width, height), _BG)
    d2 = ImageDraw.Draw(curr)
    _draw_header(d2, width)
    _draw_sidebar(d2, height)
    font_larger = _try_font(17)  # different size simulates font swap
    y = 88
    for line in [
        "Welcome back, Maharshi!",
        "Here is your daily summary.",
        "Revenue is up 8% compared to last week.",
        "3 new team members joined today.",
        "Next meeting: Project Review at 3 PM.",
    ]:
        d2.text((230, y), line, fill=_TEXT, font=font_larger)
        y += 33

    return base, curr


def generate_no_change(
    width: int = 1200, height: int = 800
) -> Tuple[Image.Image, Image.Image]:
    """Generate an identical pair (should classify as no_change)."""
    img = Image.new("RGB", (width, height), _BG)
    d = ImageDraw.Draw(img)
    _draw_header(d, width)
    _draw_sidebar(d, height)
    _draw_card(d, 220, 80, 540, 120, "Stable Page", "Nothing changed here.")
    _draw_button(d, 220, 220, "OK")
    return img.copy(), img.copy()


# ---------------------------------------------------------------------------
# Batch generation & persistence
# ---------------------------------------------------------------------------

SCENARIOS = {
    "layout_shift": generate_layout_shift,
    "color_change": generate_color_change,
    "missing_element": generate_missing_element,
    "new_element": generate_new_element,
    "font_change": generate_font_change,
    "no_change": generate_no_change,
}


def generate_all_scenarios(
    baselines_dir: str, current_dir: str, width: int = 1200, height: int = 800
) -> Dict[str, Tuple[str, str]]:
    """
    Generate and save all demo scenario images.

    Returns a dict mapping scenario name to (baseline_path, current_path).
    """
    ensure_dir(baselines_dir)
    ensure_dir(current_dir)

    paths: Dict[str, Tuple[str, str]] = {}
    for name, gen_fn in SCENARIOS.items():
        baseline, current = gen_fn(width, height)
        bp = os.path.join(baselines_dir, f"{name}.png")
        cp = os.path.join(current_dir, f"{name}.png")
        save_image(baseline, bp)
        save_image(current, cp)
        paths[name] = (bp, cp)

    return paths
