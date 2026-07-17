"""Render a static dashboard preview PNG from the real demo metrics using PIL.

Run:  python assets/_make_preview.py

This draws a simple, honest snapshot (share of voice + KPIs) from the synthetic demo
data so the README has a visual without needing a browser screenshot tool.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

# Import the project metrics.
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import appkit, metrics as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

BG = (248, 250, 252)
CARD = (255, 255, 255)
INK = (15, 23, 42)
MUTED = (100, 116, 139)
BLUE = (37, 99, 235)
GREY = (148, 163, 184)


def _font(size: int, bold: bool = False):
    """Best-effort system font; falls back to PIL default."""
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"]
    ) + ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def build() -> None:
    data = appkit.load_demo_analysis()
    n = M.total_runs(data.response_runs)
    sov = M.share_of_voice(data.brand_mentions)
    mr = M.brand_mention_rate(data.brand_mentions, data.response_runs)
    cite = M.citation_rate(data.citations, data.response_runs)
    notion_mr = float(mr[mr["brand_name"] == "Notion"]["mention_rate"].iloc[0])

    W, H = 1200, 675
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Header
    d.text((40, 32), "AI Visibility Explorer", font=_font(34, bold=True), fill=INK)
    d.text((40, 78), "Synthetic demo · 5 productivity brands · directional, not definitive",
            font=_font(18), fill=MUTED)

    # KPI cards
    kpis = [
        ("Responses", str(n)),
        ("Prompts", str(data.prompts["prompt_id"].nunique())),
        ("Notion mention rate", f"{round(notion_mr*100)}%"),
        ("Citation rate", f"{round(cite['citation_rate']*100)}%"),
    ]
    x = 40
    for label, value in kpis:
        _card(d, x, 130, 260, 96)
        d.text((x + 20, 148), label, font=_font(16), fill=MUTED)
        d.text((x + 20, 176), value, font=_font(34, bold=True), fill=BLUE)
        x += 278

    # Share of voice bar chart card
    _card(d, 40, 258, 1118, 380)
    d.text((64, 280), "Share of voice", font=_font(22, bold=True), fill=INK)

    top = 330
    row_h = 54
    bar_x = 320
    bar_max = 760
    max_share = float(sov["share_of_voice"].max()) if not sov.empty else 1.0
    for _, row in sov.iterrows():
        brand = row["brand_name"]
        share = float(row["share_of_voice"])
        color = BLUE if brand == "Notion" else GREY
        d.text((64, top + 12), brand, font=_font(18, bold=(brand == "Notion")), fill=INK)
        width = int(bar_max * (share / max_share)) if max_share else 0
        d.rounded_rectangle([bar_x, top + 8, bar_x + max(width, 4), top + 40], radius=6, fill=color)
        d.text((bar_x + width + 12, top + 12), f"{round(share*100)}%", font=_font(18), fill=MUTED)
        top += row_h

    d.text((40, H - 28), "Deterministic extraction · DuckDB · Streamlit · Plotly — Phase 1 MVP",
            font=_font(15), fill=MUTED)

    out = os.path.join(HERE, "dashboard_preview.png")
    img.save(out)
    print(f"Wrote {out}")


def _card(d: "ImageDraw.ImageDraw", x: int, y: int, w: int, h: int) -> None:
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=CARD, outline=(226, 232, 240), width=1)


if __name__ == "__main__":
    build()
