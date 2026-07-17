"""Customer Readout — a plain-language, metric-grounded summary the customer can read."""

from __future__ import annotations

import os

import streamlit as st

from src import appkit
from src.recommendations import build_readout, readout_to_markdown
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="Customer Readout", page_icon="📝", layout="wide")
appkit.ensure_state()

st.title("📝 Customer-Facing Readout")
st.caption(
    "A plain-language summary a marketing or customer-success team can read without "
    "touching the code. Every statement is grounded in a computed metric."
)

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
focal = appkit.focal_brand()
project_name = data.projects.iloc[0]["project_name"] if not data.projects.empty else "Analysis"

readout = build_readout(data, focal)

st.info(
    "This readout is generated from **deterministic templates** grounded in the metrics — "
    "no findings are invented. Demo responses are synthetic, not real platform outputs.",
    icon="🧾",
)

SECTIONS = [
    ("Executive summary", readout.executive_summary, "🎯"),
    ("Strongest visibility areas", readout.strongest_areas, "💪"),
    ("Weakest visibility areas", readout.weakest_areas, "🔻"),
    ("Competitors ahead / gaining", readout.competitors_gaining, "🏁"),
    ("Frequently cited sources", readout.frequent_sources, "🔗"),
    ("Content gaps", readout.content_gaps, "🕳️"),
    ("Recommended next actions", readout.next_actions, "✅"),
    ("Limitations & confidence", readout.limitations_confidence, "⚠️"),
]

for title, items, icon in SECTIONS:
    st.subheader(f"{icon} {title}")
    for item in items:
        st.markdown(f"- {item}")
    st.write("")

st.divider()

# -- Export -----------------------------------------------------------------
md = readout_to_markdown(readout, focal, project_name)
st.download_button(
    "⬇️ Download this readout (Markdown)",
    data=md,
    file_name=f"ai_visibility_readout_{focal.replace('.', '_')}.md",
    mime="text/markdown",
)

with st.expander("Preview export (Markdown)"):
    st.code(md, language="markdown")

# -- Optional AI narrative (clearly gated + labelled) -----------------------
st.divider()
st.subheader("Optional: AI-generated narrative")
has_key = any(os.environ.get(k) for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"])
if not has_key:
    st.caption(
        "No LLM API key detected. The deterministic readout above is the default and needs "
        "no key. To enable an optional AI-written narrative, add a key to `.env`. Any AI "
        "narrative would be clearly labelled *AI-generated* and constrained to the metrics above."
    )
else:
    st.caption(
        "An API key is present. AI narrative generation is a Phase-2 feature and is not wired "
        "up in this MVP; the deterministic readout remains the source of truth."
    )
