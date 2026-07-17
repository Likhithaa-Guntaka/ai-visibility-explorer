"""Content Action Briefs — deterministic, grounded briefs for each coverage gap."""

from __future__ import annotations

import streamlit as st

from src import appkit
from src import briefs as B
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="Content Action Briefs", page_icon="🧱", layout="wide")
appkit.ensure_state()

st.title("🧱 Content Action Briefs")
st.caption(
    "For each content-coverage gap, a ready-to-use content brief — grounded entirely in "
    "the calculated gaps, the real prompts behind each topic, and the citation-opportunity analysis."
)

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
focal = appkit.focal_brand()

max_briefs = st.slider("How many briefs (top gaps)", 1, 10, 5)
briefs = B.build_briefs(data, focal, max_briefs=max_briefs)

st.info(
    "Briefs are **deterministic templates**. Every field is derived from data — nothing is "
    "invented. Edit them freely before handing to a content team.",
    icon="🧾",
)

if not briefs:
    st.success(f"No content gaps for {focal} in the current view — competitors are not out-appearing it.")
    st.stop()

st.write(f"**{len(briefs)} brief(s)** for **{focal}**, most severe gap first.")

for i, b in enumerate(briefs, start=1):
    with st.expander(f"{i}. {b.topic} — {b.suggested_title}", expanded=(i == 1)):
        st.markdown(f"**Gap:** {b.gap_summary}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Target persona**\n\n{b.target_persona}")
        c2.markdown(f"**Journey stage**\n\n{b.journey_stage}")
        c3.markdown(f"**Prompt category**\n\n{b.prompt_category}")
        c4, c5 = st.columns(2)
        c4.markdown(f"**Suggested format**\n\n{b.suggested_format}")
        c5.markdown(f"**Recommended schema**\n\n`{b.recommended_schema}`")

        st.markdown("**Questions to answer:**")
        for q in b.questions_to_answer:
            st.markdown(f"- {q}")

        st.markdown("**Suggested headings:**")
        for h in b.suggested_headings:
            st.markdown(f"- {h}")

        st.markdown(f"**Evidence / examples needed:** {b.evidence_needed}")

        st.markdown("**Relevant third-party source opportunities:**")
        if b.source_opportunities:
            for s in b.source_opportunities:
                st.markdown(f"- {s}")
            st.caption("Association only — cited alongside competitors. Investigate; do not assume causation.")
        else:
            st.markdown("- None identified for this topic.")

# -- Export all briefs -------------------------------------------------------
st.divider()
all_md = "\n\n---\n\n".join(B.brief_to_markdown(b) for b in briefs)
header = f"# Content Action Briefs — {focal}\n\n_Grounded in calculated gaps and citations. Directional, not definitive._\n\n"
st.download_button(
    "⬇️ Download all briefs (Markdown)",
    data=header + all_md,
    file_name=f"content_briefs_{focal.replace('.', '_')}.md",
    mime="text/markdown",
)
