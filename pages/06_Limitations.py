"""Limitations & Confidence — the honesty page. Always visible, never optional."""

from __future__ import annotations

import streamlit as st

from src import appkit
from src import metrics as M

st.set_page_config(page_title="Limitations & Confidence", page_icon="⚠️", layout="wide")
appkit.ensure_state()

st.title("⚠️ Limitations & Confidence")
st.caption("Read this before treating any number here as a conclusion.")

# -- Data-driven confidence snapshot ----------------------------------------
if appkit.has_data():
    data = appkit.get_data()
    n = M.total_runs(data.response_runs)
    n_prompts = data.prompts["prompt_id"].nunique()
    n_platforms = data.response_runs["platform"].nunique()
    consistency = M.consistency_summary(
        M.response_consistency(data.brand_mentions, data.citations, data.response_runs)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responses", n)
    c2.metric("Prompts", n_prompts)
    c3.metric("Platform labels", n_platforms)
    overlap = consistency["avg_brand_overlap"]
    c4.metric("Avg brand overlap (repeats)", f"{round(overlap*100)}%" if overlap is not None else "n/a")

    if M.is_small_sample(n):
        st.error(
            f"⚠ Small sample: only {n} responses. A single additional response can move a "
            "percentage by several points. Treat everything as directional.",
            icon="⚠️",
        )
    if consistency["prompts_with_repeats"] == 0:
        st.warning("No prompts were run more than once, so run-to-run consistency cannot be measured here.")
    elif overlap is not None and overlap < 0.7:
        st.warning(
            f"Repeated runs of the same prompt agreed on the brand set only "
            f"{round(overlap*100)}% of the time on average — results are noisy."
        )
else:
    st.info("No data loaded — the general limitations below still apply to any analysis.")

st.divider()

st.markdown(
    """
### What this tool can and cannot tell you

This is an **exploratory measurement tool**, not a definitive ranking system. AI-search
answers are noisy, change frequently, and depend heavily on how prompts are written.
The following limitations always apply.
"""
)

LIMITATIONS = [
    ("Small sample sizes",
     "With few responses, one answer can swing a percentage dramatically. We flag any metric "
     "based on fewer than a handful of responses."),
    ("Prompt selection bias",
     "Results reflect the specific prompts you chose. A different (equally reasonable) prompt "
     "set could produce very different visibility numbers."),
    ("Platform variability",
     "The same question can yield very different answers on different AI platforms, and even "
     "on the same platform at different times."),
    ("Model updates",
     "Providers update their models frequently. A measurement taken today may not hold next month."),
    ("Personalization",
     "Some platforms personalize answers by account, location, or history. Your results may not "
     "generalize to every user."),
    ("Missing citations",
     "Many answers include no source links, so citation metrics only reflect the subset that do."),
    ("Incomplete source access",
     "Page audits can be blocked, rate-limited, or time out, so some pages will be missing traits."),
    ("Differences across repeated runs",
     "The same prompt run twice can return different brands and sources. We measure this with "
     "simple overlap metrics rather than over-claiming stability."),
    ("Correlation vs. causation",
     "Technical page traits are associations only. We never claim a trait *caused* an AI citation."),
    ("Non-representative prompt sets",
     "A demo or small prompt set is illustrative, not a market-representative survey."),
]

for title, body in LIMITATIONS:
    with st.container(border=True):
        st.markdown(f"**{title}** — {body}")

st.divider()
st.markdown(
    """
### How consistency is measured (kept simple on purpose)

For prompts run more than once we use transparent, explainable measures instead of heavy
statistics that a small sample can't support:

- **Brand overlap** — how similar the set of mentioned brands is across runs (Jaccard).
- **Recommendation agreement** — how similar the set of *recommended* brands is across runs.
- **Citation domain overlap** — how similar the cited domains are across runs.
- **Mention count variation** — how much the total number of mentions varies (coefficient of variation).

We deliberately avoid confidence intervals and significance tests unless the sample is large
enough to justify them. **Never present exploratory results as definitive conclusions.**
"""
)

st.success(
    "Bottom line: use these findings to form hypotheses and prioritize experiments — then "
    "validate with larger, repeated, well-designed prompt sets over time.",
    icon="✅",
)
