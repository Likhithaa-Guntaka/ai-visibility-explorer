"""Visibility Dashboard — the main interactive view of all headline metrics."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src import appkit
from src import metrics as M
from src.metrics import METRIC_DEFINITIONS
from src.sql_metrics import SqlMetrics
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="Visibility Dashboard", page_icon="📊", layout="wide")
appkit.ensure_state()

st.title("📊 Visibility Dashboard")

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
focal = appkit.focal_brand()
n = M.total_runs(data.response_runs)

if n == 0:
    st.warning("No responses match the current filters. Widen the filters in the sidebar.")
    st.stop()

if M.is_small_sample(n):
    st.warning(
        f"⚠ Only **{n} responses** in view. Percentages are exploratory and can swing "
        "with one more answer. See the **Limitations & Confidence** page.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Compute the headline metrics with DuckDB SQL (see src/sql_metrics.py + sql/).
# All frames are materialised inside one connection, then rendered with pandas/Plotly.
# ---------------------------------------------------------------------------
with SqlMetrics(data) as sqlm:
    mr = sqlm.brand_mention_rate()
    sov = sqlm.share_of_voice()
    cite = sqlm.citation_rate()
    fms = sqlm.first_mention_share()
    cat = sqlm.visibility_by_attribute("prompt_category", focal)
    per = sqlm.visibility_by_attribute("persona", focal)
    plat = sqlm.platform_comparison(focal)
    lb = sqlm.competitor_visibility()

# ---------------------------------------------------------------------------
# Headline KPI row
# ---------------------------------------------------------------------------
focal_mr = float(mr[mr["brand_name"] == focal]["mention_rate"].iloc[0]) if (not mr.empty and focal in set(mr["brand_name"])) else 0.0
focal_sov = float(sov[sov["brand_name"] == focal]["share_of_voice"].iloc[0]) if (not sov.empty and focal in set(sov["brand_name"])) else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total prompts", data.prompts["prompt_id"].nunique())
k2.metric("Total responses", n)
k3.metric(f"{focal} mention rate", f"{round(focal_mr*100)}%", help=METRIC_DEFINITIONS["brand_mention_rate"])
k4.metric(f"{focal} share of voice", f"{round(focal_sov*100)}%", help=METRIC_DEFINITIONS["share_of_voice"])
k5.metric("Citation rate", f"{round(cite['citation_rate']*100)}%", help=METRIC_DEFINITIONS["citation_rate"])

st.divider()

# Consistent color: focal brand highlighted, others muted.
def _brand_color_map(brand_names):
    return {b: ("#2563eb" if b == focal else "#94a3b8") for b in brand_names}


# ---------------------------------------------------------------------------
# Row: Share of voice + First mention share
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Share of voice")
    st.caption(METRIC_DEFINITIONS["share_of_voice"])
    if not sov.empty:
        fig = px.bar(sov, x="share_of_voice", y="brand_name", orientation="h",
                     color="brand_name", color_discrete_map=_brand_color_map(sov["brand_name"]),
                     labels={"share_of_voice": "Share of voice", "brand_name": ""})
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"}, height=340)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No mentions in view.")

with c2:
    st.subheader("First mention share")
    st.caption(METRIC_DEFINITIONS["first_mention_share"])
    if not fms.empty:
        fig = px.bar(fms, x="first_mention_share", y="brand_name", orientation="h",
                     color="brand_name", color_discrete_map=_brand_color_map(fms["brand_name"]),
                     labels={"first_mention_share": "First mention share", "brand_name": ""})
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"}, height=340)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No first-mention data in view.")

# ---------------------------------------------------------------------------
# Row: Visibility by category + by persona (focal brand)
# ---------------------------------------------------------------------------
# `enriched` (pandas) is still needed for content-coverage gaps and the prompt-level
# table below, which are computed in pandas — see the note near the bottom of the page.
enriched = M.enrich_mentions(data.brand_mentions, data.response_runs, data.prompts)
c3, c4 = st.columns(2)
with c3:
    st.subheader(f"{focal} — visibility by prompt category")
    st.caption(METRIC_DEFINITIONS["prompt_category_performance"])
    if not cat.empty:
        fig = px.bar(cat, x="mention_rate", y="prompt_category", orientation="h",
                     labels={"mention_rate": "Mention rate", "prompt_category": ""})
        fig.update_traces(marker_color="#2563eb")
        fig.update_layout(xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"}, height=340)
        st.plotly_chart(fig, width="stretch")

with c4:
    st.subheader(f"{focal} — visibility by persona")
    st.caption(METRIC_DEFINITIONS["persona_performance"])
    if not per.empty:
        fig = px.bar(per, x="mention_rate", y="persona", orientation="h",
                     labels={"mention_rate": "Mention rate", "persona": ""})
        fig.update_traces(marker_color="#2563eb")
        fig.update_layout(xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"}, height=340)
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Row: Platform comparison
# ---------------------------------------------------------------------------
st.subheader(f"{focal} — platform comparison")
st.caption(METRIC_DEFINITIONS["platform_comparison"])
if not plat.empty:
    fig = px.bar(plat, x="platform", y="mention_rate", labels={"mention_rate": "Mention rate", "platform": ""})
    fig.update_traces(marker_color="#2563eb")
    fig.update_layout(yaxis_tickformat=".0%", height=320)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Platform labels are as-entered. In the demo they are synthetic labels, not real "
        "platform outputs."
    )

st.divider()

# ---------------------------------------------------------------------------
# Competitive leaderboard table
# ---------------------------------------------------------------------------
st.subheader("Competitive visibility")
st.caption(METRIC_DEFINITIONS["competitor_visibility"])
if not lb.empty:
    show = lb.copy()
    show["mention_rate"] = (show["mention_rate"] * 100).round(0).astype(int).astype(str) + "%"
    show["share_of_voice"] = (show["share_of_voice"] * 100).round(0).astype(int).astype(str) + "%"
    show["recommendation_rate"] = (show["recommendation_rate"] * 100).round(0).astype(int).astype(str) + "%"
    show.columns = ["Brand", "Mention rate", "Share of voice", "Recommendation rate"]
    st.dataframe(show, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Content coverage gaps
# ---------------------------------------------------------------------------
st.subheader(f"Content coverage gaps for {focal}")
st.caption(METRIC_DEFINITIONS["content_coverage_gaps"])
gaps = M.content_coverage_gaps(enriched, data.response_runs, data.prompts, focal, dimension="topic")
gaps_pos = gaps[gaps["gap"] > 0]
if not gaps_pos.empty:
    show = gaps_pos.copy()
    for col in ["focal_rate", "competitor_rate", "gap"]:
        show[col] = (show[col] * 100).round(0).astype(int).astype(str) + "%"
    show = show[["topic", "focal_rate", "top_competitor", "competitor_rate", "gap", "total_runs"]]
    show.columns = ["Topic", f"{focal} rate", "Top competitor", "Competitor rate", "Gap", "Responses"]
    st.dataframe(show, width="stretch", hide_index=True)
else:
    st.success(f"No topics where a competitor beats {focal} in the current view.")

# ---------------------------------------------------------------------------
# Prompt-level results table
# ---------------------------------------------------------------------------
with st.expander("Prompt-level results (raw)"):
    if not enriched.empty:
        pl = (
            enriched.groupby(["prompt_id", "brand_name"])
            .agg(runs=("run_id", "nunique"), mentions=("mention_count", "sum"))
            .reset_index()
        )
        prompt_text = data.prompts.set_index("prompt_id")["prompt_text"].to_dict()
        pl["prompt"] = pl["prompt_id"].map(prompt_text)
        st.dataframe(pl[["prompt_id", "prompt", "brand_name", "runs", "mentions"]], width="stretch", hide_index=True)

with st.expander("ℹ️ Metric definitions"):
    for _key, definition in METRIC_DEFINITIONS.items():
        st.markdown(f"- {definition}")

st.caption(
    "⚙️ Engine: the headline metrics on this page — mention rate, share of voice, first-mention "
    "share, citation rate, category & persona performance, platform comparison and the competitive "
    "leaderboard — are computed in **DuckDB SQL** (`src/sql_metrics.py`, `sql/*.sql`). Content "
    "coverage gaps and the prompt-level table use **pandas**. Both engines share the same metric "
    "definitions and are checked for equivalence in `tests/test_sql_metrics.py`."
)
