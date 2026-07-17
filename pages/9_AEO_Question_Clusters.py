"""AEO Question Clusters — group questions and see where the brand wins or loses."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src import appkit
from src import clusters as C
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="AEO Question Clusters", page_icon="🗺️", layout="wide")
appkit.ensure_state()

st.title("🗺️ AEO Question Clusters")
st.caption(
    "Group prompts into answer-engine question clusters using your existing metadata — "
    "topic, search intent, persona, journey stage, brand/non-brand, or your own cluster "
    "label — then see where the brand wins, loses, or is absent."
)

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
focal = appkit.focal_brand()

if data.response_runs.empty:
    st.warning("No responses match the current filters. Widen the filters in the sidebar.")
    st.stop()

dimension = st.selectbox(
    "Cluster questions by",
    C.CLUSTER_DIMENSIONS,
    format_func=lambda d: C.DIMENSION_LABELS.get(d, d),
    help="Clusters come from your structured prompt metadata, not keyword matching.",
)

summary = C.cluster_summary(data, focal, dimension)
if summary.empty:
    st.info("No clusters available for this dimension in the current view.")
    st.stop()

# -- Cluster overview --------------------------------------------------------
st.subheader("Cluster overview")
show = summary.copy()
for col in ["focal_mention_rate", "share_of_voice", "recommendation_rate", "citation_rate",
            "competitor_mention_rate", "gap"]:
    show[col] = (show[col] * 100).round(0).astype(int).astype(str) + "%"
show = show[["cluster", "prompts", "runs", "focal_mention_rate", "share_of_voice",
             "recommendation_rate", "citation_rate", "top_competitor",
             "competitor_mention_rate", "gap"]]
show.columns = ["Cluster", "Prompts", "Responses", f"{focal} mention rate", f"{focal} SoV",
                f"{focal} rec. rate", "Citation rate", "Top competitor",
                "Competitor mention rate", "Gap"]
st.dataframe(show, use_container_width=True, hide_index=True)

fig = px.bar(
    summary, x="gap", y="cluster", orientation="h",
    labels={"gap": f"Competitor lead over {focal} (mention-rate gap)", "cluster": ""},
)
fig.update_traces(marker_color="#2563eb")
fig.update_layout(xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"}, height=360)
st.plotly_chart(fig, use_container_width=True)
st.caption("Positive gap = the strongest competitor appears in more responses than the focal brand.")

st.divider()

# -- Single cluster deep dive ------------------------------------------------
cluster_value = st.selectbox("Inspect a cluster", summary["cluster"].tolist())
row = summary[summary["cluster"] == cluster_value].iloc[0]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Prompts", int(row["prompts"]))
k2.metric("Responses", int(row["runs"]))
k3.metric(f"{focal} mention rate", f"{round(row['focal_mention_rate']*100)}%")
k4.metric(f"{focal} share of voice", f"{round(row['share_of_voice']*100)}%")
k5.metric("Citation rate", f"{round(row['citation_rate']*100)}%")

if int(row["runs"]) < 5:
    st.warning(f"⚠ Only {int(row['runs'])} responses in this cluster — treat these rates as exploratory.")

# Competitor rates within the cluster.
st.markdown("##### All brands in this cluster")
comp = C.competitor_rates_in_cluster(data, dimension, cluster_value)
if not comp.empty:
    cshow = comp.copy()
    for col in ["mention_rate", "share_of_voice", "recommendation_rate"]:
        cshow[col] = (cshow[col] * 100).round(0).astype(int).astype(str) + "%"
    cshow.columns = ["Brand", "Mention rate", "Share of voice", "Recommendation rate"]
    st.dataframe(cshow, use_container_width=True, hide_index=True)

# -- Question outcomes -------------------------------------------------------
st.markdown("##### Question-by-question outcomes")
outcomes = C.question_outcomes(data, focal, dimension, cluster_value)
if outcomes.empty:
    st.info("No responses for the questions in this cluster.")
else:
    tabs = st.tabs([
        f"✅ {focal} wins", "🏁 Competitors win", "⚪ No tracked brand", "All questions",
    ])
    groups = [
        (tabs[0], outcomes[outcomes["outcome"] == C.OUTCOME_FOCAL_WINS]),
        (tabs[1], outcomes[outcomes["outcome"] == C.OUTCOME_COMPETITOR_WINS]),
        (tabs[2], outcomes[outcomes["outcome"] == C.OUTCOME_NO_BRAND]),
        (tabs[3], outcomes),
    ]
    for tab, df in groups:
        with tab:
            if df.empty:
                st.write("— none —")
                continue
            d = df.copy()
            d["focal_rate"] = (d["focal_rate"] * 100).round(0).astype(int).astype(str) + "%"
            d["competitor_rate"] = (d["competitor_rate"] * 100).round(0).astype(int).astype(str) + "%"
            d = d[["prompt_id", "prompt_text", "runs", "focal_rate", "top_competitor", "competitor_rate", "outcome"]]
            d.columns = ["ID", "Question", "Responses", f"{focal} rate", "Top competitor", "Competitor rate", "Outcome"]
            st.dataframe(d, use_container_width=True, hide_index=True)

# -- Coverage gaps within the cluster ---------------------------------------
st.markdown("##### Content coverage gaps in this cluster")
gaps = C.cluster_coverage_gaps(data, focal, dimension, cluster_value)
gaps = gaps[gaps["gap"] > 0]
if gaps.empty:
    st.success(f"No topics inside this cluster where a competitor out-appears {focal}.")
else:
    g = gaps.copy()
    for col in ["focal_rate", "competitor_rate", "gap"]:
        g[col] = (g[col] * 100).round(0).astype(int).astype(str) + "%"
    g.columns = ["Topic", f"{focal} rate", "Top competitor", "Competitor rate", "Gap", "Responses"]
    st.dataframe(g, use_container_width=True, hide_index=True)

# -- Page consolidation recommendation --------------------------------------
st.divider()
st.subheader("One page or several?")
rec = C.page_consolidation_recommendation(data, dimension, cluster_value)
st.success(f"**Recommendation: {rec['recommendation']}**")
st.write(rec["reason"])
ev = rec.get("evidence") or {}
if ev:
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Questions", ev.get("questions", 0))
    e2.metric("Search intents", len(ev.get("search_intents", [])))
    e3.metric("Journey stages", len(ev.get("journey_stages", [])))
    e4.metric("Personas", len(ev.get("personas", [])))
    with st.expander("Evidence and the exact rules used"):
        st.markdown(f"- **Search intents:** {', '.join(ev.get('search_intents', [])) or '—'}")
        st.markdown(f"- **Journey stages:** {', '.join(ev.get('journey_stages', [])) or '—'}")
        st.markdown(f"- **Personas:** {', '.join(ev.get('personas', [])) or '—'}")
        st.markdown(f"**Rules:** {rec['rules']}")
st.caption(
    "This recommendation is a deterministic rule over the cluster's own metadata spread. "
    "It is a starting point for editorial judgement, not a guarantee of ranking or citation."
)

st.info(
    "Next: open **Content Action Briefs** for briefs on these gaps, or **AEO Experiments** to "
    "measure what changes after you ship content for a cluster.",
    icon="➡️",
)
