"""Citation Analysis — which websites and pages AI answers cite most."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src import appkit
from src import metrics as M
from src.metrics import METRIC_DEFINITIONS
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="Citation Analysis", page_icon="🔗", layout="wide")
appkit.ensure_state()

st.title("🔗 Citation Analysis")

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
n = M.total_runs(data.response_runs)

cite = M.citation_rate(data.citations, data.response_runs)
k1, k2, k3 = st.columns(3)
k1.metric("Responses in view", n)
k2.metric("Responses with ≥1 source", cite["runs_with_citations"], help=METRIC_DEFINITIONS["citation_rate"])
k3.metric("Citation rate", f"{round(cite['citation_rate']*100)}%", help=METRIC_DEFINITIONS["citation_rate"])

if data.citations.empty:
    st.info("No citations were extracted from the responses in view.")
    st.stop()

st.divider()
st.subheader("Top cited source domains")
st.caption(METRIC_DEFINITIONS["source_domain_share"])
domains = M.source_domain_share(data.citations, top_n=15)
fig = px.bar(domains, x="citations", y="citation_domain", orientation="h",
             labels={"citations": "Citations", "citation_domain": ""})
fig.update_traces(marker_color="#2563eb")
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
st.plotly_chart(fig, use_container_width=True)

show = domains.copy()
show["domain_share"] = (show["domain_share"] * 100).round(0).astype(int).astype(str) + "%"
show.columns = ["Domain", "Citations", "Responses citing it", "Share of all citations"]
st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Brand-owned vs third-party sources")
brand_domains = set(data.brands["brand_domain"].dropna().astype(str).str.lower()) - {""}
dom = M.source_domain_share(data.citations)
dom["type"] = dom["citation_domain"].apply(lambda d: "Brand-owned" if d.lower() in brand_domains else "Third-party")
by_type = dom.groupby("type")["citations"].sum().reset_index()
if not by_type.empty:
    fig2 = px.pie(by_type, names="type", values="citations", color="type",
                  color_discrete_map={"Brand-owned": "#2563eb", "Third-party": "#94a3b8"})
    fig2.update_layout(height=340)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Third-party sources (review sites, roundups, forums) often shape AI recommendations. "
        "Understanding which ones matter is more actionable than only optimizing your own site."
    )

st.divider()
st.subheader("All cited URLs")
with st.expander("Show every extracted citation"):
    st.dataframe(
        data.citations[["run_id", "citation_url", "citation_domain", "citation_position"]],
        use_container_width=True,
        hide_index=True,
    )

st.info(
    "Next: run the **Page Audit** to inspect the public technical traits of these cited "
    "pages (title, headings, schema, freshness). Traits are associations, not proven causes.",
    icon="➡️",
)
