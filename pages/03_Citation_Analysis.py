"""Citation Analysis — which websites and pages AI answers cite most."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src import appkit
from src import metrics as M
from src.metrics import METRIC_DEFINITIONS
from src.sql_metrics import SqlMetrics
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="Citation Analysis", page_icon="🔗", layout="wide")
appkit.ensure_state()

st.title("🔗 Citation Analysis")

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
n = M.total_runs(data.response_runs)

# Citation rate + top source domains are computed in DuckDB SQL (src/sql_metrics.py).
with SqlMetrics(data) as sqlm:
    cite = sqlm.citation_rate()
    domains = sqlm.source_domain_share(top_n=15)
k1, k2, k3 = st.columns(3)
k1.metric("Responses in view", n)
k2.metric("Responses with ≥1 source", cite["runs_with_citations"], help=METRIC_DEFINITIONS["citation_rate"])
k3.metric("Citation rate", f"{round(cite['citation_rate']*100)}%", help=METRIC_DEFINITIONS["citation_rate"])

if data.citations.empty:
    st.info("No citations were extracted from the responses in view.")
    st.stop()

st.divider()
st.subheader("Top cited source domains")
st.caption(METRIC_DEFINITIONS["source_domain_share"] + "  ·  Computed in DuckDB SQL.")
fig = px.bar(domains, x="citations", y="citation_domain", orientation="h",
             labels={"citations": "Citations", "citation_domain": ""})
fig.update_traces(marker_color="#2563eb")
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
st.plotly_chart(fig, width="stretch")

show = domains.copy()
show["domain_share"] = (show["domain_share"] * 100).round(0).astype(int).astype(str) + "%"
show.columns = ["Domain", "Citations", "Responses citing it", "Share of all citations"]
st.dataframe(show, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Source classification, quality, and opportunities (upgrade features).
# ---------------------------------------------------------------------------
from src import citation_quality as CQ  # noqa: E402

focal = appkit.focal_brand()
classified = CQ.classify_citations(data.citations, data.brands, focal)

st.divider()
st.subheader("Citation quality")
div = CQ.citation_diversity(data.citations)
conc = CQ.citation_concentration(data.citations)
bt = CQ.brand_owned_vs_third_party(classified)
q1, q2, q3, q4 = st.columns(4)
q1.metric("Unique domains", div["unique_domains"], help="Diversity = unique domains ÷ total citations. Higher = more varied sources.")
q2.metric("Diversity", f"{round(div['diversity']*100)}%", help="Unique domains as a share of all citations.")
q3.metric("Concentration (HHI)", f"{conc['hhi']:.2f}", help="Herfindahl index of domain shares. 1.0 = one domain dominates; lower = spread out.")
q4.metric("Top-3 domain share", f"{round(conc['top3_share']*100)}%", help="Share of citations held by the three most-cited domains.")

st.divider()
st.subheader(f"Source types influencing answers (relative to {focal})")
st.caption(
    "Each cited domain is classified into a transparent source type. Brand/Competitor "
    "owned are relative to the focal brand."
)
stb = CQ.source_type_breakdown(classified)
c1, c2 = st.columns([3, 2])
with c1:
    fig2 = px.bar(stb, x="citations", y="source_type", orientation="h",
                  labels={"citations": "Citations", "source_type": ""})
    fig2.update_traces(marker_color="#2563eb")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=340)
    st.plotly_chart(fig2, width="stretch")
with c2:
    st.metric("Brand-owned share", f"{round(bt['brand_owned_share']*100)}%")
    st.metric("Competitor-owned share", f"{round(bt['competitor_owned_share']*100)}%")
    st.metric("Third-party share", f"{round(bt['third_party_share']*100)}%")

st.divider()
st.subheader("Citation opportunities")
st.caption(
    "Third-party domains cited alongside competitors more often than the focal brand — "
    "places worth investigating for coverage or presence. **Association, not causation.**"
)
opps = CQ.citation_opportunities(data.citations, data.brand_mentions, data.brands, focal, top_n=12)
opps = opps[opps["opportunity_gap"] > 0]
if not opps.empty:
    show = opps.rename(columns={
        "citation_domain": "Domain", "source_type": "Type", "runs_citing": "Responses citing",
        "runs_with_focal": f"…with {focal}", "runs_with_competitor": "…with a competitor",
        "opportunity_gap": "Opportunity gap",
    })
    st.dataframe(show, width="stretch", hide_index=True)
    st.info(
        f"These sources appear alongside competitors more than {focal}. Consider earning accurate, "
        "up-to-date coverage on them, and see the **Content Action Briefs** page for concrete next steps.",
        icon="💡",
    )
else:
    st.success(f"No clear third-party citation gaps for {focal} in the current view.")

st.divider()
st.subheader("All cited URLs")
with st.expander("Show every extracted citation (with source type)"):
    st.dataframe(
        classified[["run_id", "citation_url", "citation_domain", "source_type", "citation_position"]],
        width="stretch",
        hide_index=True,
    )

st.info(
    "Next: run the **Page Audit** for an AI Answer Readiness check of these cited pages "
    "(direct answers, question headings, schema, freshness). Traits are associations, not proven causes.",
    icon="➡️",
)
