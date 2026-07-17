"""Page Audit — optional public inspection of cited URLs (SEO / content signals)."""

from __future__ import annotations

import streamlit as st

from src import appkit
from src.page_audit import audit_urls, summarize_audits
from src.ui import require_data

st.set_page_config(page_title="Page Audit", page_icon="🧪", layout="wide")
appkit.ensure_state()

st.title("🧪 Page Audit")
st.caption(
    "Optionally inspect public technical traits of the pages AI answers cite: robots.txt, "
    "sitemap, canonical, title, headings, schema.org types, dates, and word count."
)

st.warning(
    "**Association, not causation.** These traits describe cited pages; they do **not** "
    "prove that any trait caused an AI citation. This makes a network request to each URL "
    "with a polite, identifiable User-Agent and a short timeout. It only works when live "
    "internet access is available (e.g. locally or in Codespaces).",
    icon="⚠️",
)

data = require_data(st)
if data is None:
    st.stop()

if data.citations.empty:
    st.info("No cited URLs to audit. Load data with citations first.")
    st.stop()

urls = data.citations["citation_url"].dropna().unique().tolist()
st.write(f"**{len(urls)} unique cited URL(s)** available to audit.")
max_urls = st.slider("How many URLs to audit (keep small to be polite)", 1, min(len(urls), 25), min(5, len(urls)))
selected = urls[:max_urls]

with st.expander("URLs that will be audited"):
    for u in selected:
        st.write(u)

if st.button("▶ Run page audit", type="primary"):
    with st.spinner(f"Auditing {len(selected)} page(s)…"):
        audits = audit_urls(selected)
        st.session_state["page_audits"] = audits
    st.success("Audit complete.")

audits = st.session_state.get("page_audits")
if audits is not None and not audits.empty:
    st.divider()
    st.subheader("Audit results")

    status_counts = audits["audit_status"].value_counts().to_dict()
    ok = status_counts.get("ok", 0)
    st.write(
        f"Successfully audited **{ok} of {len(audits)}** pages. "
        f"Status breakdown: {status_counts}"
    )
    if "unavailable" in status_counts:
        st.warning(
            "Some/all audits returned `unavailable` — this environment may not have live "
            "internet access. Try running locally or in Codespaces."
        )

    enriched = summarize_audits(audits, data.citations)
    display_cols = [
        "citation_url", "citation_domain", "times_cited", "audit_status", "page_title",
        "h1_count", "h2_count", "h3_count", "schema_types", "word_count", "external_link_count",
        "question_heading_count", "answer_upfront", "has_author",
        "published_date", "modified_date", "robots_accessible", "sitemap_found", "canonical_url",
    ]
    display_cols = [c for c in display_cols if c in enriched.columns]
    st.dataframe(enriched[display_cols], use_container_width=True, hide_index=True)

    # -- AI Answer Readiness, per page, factor-by-factor --------------------
    st.divider()
    st.subheader("AI Answer Readiness — per page")
    st.caption(
        "Each of 12 factors is shown separately. The optional summary score is fully "
        "transparent: its exact formula, weights, and per-factor points are always displayed. "
        "There is deliberately no opaque overall score."
    )
    from src import page_audit as PA  # noqa: E402

    _STATUS_ICON = {"pass": "✅", "partial": "🟡", "fail": "❌", "unknown": "⚪"}
    url_options = enriched["citation_url"].tolist()
    chosen = st.selectbox("Choose an audited page", url_options)
    row = enriched[enriched["citation_url"] == chosen].iloc[0].to_dict()

    score = PA.readiness_score(row)
    if score["score"] is not None:
        st.metric("Readiness score (transparent)", f"{round(score['score'])}/100",
                  help="Optional weighted score — formula and components shown below.")
    else:
        st.info("Not enough was observable on this page to compute a score (e.g. blocked/timed out).")

    factors = PA.readiness_factors(row)
    for f in factors:
        weight = PA.READINESS_WEIGHTS.get(f["factor"], 0)
        st.markdown(f"{_STATUS_ICON[f['status']]} **{f['factor']}** ({f['status']}, weight {weight}) — {f['observed']}")

    with st.expander("Show the exact scoring formula, weights, and points"):
        st.markdown(f"**Formula:** {score['formula']}")
        comp_rows = [
            {
                "Factor": c["factor"], "Status": c["status"], "Weight": c["weight"],
                "Credit": c["credit"], "Points": c["points"],
            }
            for c in score["components"]
        ]
        st.dataframe(comp_rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Points earned {score['points_earned']:.1f} ÷ points considered "
            f"{score['points_considered']:.1f} × 100. Unknown factors are excluded from both."
        )

    st.caption(
        "Pattern to *investigate*, not conclude: do frequently-cited pages tend to score higher? "
        "Treat any correlation as a hypothesis. Technical traits do not prove causation of AI citations."
    )
else:
    st.info("No audit has been run yet. Click **Run page audit** above.")
