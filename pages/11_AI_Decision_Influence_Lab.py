"""AI Decision Influence Lab — why brands are selected or rejected, and what to do next.

A research extension beyond visibility monitoring. Six tabs investigate the *decision*:
recommendation outcomes, claim provenance, the customer journey, truth & freshness,
evidence opportunities, and a transparent priority framework. All findings are
associations, not proof of causation; synthetic demo data is labelled as synthetic.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.express as px
import streamlit as st

from src import appkit
from src import case_study as CS
from src import claims as CL
from src import decision_lab as DL
from src import evidence_engine as EV
from src import journeys as JN
from src import prioritization as PR
from src import truth_monitor as TM
from src.database import BRAND_FACTS_COLUMNS, FACT_TYPES
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="AI Decision Influence Lab", page_icon="🔬", layout="wide")
appkit.ensure_state()

st.title("🔬 AI Decision Influence Lab")
st.caption(
    "Beyond *whether* a brand appears — investigate **why** it was recommended or rejected, "
    "which claims and sources shaped the answer, and where it drops out of the decision journey."
)
st.warning(
    "Everything here is **association, not proof of causation**. A source appearing alongside a "
    "recommendation does not mean it caused it. The Truth tab is an *authoritative-source "
    "comparison*, not a verification of absolute truth.",
    icon="⚠️",
)

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
focal = appkit.focal_brand()

if data.recommendation_outcomes.empty:
    st.info("No decision signals yet. Load the demo, or add responses and run extraction on **Data Input**.")
    st.stop()

tab_out, tab_claim, tab_journey, tab_truth, tab_evi, tab_prio = st.tabs([
    "1 · Recommendation Outcomes",
    "2 · Claim Provenance",
    "3 · Customer Journey",
    "4 · Truth & Freshness",
    "5 · Evidence Opportunities",
    "6 · Priority Framework",
])

# ---------------------------------------------------------------------------
# Tab 1 — Recommendation Outcomes
# ---------------------------------------------------------------------------
with tab_out:
    st.subheader("Recommendation outcomes")
    summ = DL.outcome_summary(data.recommendation_outcomes)
    if not summ.empty:
        show = summ.copy()
        for c in ["mention_to_recommendation_rate", "comparison_survival_rate", "rejection_rate"]:
            show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
        show = show[["brand_name", "recommended", "not_recommended", "rejected", "not_mentioned",
                     "mention_to_recommendation_rate", "comparison_survival_rate", "rejection_rate"]]
        show.columns = ["Brand", "Recommended", "Mentioned not rec.", "Rejected", "Not mentioned",
                        "Mention→Rec", "Comparison survival", "Rejection rate"]
        st.dataframe(show, width="stretch", hide_index=True)

        # Stacked outcome mix per brand.
        mix = data.recommendation_outcomes.groupby(["brand_name", "outcome"]).size().rename("n").reset_index()
        fig = px.bar(mix, x="n", y="brand_name", color="outcome", orientation="h",
                     labels={"n": "Responses", "brand_name": ""})
        fig.update_layout(height=320, legend_title_text="Outcome")
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"##### Why {focal} is rejected — reason categories")
    reasons = DL.rejection_reason_totals(data.recommendation_outcomes, focal)
    if reasons.empty:
        st.success(f"No 'compared but rejected' outcomes for {focal} in this view.")
    else:
        st.dataframe(reasons.rename(columns={"reason": "Rejection reason", "count": "Responses"}),
                     width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Rejection reasons by persona")
            st.dataframe(DL.rejection_reasons_by(data.recommendation_outcomes, data.response_runs, data.prompts, "persona", focal),
                         width="stretch", hide_index=True)
        with c2:
            st.caption("Rejection reasons by journey stage")
            st.dataframe(DL.rejection_reasons_by(data.recommendation_outcomes, data.response_runs, data.prompts, "journey_stage", focal),
                         width="stretch", hide_index=True)

    st.markdown(f"##### {focal} conversion by platform & question cluster")
    c3, c4 = st.columns(2)
    with c3:
        st.dataframe(DL.conversion_by(data.recommendation_outcomes, data.response_runs, data.prompts, "platform", focal),
                     width="stretch", hide_index=True)
    with c4:
        st.dataframe(DL.conversion_by(data.recommendation_outcomes, data.response_runs, data.prompts, "question_cluster", focal),
                     width="stretch", hide_index=True)

    with st.expander("✏️ Correct outcome classifications (editable)"):
        st.caption("Every classification is editable. Evidence text shows the exact snippet the rule used.")
        edited = st.data_editor(data.recommendation_outcomes, num_rows="dynamic", width="stretch",
                                height=280, key="outcomes_editor")
        if st.button("Save outcome corrections"):
            appkit.set_data(replace(appkit.get_data(), recommendation_outcomes=edited))
            st.success("Saved. Re-open the tab to see updated metrics.")

# ---------------------------------------------------------------------------
# Tab 2 — Claim Provenance
# ---------------------------------------------------------------------------
with tab_claim:
    st.subheader("Claim provenance")
    st.caption("Typed claims about the brand, connected to response, prompt, platform, citations and outcome.")
    prov = CL.claims_with_provenance(data.brand_claims, data.response_runs, data.citations, data.recommendation_outcomes)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Most common claims for {focal}**")
        st.dataframe(CL.claim_frequency(data.brand_claims, focal), width="stretch", hide_index=True)
        st.markdown("**Claims in recommended vs rejected responses**")
        rec_c = CL.claims_by_outcome(prov, focal, DL.OUT_RECOMMENDED).rename(columns={"claims": "in recommended"})
        rej_c = CL.claims_by_outcome(prov, focal, DL.OUT_REJECTED).rename(columns={"claims": "in rejected"})
        merged = rec_c.merge(rej_c, on="claim_type", how="outer").fillna(0)
        st.dataframe(merged, width="stretch", hide_index=True)
    with c2:
        st.markdown("**Citation support (claims appearing alongside a citation)**")
        st.caption("Association only — a missing citation does not prove a claim is unsupported.")
        st.dataframe(CL.claims_citation_support(prov, focal), width="stretch", hide_index=True)
        st.markdown("**Conflicting claims across responses**")
        conf = CL.conflicting_claims(data.brand_claims, focal)
        st.dataframe(conf if not conf.empty else pd.DataFrame([{"note": "none detected"}]), width="stretch", hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Claims by platform** (differ by platform)")
        st.dataframe(CL.claims_by_platform(data.brand_claims, data.response_runs, focal), width="stretch", hide_index=True, height=240)
    with c4:
        st.markdown("**Claims over time** (change across collection dates)")
        st.dataframe(CL.claims_over_time(data.brand_claims, data.response_runs, focal), width="stretch", hide_index=True, height=240)

    with st.expander("Provenance relationship table (brand · claim · run · platform · outcome · citations)"):
        cols = [c for c in ["brand_name", "claim_type", "claim_text", "run_id", "platform", "outcome", "has_citation", "citation_domains"] if c in prov.columns]
        st.dataframe(prov[prov["brand_name"] == focal][cols], width="stretch", hide_index=True, height=300)

# ---------------------------------------------------------------------------
# Tab 3 — Customer Journey
# ---------------------------------------------------------------------------
with tab_journey:
    st.subheader("Customer decision journey")
    journey = JN.resolve_journey(data)
    jk = journey["journey_kind"].iloc[0] if not journey.empty else "n/a"
    if "Simulated" in str(jk):
        st.info("**Simulated journey** — built from independent prompts mapped to decision stages. "
                "These separate AI responses are **not** one real person's conversation.", icon="🧭")
    else:
        st.info("**Linked conversation journey** — from user-provided linked multi-turn data.", icon="🔗")

    funnel = JN.journey_funnel(data, focal, journey)
    if funnel.empty:
        st.warning("No journey stages could be derived. Ensure prompts have journey_stage values.")
    else:
        head = JN.journey_headline_metrics(funnel)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Discovery inclusion", f"{round((head['discovery_inclusion'] or 0)*100)}%")
        k2.metric("Consideration survival", f"{round((head['consideration_survival'] or 0)*100)}%")
        k3.metric("Decision recommendation", f"{round((head['decision_recommendation'] or 0)*100)}%")
        k4.metric("Stage most often lost", head["stage_most_lost"] or "—")

        fig = px.funnel(funnel, x="mention_rate", y="stage",
                        labels={"mention_rate": "Focal brand inclusion (survival)", "stage": ""})
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")

        show = funnel.copy()
        for c in ["mention_rate", "recommendation_rate", "rejection_rate"]:
            show[c] = (show[c] * 100).round(0).astype(int).astype(str) + "%"
        show.columns = ["Stage", "Order", "Prompts", "Responses", "Inclusion", "Recommendation", "Rejection"]
        st.dataframe(show, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Competitor gaining most between stages**")
            st.dataframe(JN.competitor_gain_between_stages(data, focal, journey), width="stretch", hide_index=True)
            st.markdown("**Citation source by stage**")
            st.dataframe(JN.citation_sources_by_stage(data, journey), width="stretch", hide_index=True)
        with c2:
            st.markdown("**Objections introduced by stage**")
            st.dataframe(JN.objections_by_stage(data, focal, journey), width="stretch", hide_index=True, height=280)

# ---------------------------------------------------------------------------
# Tab 4 — Truth & Freshness
# ---------------------------------------------------------------------------
with tab_truth:
    st.subheader("Truth & freshness monitor")
    st.caption("**Authoritative-source comparison** of AI claims vs the facts you enter — not a verification of absolute truth.")

    st.markdown("**Authoritative brand facts** (enter or edit)")
    facts_seed = data.brand_facts if not data.brand_facts.empty else pd.DataFrame(
        [{"brand_name": focal, "fact_type": FACT_TYPES[0], "fact_value": "", "source_url": "", "as_of_date": ""}]
    )
    edited_facts = st.data_editor(
        facts_seed, num_rows="dynamic", width="stretch", key="facts_editor",
        column_config={"fact_type": st.column_config.SelectboxColumn("fact_type", options=FACT_TYPES)},
    )
    if st.button("Save & compare facts"):
        appkit.set_data(replace(appkit.get_data(), brand_facts=edited_facts.reindex(columns=BRAND_FACTS_COLUMNS)))
        st.success("Facts saved.")
        data = appkit.get_data()

    comp = TM.compare_facts(data.brand_facts, data.brand_claims, data.response_runs, data.citations)
    if comp.empty:
        st.info("Add authoritative facts above to run the comparison.")
    else:
        summary = TM.truth_summary(comp)
        st.dataframe(summary.rename(columns={"verdict": "Verdict", "count": "Count", "share": "Share"}),
                     width="stretch", hide_index=True)
        st.dataframe(comp.rename(columns={
            "brand_name": "Brand", "fact_type": "Fact type", "fact_value": "Official fact",
            "verdict": "Verdict", "ai_claim": "AI claim", "ai_response_text": "AI response text",
            "ai_citation": "AI citation", "source_url": "Official source", "as_of_date": "Collected",
            "business_risk": "Risk", "recommended_action": "Recommended action"}),
            width="stretch", hide_index=True, height=320)

# ---------------------------------------------------------------------------
# Tab 5 — Evidence Opportunities
# ---------------------------------------------------------------------------
with tab_evi:
    st.subheader("Evidence opportunities")
    st.caption("Actions grounded in observed objections behind rejections — not just missing topics.")
    eo = EV.evidence_opportunities(data, focal)
    if eo.empty:
        st.success(f"No rejection objections for {focal} in this view — no evidence gaps to act on.")
    else:
        st.dataframe(eo.rename(columns={
            "observed_gap": "Observed gap", "cluster": "Cluster", "objection": "Objection",
            "occurrences": "Occurrences", "competitor_evidence": "Competitor evidence",
            "citation_evidence": "Citation evidence", "recommended_asset": "Recommended asset",
            "confidence": "Confidence", "limitation": "Limitation"}),
            width="stretch", hide_index=True, height=320)

    st.markdown("**Claims that may need citable grounding**")
    prov = CL.claims_with_provenance(data.brand_claims, data.response_runs, data.citations, data.recommendation_outcomes)
    grounding = EV.claim_grounding_opportunities(prov, focal)
    st.dataframe(grounding if not grounding.empty else pd.DataFrame([{"note": "claims generally appear alongside citations"}]),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Tab 6 — Priority Framework
# ---------------------------------------------------------------------------
with tab_prio:
    st.subheader("Decision impact prioritization")
    st.caption("Every component, weight, raw value and weighted contribution is shown — no unexplained score.")
    st.code(PR.FORMULA)

    st.markdown("**Editable component weights**")
    weight_cols = st.columns(4)
    weights = {}
    for i, (comp, default) in enumerate(PR.DEFAULT_WEIGHTS.items()):
        with weight_cols[i % 4]:
            weights[comp] = st.number_input(comp, min_value=0.0, max_value=5.0, value=float(default), step=0.5, key=f"w_{comp}")

    pt = PR.priority_table(data, focal, weights=weights)
    if pt.empty:
        st.info("Not enough data to prioritize.")
    else:
        disp = pt[["question_cluster", "runs", "priority", "explanation"]].rename(
            columns={"question_cluster": "Cluster", "runs": "Responses", "priority": "Priority", "explanation": "Explanation"})
        st.dataframe(disp, width="stretch", hide_index=True)
        fig = px.bar(pt.head(10), x="priority", y="question_cluster", orientation="h",
                     labels={"priority": "Priority (0-100)", "question_cluster": ""})
        fig.update_traces(marker_color="#2563eb")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=340)
        st.plotly_chart(fig, width="stretch")

        chosen = st.selectbox("Show full breakdown for cluster", pt["question_cluster"].tolist())
        row = pt[pt["question_cluster"] == chosen].iloc[0]
        st.dataframe(PR.priority_breakdown(row, weights), width="stretch", hide_index=True)
        st.caption("Definitions: " + " · ".join(f"**{k}**: {v}" for k, v in list(PR.COMPONENT_DEFINITIONS.items())[:4]))
        st.caption("Limitations: components are directional; small samples reduce confidence (see sample_confidence).")

# ---------------------------------------------------------------------------
# Case study export
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📄 Decision-influence case study export")
rq = st.text_input("Research question", value=f"Where does {focal} win or lose in AI-driven decisions, and why?")
md = CS.build_case_study_markdown(data, focal, rq)
st.download_button("⬇️ Download case study (Markdown)", data=md,
                   file_name=f"decision_influence_case_study_{focal.replace('.', '_')}.md",
                   mime="text/markdown", width="stretch")
with st.expander("Preview case study"):
    st.code(md, language="markdown")
