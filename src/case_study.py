"""Decision-influence case study export (AI Decision Influence Lab).

Assembles a Markdown research write-up from the lab's grounded outputs. Synthetic
datasets are labelled as synthetic demonstrations; real datasets are labelled as such.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import claims as CL
from . import decision_lab as DL
from . import evidence_engine as EV
from . import journeys as JN
from . import prioritization as PR
from . import truth_monitor as TM
from .database import AnalysisData


def _dataset_label(data: AnalysisData) -> str:
    kinds = sorted(data.response_runs["dataset_kind"].dropna().unique().tolist()) if "dataset_kind" in data.response_runs.columns and not data.response_runs.empty else []
    if kinds == ["Synthetic"]:
        return ("**Synthetic demonstration** — generated demo data, NOT real AI platform output and "
                "NOT evidence of real brand performance.")
    if "Synthetic" in kinds and len(kinds) > 1:
        return "**Mixed dataset (contains synthetic data)** — filter to a single dataset type before quoting results."
    if kinds:
        return f"**{'/'.join(kinds)} data** — collected by the user; associations only, not proof of causation."
    return "**Unlabelled dataset.**"


def build_case_study_markdown(data: AnalysisData, focal_brand: str, research_question: str = "") -> str:
    """Build the 12-section decision-influence case study as Markdown."""
    lines: list[str] = [f"# AI Decision Influence Case Study — {focal_brand}", ""]
    lines.append(f"> Dataset: {_dataset_label(data)}")
    lines.append("> Findings are associations, not proof of causation. This is an authoritative-source")
    lines.append("> comparison, not a verification of absolute truth.")
    lines.append("")

    # 1. Research question
    lines += ["## 1. Research question", research_question or "_(not specified)_", ""]

    # 2. Brands
    brands = ", ".join(data.brands["brand_name"].tolist()) if not data.brands.empty else "—"
    lines += ["## 2. Brands", f"Focal: **{focal_brand}**. Tracked: {brands}.", ""]

    # 3. Platforms
    plats = ", ".join(sorted(data.response_runs["platform"].dropna().unique())) if not data.response_runs.empty else "—"
    lines += ["## 3. Platforms", plats, ""]

    # 4. Prompt methodology
    n_prompts = data.prompts["prompt_id"].nunique() if not data.prompts.empty else 0
    n_runs = len(data.response_runs)
    lines += ["## 4. Prompt methodology",
              f"{n_prompts} prompts, {n_runs} responses. Prompts are classified by category, topic, "
              "persona, journey stage and question cluster (existing structured metadata; not keyword guessing).", ""]

    # 5. Journey design
    funnel = JN.journey_funnel(data, focal_brand)
    jk = JN.resolve_journey(data)["journey_kind"].iloc[0] if not JN.resolve_journey(data).empty else "n/a"
    lines += ["## 5. Journey design", f"Journey type: {jk}. Stages measured: "
              + (", ".join(funnel["stage"].tolist()) if not funnel.empty else "none") + ".", ""]

    # 6. Recommendation outcomes
    summ = DL.outcome_summary(data.recommendation_outcomes)
    lines += ["## 6. Recommendation outcomes"]
    if not summ.empty and focal_brand in set(summ["brand_name"]):
        r = summ[summ["brand_name"] == focal_brand].iloc[0]
        lines.append(f"- {focal_brand}: {int(r['recommended'])} recommended, {int(r['rejected'])} rejected of "
                     f"{int(r['mentioned'])} mentions — mention→recommendation {round(r['mention_to_recommendation_rate']*100)}%, "
                     f"rejection rate {round(r['rejection_rate']*100)}%.")
    else:
        lines.append("- No outcomes computed.")
    lines.append("")

    # 7. Claim and citation findings
    freq = CL.claim_frequency(data.brand_claims, focal_brand)
    lines += ["## 7. Claim and citation findings"]
    if not freq.empty:
        top = ", ".join(f"{r['claim_type']} ({int(r['claims'])})" for _, r in freq.head(3).iterrows())
        lines.append(f"- Most common claims for {focal_brand}: {top}.")
    prov = CL.claims_with_provenance(data.brand_claims, data.response_runs, data.citations, data.recommendation_outcomes)
    support = CL.claims_citation_support(prov, focal_brand)
    if not support.empty:
        weakest = support.iloc[0]
        lines.append(f"- Weakest citation support: {weakest['claim_type']} ({round(weakest['supported_share']*100)}% "
                     "of instances appeared alongside a citation — association only).")
    lines.append("")

    # 8. Truth / freshness risks
    comp = TM.compare_facts(data.brand_facts, data.brand_claims, data.response_runs, data.citations)
    lines += ["## 8. Truth or freshness risks (authoritative-source comparison)"]
    if not comp.empty:
        risky = comp[comp["business_risk"].isin(["High", "Medium"])]
        focal_risky = risky[risky["brand_name"] == focal_brand]
        use = focal_risky if not focal_risky.empty else risky
        for _, row in use.head(4).iterrows():
            lines.append(f"- {row['brand_name']} · {row['fact_type']}: **{row['verdict']}** ({row['business_risk']} risk) — {row['recommended_action']}")
    else:
        lines.append("- No authoritative facts entered, so no comparison was run.")
    lines.append("")

    # 9. Evidence opportunities
    eo = EV.evidence_opportunities(data, focal_brand)
    lines += ["## 9. Evidence opportunities"]
    for _, row in eo.head(5).iterrows():
        lines.append(f"- **{row['recommended_asset']}** for '{row['objection']}' in '{row['cluster']}' "
                     f"({int(row['occurrences'])}×, {row['confidence']}).")
    if eo.empty:
        lines.append("- No rejection objections in the data, so no evidence actions were generated.")
    lines.append("")

    # 10. Priority recommendations
    pt = PR.priority_table(data, focal_brand)
    lines += ["## 10. Priority recommendations"]
    for _, row in pt.head(5).iterrows():
        lines.append(f"- {row['question_cluster']} — priority {row['priority']}/100. {row['explanation']}")
    if pt.empty:
        lines.append("- Not enough data to prioritize.")
    lines.append("")

    # 11. Limitations
    lines += ["## 11. Limitations",
              "- Deterministic keyword extraction can miss or mis-label claims and objections; every "
              "classification is editable.",
              "- Outcomes/claims are associations. A citation appearing alongside a recommendation does NOT prove it caused it.",
              "- The truth monitor is an authoritative-source comparison, not a verification of absolute truth.",
              "- Simulated journeys combine independent responses; they are not one real person's conversation.",
              "- Small samples are fragile — see per-metric sample sizes.", ""]

    # 12. Next experiment
    lines += ["## 12. Next experiment",
              f"Ship the top-priority evidence asset for {focal_brand}, then re-collect the same prompts as a "
              "labelled benchmark and compare with the AEO Experiments page. Use a holdout of untouched "
              "questions and repeated runs to move from association toward a stronger claim.", ""]

    return "\n".join(lines)
