"""Evidence Opportunity Engine (AI Decision Influence Lab).

Turns *observed recommendation gaps* — specifically the objections behind rejections —
into concrete evidence/content actions. Recommendations are generated only for
objections that actually occur in the data; nothing is invented.

Each action ties an objection to a recommended asset, the cluster/stage where it
appears, competitor evidence, and citation evidence — all framed as associations.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import clusters as C
from . import decision_lab as DL
from .database import AnalysisData

# Objection (rejection reason) -> recommended evidence asset.
REASON_TO_ASSET: dict[str, str] = {
    "Pricing concern": "Pricing comparison",
    "Missing capability": "Product specification page",
    "Complexity": "Migration guide",
    "Ease of use concern": "Customer case study",
    "Integration concern": "Integration documentation",
    "Scalability concern": "Independent benchmark",
    "Persona mismatch": "Persona specific proof",
    "Trust or evidence concern": "Third party review coverage",
    "Competitor advantage": "Independent benchmark",
    "Other or unknown": "Limitation clarification",
}

# Claim types with weak citation support -> asset to add grounding.
_LOW_SUPPORT_ASSET = "Structured product data / updated authoritative page"

_LIMITATION = (
    "Association only: this objection appeared alongside rejections in the current sample. "
    "It does not prove the objection caused the rejection. Validate with more responses and repeated runs."
)


def _confidence(n: int) -> str:
    if n >= 5:
        return "Moderate (multiple responses)"
    if n >= 2:
        return "Low (few responses)"
    return "Very low (single response)"


def evidence_opportunities(data: AnalysisData, focal_brand: str, dimension: str = "question_cluster") -> pd.DataFrame:
    """Grounded evidence actions from the focal brand's rejection objections per cluster.

    Returns columns: ``observed_gap``, ``cluster``, ``objection``, ``occurrences``,
    ``competitor_evidence``, ``citation_evidence``, ``recommended_asset``,
    ``confidence``, ``limitation``. Only rows backed by data are produced.
    """
    cols = ["observed_gap", "cluster", "objection", "occurrences", "competitor_evidence",
            "citation_evidence", "recommended_asset", "confidence", "limitation"]
    outcomes = data.recommendation_outcomes
    if outcomes.empty:
        return pd.DataFrame(columns=cols)

    reasons = DL.rejection_reasons_by(outcomes, data.response_runs, data.prompts, dimension, focal_brand)
    if reasons.empty:
        return pd.DataFrame(columns=cols)

    prompts = C.prepare_prompts(data.prompts)
    rows = []
    for _, r in reasons.iterrows():
        cluster_value = r[dimension]
        reason = r["reason"]
        n = int(r["count"])

        # Competitor evidence: strongest competitor in the cluster.
        comp = C.competitor_rates_in_cluster(data, dimension, cluster_value)
        comp = comp[comp["brand_name"] != focal_brand] if not comp.empty else comp
        competitor_evidence = (
            f"{comp.iloc[0]['brand_name']} leads here ({round(comp.iloc[0]['share_of_voice']*100)}% SoV)"
            if not comp.empty else "No competitor lead measured"
        )

        # Citation evidence: which third-party domain appears in this cluster's responses.
        cluster_pids = set(prompts[prompts[dimension] == cluster_value]["prompt_id"]) if dimension in prompts.columns else set()
        cluster_runs = set(data.response_runs[data.response_runs["prompt_id"].isin(cluster_pids)]["run_id"])
        cit = data.citations[data.citations["run_id"].isin(cluster_runs)] if not data.citations.empty else data.citations
        if not cit.empty:
            top_domain = cit["citation_domain"].value_counts().index[0]
            citation_evidence = f"Cited most in this cluster: {top_domain}"
        else:
            citation_evidence = "No citations in this cluster"

        rows.append({
            "observed_gap": f"{focal_brand} rejected on '{reason}' in the '{cluster_value}' cluster",
            "cluster": cluster_value,
            "objection": reason,
            "occurrences": n,
            "competitor_evidence": competitor_evidence,
            "citation_evidence": citation_evidence,
            "recommended_asset": REASON_TO_ASSET.get(reason, "Limitation clarification"),
            "confidence": _confidence(n),
            "limitation": _LIMITATION,
        })
    return pd.DataFrame(rows, columns=cols).sort_values("occurrences", ascending=False).reset_index(drop=True)


def claim_grounding_opportunities(provenance: pd.DataFrame, focal_brand: str, threshold: float = 0.5) -> pd.DataFrame:
    """Claim types often stated WITHOUT a citation → add citable evidence.

    ``provenance`` is the output of :func:`src.claims.claims_with_provenance`.
    Returns: ``claim_type``, ``uncited_instances``, ``supported_share``,
    ``recommended_asset``, ``limitation``.
    """
    from .claims import claims_citation_support

    cols = ["claim_type", "uncited_instances", "supported_share", "recommended_asset", "limitation"]
    support = claims_citation_support(provenance, focal_brand)
    if support.empty:
        return pd.DataFrame(columns=cols)
    weak = support[support["supported_share"] < threshold]
    rows = []
    for _, s in weak.iterrows():
        rows.append({
            "claim_type": s["claim_type"],
            "uncited_instances": int(s["without_citation"]),
            "supported_share": float(s["supported_share"]),
            "recommended_asset": _LOW_SUPPORT_ASSET,
            "limitation": "Citation co-occurrence only; a missing citation does not prove the claim is unsupported.",
        })
    return pd.DataFrame(rows, columns=cols).sort_values("uncited_instances", ascending=False).reset_index(drop=True)
