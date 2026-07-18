"""Decision Impact Prioritization (AI Decision Influence Lab).

Ranks opportunities (question clusters) with a fully transparent, user-weighted score.
Every component, its raw value, its weight, its weighted contribution, and the final
score are exposed — there is never an unexplained number.

priority = ( Σ weight_i × component_i ) / ( Σ weight_i ) × 100
where every component_i is normalised to 0..1 and higher = more worth investing in.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import clusters as C
from . import decision_lab as DL
from .database import AnalysisData

FORMULA = "priority = ( Σ weightᵢ × componentᵢ ) / ( Σ weightᵢ ) × 100   (each componentᵢ ∈ [0,1])"

# Default weights (editable in the UI). Keys are the component names.
DEFAULT_WEIGHTS: dict[str, float] = {
    "visibility_gap": 1.0,
    "purchase_intent": 1.0,
    "mention_to_rec_gap": 1.5,
    "journey_stage_importance": 1.0,
    "persona_importance": 0.5,
    "evidence_deficit": 1.0,
    "competitor_advantage": 1.0,
    "sample_confidence": 0.5,
}

COMPONENT_DEFINITIONS: dict[str, str] = {
    "visibility_gap": "1 − focal brand mention rate in the cluster (how invisible the brand is).",
    "purchase_intent": "Share of the cluster's responses from Transactional/Decision prompts.",
    "mention_to_rec_gap": "Focal mention rate − recommendation rate (named but not picked).",
    "journey_stage_importance": "Importance of the cluster's dominant decision stage (Decision highest).",
    "persona_importance": "Configured importance of the cluster's dominant persona (default 0.5 for all).",
    "evidence_deficit": "Rejection rate for the focal brand in the cluster (unmet objections).",
    "competitor_advantage": "Best competitor recommendation rate − focal recommendation rate.",
    "sample_confidence": "min(1, responses / 10) — down-weights tiny samples.",
}

# Stage importance weights (0..1) for the dominant decision stage of a cluster.
_STAGE_IMPORTANCE = {"Discovery": 0.4, "Consideration": 0.6, "Evaluation": 0.8, "Decision": 1.0, "Retention": 0.5}
_INTENT_PURCHASE = {"Transactional", "Commercial investigation"}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def priority_table(
    data: AnalysisData,
    focal_brand: str,
    weights: Optional[dict[str, float]] = None,
    persona_importance: Optional[dict[str, float]] = None,
    dimension: str = "question_cluster",
) -> pd.DataFrame:
    """Per-cluster transparent priority table for the focal brand.

    Returns one row per cluster with every raw component value, the weighted
    contribution of each, the final ``priority`` (0..100), and a short ``explanation``.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    persona_importance = persona_importance or {}
    prompts = C.prepare_prompts(data.prompts)
    outcomes = data.recommendation_outcomes
    comp_cols = list(DEFAULT_WEIGHTS.keys())
    out_cols = ([dimension, "runs"] + comp_cols
                + [f"wc_{c}" for c in comp_cols] + ["priority", "explanation"])
    if prompts.empty or data.response_runs.empty or dimension not in prompts.columns:
        return pd.DataFrame(columns=out_cols)

    summary = C.cluster_summary(data, focal_brand, dimension)
    if summary.empty:
        return pd.DataFrame(columns=out_cols)

    weight_sum = sum(weights.get(c, 0.0) for c in comp_cols) or 1.0
    rows = []
    for _, s in summary.iterrows():
        cluster_value = s["cluster"]
        n = int(s["runs"])
        cluster_prompts = prompts[prompts[dimension] == cluster_value]
        cluster_pids = set(cluster_prompts["prompt_id"])
        cluster_runs = set(data.response_runs[data.response_runs["prompt_id"].isin(cluster_pids)]["run_id"])

        # Components (all 0..1).
        visibility_gap = _clamp(1.0 - float(s["focal_mention_rate"]))
        # purchase intent = share of responses whose prompt intent is purchase-like.
        if "search_intent" in cluster_prompts.columns and not cluster_prompts.empty:
            runs_intent = data.response_runs[data.response_runs["prompt_id"].isin(cluster_pids)].merge(
                cluster_prompts[["prompt_id", "search_intent"]], on="prompt_id", how="left")
            purchase_intent = _clamp(runs_intent["search_intent"].isin(_INTENT_PURCHASE).mean()) if len(runs_intent) else 0.0
        else:
            purchase_intent = 0.0
        # mention→rec gap.
        if not outcomes.empty:
            sub = outcomes[(outcomes["run_id"].isin(cluster_runs)) & (outcomes["brand_name"] == focal_brand)]
            mentioned = int((sub["outcome"] != DL.OUT_NOT_MENTIONED).sum())
            rec = int((sub["outcome"] == DL.OUT_RECOMMENDED).sum())
            rej = int((sub["outcome"] == DL.OUT_REJECTED).sum())
            mention_rate = mentioned / n if n else 0.0
            rec_rate = rec / n if n else 0.0
            mention_to_rec_gap = _clamp(mention_rate - rec_rate)
            evidence_deficit = _clamp(rej / mentioned) if mentioned else 0.0
        else:
            mention_to_rec_gap = evidence_deficit = 0.0
        # dominant stage importance.
        from .database import JOURNEY_STAGE_MAP
        stages = cluster_prompts["journey_stage"].map(lambda x: JOURNEY_STAGE_MAP.get(str(x).strip(), "Consideration")) if "journey_stage" in cluster_prompts.columns else pd.Series(["Consideration"])
        dom_stage = stages.mode().iloc[0] if not stages.empty else "Consideration"
        journey_stage_importance = _STAGE_IMPORTANCE.get(dom_stage, 0.6)
        # dominant persona importance.
        dom_persona = cluster_prompts["persona"].mode().iloc[0] if ("persona" in cluster_prompts.columns and not cluster_prompts["persona"].dropna().empty) else ""
        persona_imp = _clamp(persona_importance.get(dom_persona, 0.5))
        # competitor advantage (rec rate best competitor − focal).
        competitor_advantage = _clamp(float(s["competitor_mention_rate"]) - float(s["focal_mention_rate"]))
        # sample confidence.
        sample_confidence = _clamp(n / 10.0)

        components = {
            "visibility_gap": visibility_gap, "purchase_intent": purchase_intent,
            "mention_to_rec_gap": mention_to_rec_gap, "journey_stage_importance": journey_stage_importance,
            "persona_importance": persona_imp, "evidence_deficit": evidence_deficit,
            "competitor_advantage": competitor_advantage, "sample_confidence": sample_confidence,
        }
        weighted = {f"wc_{c}": weights.get(c, 0.0) * components[c] for c in comp_cols}
        priority = sum(weighted.values()) / weight_sum * 100.0

        top_driver = max(comp_cols, key=lambda c: weighted[f"wc_{c}"])
        row = {dimension: cluster_value, "runs": n, **components, **weighted, "priority": round(priority, 1),
               "explanation": f"Top driver: {top_driver.replace('_',' ')} ({round(weighted['wc_'+top_driver],2)} of {round(priority,1)})."}
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols).sort_values("priority", ascending=False).reset_index(drop=True)


def priority_breakdown(row: pd.Series, weights: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """Expand one priority row into a per-component table for full transparency.

    Columns: ``component``, ``definition``, ``raw_value``, ``weight``,
    ``weighted_contribution``.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    rows = []
    for c in DEFAULT_WEIGHTS:
        rows.append({
            "component": c,
            "definition": COMPONENT_DEFINITIONS[c],
            "raw_value": round(float(row[c]), 3),
            "weight": weights.get(c, 0.0),
            "weighted_contribution": round(float(row.get(f"wc_{c}", 0.0)), 3),
        })
    return pd.DataFrame(rows)
