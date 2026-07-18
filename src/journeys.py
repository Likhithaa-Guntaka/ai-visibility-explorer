"""Multi-turn customer decision journeys (AI Decision Influence Lab).

Groups prompts into an ordered decision journey (Discovery → Consideration → Evaluation
→ Decision → Retention) and measures where the focal brand is included, survives, is
recommended, or drops out.

By default the journey is *derived from existing prompt metadata* (each prompt's
``journey_stage`` mapped to the ordered decision stages) and is labelled a **simulated**
journey of independent prompts. A user-supplied ``journeys`` table can instead describe
an **actual linked conversation**. The two are always labelled separately, because
separate AI responses are not one real person's conversation.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .database import (
    DECISION_JOURNEY_STAGES,
    JOURNEYS_COLUMNS,
    JOURNEY_STAGE_MAP,
    AnalysisData,
)
from .decision_lab import OUT_NOT_MENTIONED, OUT_RECOMMENDED, OUT_REJECTED
from .entities import _split
from . import metrics as M

_STAGE_ORDER = {s: i + 1 for i, s in enumerate(DECISION_JOURNEY_STAGES)}


def derive_default_journey(prompts: pd.DataFrame) -> pd.DataFrame:
    """Build a *simulated* journey by mapping each prompt's ``journey_stage`` to a
    decision stage. Returns a DataFrame matching ``JOURNEYS_COLUMNS``."""
    if prompts.empty or "journey_stage" not in prompts.columns:
        return pd.DataFrame(columns=JOURNEYS_COLUMNS)
    rows = []
    for _, p in prompts.iterrows():
        stage = JOURNEY_STAGE_MAP.get(str(p.get("journey_stage", "")).strip())
        if stage is None:
            continue
        rows.append({
            "journey_id": "default",
            "journey_name": "Default journey (from prompt metadata)",
            "journey_kind": "Simulated (independent prompts)",
            "stage": stage,
            "stage_order": _STAGE_ORDER[stage],
            "prompt_id": p["prompt_id"],
        })
    return pd.DataFrame(rows, columns=JOURNEYS_COLUMNS)


def resolve_journey(data: AnalysisData) -> pd.DataFrame:
    """Return the active journey: the user-defined ``journeys`` table if present,
    otherwise the derived default. Always includes ``journey_kind``."""
    if not data.journeys.empty:
        j = data.journeys.copy()
        if "stage_order" not in j.columns or j["stage_order"].isna().any():
            j["stage_order"] = j["stage"].map(_STAGE_ORDER).fillna(99).astype(int)
        return j
    return derive_default_journey(data.prompts)


def _stage_runs(data: AnalysisData, journey: pd.DataFrame) -> pd.DataFrame:
    """Return response_runs annotated with their journey stage/order."""
    if journey.empty or data.response_runs.empty:
        return pd.DataFrame(columns=["run_id", "prompt_id", "stage", "stage_order"])
    pmap = journey[["prompt_id", "stage", "stage_order"]].drop_duplicates("prompt_id")
    return data.response_runs.merge(pmap, on="prompt_id", how="inner")


def journey_funnel(data: AnalysisData, focal_brand: str, journey: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-stage funnel for the focal brand.

    Columns: ``stage``, ``stage_order``, ``prompts``, ``runs``, ``mention_rate``
    (inclusion / survival), ``recommendation_rate``, ``rejection_rate``. Uses the
    recommendation_outcomes table for recommend/reject counts.
    """
    cols = ["stage", "stage_order", "prompts", "runs", "mention_rate", "recommendation_rate", "rejection_rate"]
    journey = resolve_journey(data) if journey is None else journey
    runs_ctx = _stage_runs(data, journey)
    if runs_ctx.empty:
        return pd.DataFrame(columns=cols)
    outcomes = data.recommendation_outcomes
    rows = []
    for (stage, order), grp in runs_ctx.groupby(["stage", "stage_order"]):
        run_ids = set(grp["run_id"])
        n = len(run_ids)
        n_prompts = grp["prompt_id"].nunique()
        if outcomes.empty:
            mention_rate = rec_rate = rej_rate = 0.0
        else:
            sub = outcomes[(outcomes["run_id"].isin(run_ids)) & (outcomes["brand_name"] == focal_brand)]
            mentioned = int((sub["outcome"] != OUT_NOT_MENTIONED).sum())
            rec = int((sub["outcome"] == OUT_RECOMMENDED).sum())
            rej = int((sub["outcome"] == OUT_REJECTED).sum())
            mention_rate = mentioned / n if n else 0.0
            rec_rate = rec / n if n else 0.0
            rej_rate = rej / n if n else 0.0
        rows.append({"stage": stage, "stage_order": int(order), "prompts": int(n_prompts), "runs": int(n),
                     "mention_rate": mention_rate, "recommendation_rate": rec_rate, "rejection_rate": rej_rate})
    return pd.DataFrame(rows, columns=cols).sort_values("stage_order").reset_index(drop=True)


def journey_headline_metrics(funnel: pd.DataFrame) -> dict:
    """Derive the headline journey metrics from a funnel.

    Returns discovery_inclusion, consideration_survival, evaluation_survival,
    decision_recommendation, full_journey_survival (min mention rate across populated
    stages), and stage_most_lost (lowest mention rate).
    """
    empty = {"discovery_inclusion": None, "consideration_survival": None, "evaluation_survival": None,
             "decision_recommendation": None, "full_journey_survival": None, "stage_most_lost": None,
             "journey_kind": None}
    if funnel.empty:
        return empty
    by_stage = {r["stage"]: r for _, r in funnel.iterrows()}

    def mr(stage):
        return float(by_stage[stage]["mention_rate"]) if stage in by_stage else None

    def rr(stage):
        return float(by_stage[stage]["recommendation_rate"]) if stage in by_stage else None

    mention_rates = funnel["mention_rate"].tolist()
    lost = funnel.loc[funnel["mention_rate"].idxmin(), "stage"] if not funnel.empty else None
    return {
        "discovery_inclusion": mr("Discovery"),
        "consideration_survival": mr("Consideration"),
        "evaluation_survival": mr("Evaluation"),
        "decision_recommendation": rr("Decision"),
        "full_journey_survival": float(min(mention_rates)) if mention_rates else None,
        "stage_most_lost": lost,
    }


def competitor_gain_between_stages(data: AnalysisData, focal_brand: str, journey: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Competitor with the largest share-of-voice increase between consecutive stages.

    Columns: ``from_stage``, ``to_stage``, ``competitor``, ``sov_change``.
    """
    cols = ["from_stage", "to_stage", "competitor", "sov_change"]
    journey = resolve_journey(data) if journey is None else journey
    runs_ctx = _stage_runs(data, journey)
    if runs_ctx.empty or data.brand_mentions.empty:
        return pd.DataFrame(columns=cols)
    # SoV per stage per brand.
    sov_by_stage: dict[str, dict[str, float]] = {}
    order = {}
    for (stage, so), grp in runs_ctx.groupby(["stage", "stage_order"]):
        run_ids = set(grp["run_id"])
        m = data.brand_mentions[data.brand_mentions["run_id"].isin(run_ids)]
        sov = M.share_of_voice(m)
        sov_by_stage[stage] = dict(zip(sov["brand_name"], sov["share_of_voice"])) if not sov.empty else {}
        order[stage] = so
    stages = sorted(sov_by_stage, key=lambda s: order[s])
    rows = []
    for a, b in zip(stages, stages[1:]):
        gains = {}
        for brand in set(sov_by_stage[a]) | set(sov_by_stage[b]):
            if brand == focal_brand:
                continue
            gains[brand] = sov_by_stage[b].get(brand, 0.0) - sov_by_stage[a].get(brand, 0.0)
        if gains:
            top = max(gains, key=gains.get)
            rows.append({"from_stage": a, "to_stage": b, "competitor": top, "sov_change": gains[top]})
    return pd.DataFrame(rows, columns=cols)


def objections_by_stage(data: AnalysisData, focal_brand: str, journey: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rejection reasons for the focal brand per journey stage (objections at each stage).

    Columns: ``stage``, ``stage_order``, ``reason``, ``count``.
    """
    cols = ["stage", "stage_order", "reason", "count"]
    journey = resolve_journey(data) if journey is None else journey
    runs_ctx = _stage_runs(data, journey)
    if runs_ctx.empty or data.recommendation_outcomes.empty:
        return pd.DataFrame(columns=cols)
    outcomes = data.recommendation_outcomes.merge(runs_ctx[["run_id", "stage", "stage_order"]], on="run_id", how="inner")
    sub = outcomes[(outcomes["brand_name"] == focal_brand) & (outcomes["outcome"] == OUT_REJECTED)]
    counts: dict[tuple, int] = {}
    for _, row in sub.iterrows():
        for reason in _split(row["reason_categories"]):
            key = (row["stage"], int(row["stage_order"]), reason)
            counts[key] = counts.get(key, 0) + 1
    rows = [{"stage": k[0], "stage_order": k[1], "reason": k[2], "count": v} for k, v in counts.items()]
    return pd.DataFrame(rows, columns=cols).sort_values(["stage_order", "count"], ascending=[True, False]).reset_index(drop=True)


def citation_sources_by_stage(data: AnalysisData, journey: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Top cited domain per journey stage (citation source changes between stages).

    Columns: ``stage``, ``stage_order``, ``top_domain``, ``citations``.
    """
    cols = ["stage", "stage_order", "top_domain", "citations"]
    journey = resolve_journey(data) if journey is None else journey
    runs_ctx = _stage_runs(data, journey)
    if runs_ctx.empty or data.citations.empty:
        return pd.DataFrame(columns=cols)
    cit = data.citations.merge(runs_ctx[["run_id", "stage", "stage_order"]], on="run_id", how="inner")
    rows = []
    for (stage, so), grp in cit.groupby(["stage", "stage_order"]):
        vc = grp["citation_domain"].value_counts()
        if vc.empty:
            continue
        rows.append({"stage": stage, "stage_order": int(so), "top_domain": vc.index[0], "citations": int(vc.iloc[0])})
    return pd.DataFrame(rows, columns=cols).sort_values("stage_order").reset_index(drop=True)
