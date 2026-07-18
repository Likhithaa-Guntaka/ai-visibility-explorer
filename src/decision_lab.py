"""Recommendation outcome classification + rejection reasons (AI Decision Influence Lab).

For every (response, tracked brand) pair this assigns a transparent, editable outcome —
recommended, mentioned-not-recommended, compared-but-rejected, or not-mentioned — and,
for rejections, extracts reason categories from deterministic keyword lexicons. Every
classification carries the exact text evidence the rule keyed on.

Nothing here claims a source *caused* an outcome; these are observed associations.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .database import (
    RECOMMENDATION_OUTCOMES_COLUMNS,
    REJECTION_REASONS,
)
from .entities import _brand_context, _split
from .extraction import _iter_term_matches, _unique_terms

# Outcome labels (kept local for clarity; also in database.OUTCOMES).
OUT_RECOMMENDED = "Mentioned and recommended"
OUT_NOT_RECOMMENDED = "Mentioned but not recommended"
OUT_REJECTED = "Compared but rejected"
OUT_NOT_MENTIONED = "Not mentioned"

# Lexicons mapping a rejection reason to the phrases that signal it (lowercase).
REJECTION_LEXICON: dict[str, list[str]] = {
    "Pricing concern": ["expensive", "pricey", "gets expensive", "costly", "overpriced", "can get pricey", "too costly", "price is a concern"],
    "Missing capability": ["lacks", "missing", "does not have", "doesn't have", "no built-in", "limited features", "cannot", "can't do"],
    "Complexity": ["complex", "complicated", "overwhelming", "steep learning curve", "hard to set up", "difficult to configure"],
    "Ease of use concern": ["not intuitive", "difficult to use", "confusing", "clunky", "cumbersome", "unintuitive"],
    "Integration concern": ["limited integrations", "does not integrate", "doesn't integrate", "poor integration", "few integrations"],
    "Scalability concern": ["does not scale", "doesn't scale", "struggles at scale", "not for large teams", "scalability", "hard to scale"],
    "Persona mismatch": ["not ideal for", "not suited", "overkill for", "better for small", "not for enterprise", "not a fit for"],
    "Trust or evidence concern": ["unproven", "lacks reviews", "little evidence", "not well known", "concerns about", "unclear track record"],
    "Competitor advantage": ["better alternative", "competitors offer", "outperformed by", "falls behind", "others do it better"],
    # "Other or unknown" is the fallback when a rejection is detected without a specific cue.
}

# Generic negative cues that mark a mention as a rejection even without a specific reason.
_GENERIC_NEGATIVE_CUES = ["but it", "however", "downside", "though it", "on the downside", "the catch", "drawback", "weakness"]

_EVIDENCE_MAX = 240


def _evidence_snippet(context: str) -> str:
    return (context or "").strip()[:_EVIDENCE_MAX]


def detect_rejection_reasons(context: str) -> tuple[list[str], str]:
    """Return (reason_categories, evidence) for a brand's context block.

    Matches the reason lexicons; if none match but a generic negative cue is present,
    returns ["Other or unknown"]. Returns ([], "") when no rejection signal is found.
    """
    context = (context or "").lower()
    if not context:
        return [], ""
    reasons: list[str] = []
    evidence_bits: list[str] = []
    for label, phrases in REJECTION_LEXICON.items():
        for phrase in phrases:
            if next(_iter_term_matches(context, phrase), None) is not None:
                reasons.append(label)
                evidence_bits.append(phrase)
                break
    if reasons:
        return reasons, _evidence_snippet(context)
    for cue in _GENERIC_NEGATIVE_CUES:
        if cue in context:
            return ["Other or unknown"], _evidence_snippet(context)
    return [], ""


def classify_outcomes(
    response_runs: pd.DataFrame,
    brand_mentions: pd.DataFrame,
    brand_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    """Classify the outcome for every (response, tracked brand) pair.

    Rules (deterministic, transparent):
    * not in brand_mentions for the run -> "Not mentioned"
    * mentioned and ``is_recommended`` -> "Mentioned and recommended"
    * mentioned, not recommended, and a rejection cue is present -> "Compared but rejected"
    * mentioned, not recommended, no rejection cue -> "Mentioned but not recommended"

    Returns a DataFrame matching ``RECOMMENDATION_OUTCOMES_COLUMNS``. Editable downstream.
    """
    if response_runs.empty or not brand_aliases:
        return pd.DataFrame(columns=RECOMMENDATION_OUTCOMES_COLUMNS)

    all_terms = {b: _unique_terms([b, *aliases]) for b, aliases in brand_aliases.items()}
    rows: list[dict] = []
    # Index mentions by run for speed.
    mentions_by_run: dict[str, pd.DataFrame] = {
        rid: grp for rid, grp in brand_mentions.groupby("run_id")
    } if not brand_mentions.empty else {}

    for _, run in response_runs.iterrows():
        run_id = str(run["run_id"])
        text = str(run.get("response_text", ""))
        run_mentions = mentions_by_run.get(run_id)
        mentioned = set(run_mentions["brand_name"]) if run_mentions is not None else set()
        recommended = (
            set(run_mentions[run_mentions["is_recommended"].astype(bool)]["brand_name"])
            if run_mentions is not None else set()
        )
        for brand, terms in all_terms.items():
            if brand not in mentioned:
                rows.append({"run_id": run_id, "brand_name": brand, "outcome": OUT_NOT_MENTIONED,
                             "reason_categories": "", "evidence_text": ""})
                continue
            context = _brand_context(text, terms)
            if brand in recommended:
                rows.append({"run_id": run_id, "brand_name": brand, "outcome": OUT_RECOMMENDED,
                             "reason_categories": "", "evidence_text": _evidence_snippet(context)})
                continue
            reasons, evidence = detect_rejection_reasons(context)
            if reasons:
                rows.append({"run_id": run_id, "brand_name": brand, "outcome": OUT_REJECTED,
                             "reason_categories": "; ".join(reasons), "evidence_text": evidence})
            else:
                rows.append({"run_id": run_id, "brand_name": brand, "outcome": OUT_NOT_RECOMMENDED,
                             "reason_categories": "", "evidence_text": _evidence_snippet(context)})
    return pd.DataFrame(rows, columns=RECOMMENDATION_OUTCOMES_COLUMNS)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def outcome_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per-brand outcome counts + conversion / survival / rejection rates.

    * mention_to_recommendation_rate = recommended / mentioned
    * comparison_survival_rate       = (mentioned - rejected) / mentioned
    * rejection_rate                 = rejected / mentioned
    where mentioned = recommended + not_recommended + rejected (i.e. not "Not mentioned").
    Returns one row per brand.
    """
    cols = ["brand_name", "recommended", "not_recommended", "rejected", "not_mentioned",
            "mentioned", "mention_to_recommendation_rate", "comparison_survival_rate", "rejection_rate"]
    if outcomes.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for brand, grp in outcomes.groupby("brand_name"):
        counts = grp["outcome"].value_counts().to_dict()
        rec = int(counts.get(OUT_RECOMMENDED, 0))
        notrec = int(counts.get(OUT_NOT_RECOMMENDED, 0))
        rej = int(counts.get(OUT_REJECTED, 0))
        notm = int(counts.get(OUT_NOT_MENTIONED, 0))
        mentioned = rec + notrec + rej
        rows.append({
            "brand_name": brand, "recommended": rec, "not_recommended": notrec,
            "rejected": rej, "not_mentioned": notm, "mentioned": mentioned,
            "mention_to_recommendation_rate": (rec / mentioned) if mentioned else 0.0,
            "comparison_survival_rate": ((mentioned - rej) / mentioned) if mentioned else 0.0,
            "rejection_rate": (rej / mentioned) if mentioned else 0.0,
        })
    return pd.DataFrame(rows, columns=cols).sort_values("mention_to_recommendation_rate", ascending=False).reset_index(drop=True)


def _enrich_outcomes(outcomes: pd.DataFrame, response_runs: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    """Attach platform + prompt attributes to outcome rows."""
    if outcomes.empty:
        return outcomes
    runs = response_runs[["run_id", "prompt_id", "platform"]] if not response_runs.empty else pd.DataFrame(columns=["run_id", "prompt_id", "platform"])
    keep = [c for c in ["prompt_id", "persona", "journey_stage", "question_cluster", "prompt_category"] if c in prompts.columns]
    prom = prompts[keep] if not prompts.empty and keep else pd.DataFrame(columns=["prompt_id"])
    return outcomes.merge(runs, on="run_id", how="left").merge(prom, on="prompt_id", how="left")


def conversion_by(
    outcomes: pd.DataFrame, response_runs: pd.DataFrame, prompts: pd.DataFrame,
    dimension: str, brand_name: str,
) -> pd.DataFrame:
    """Mention-to-recommendation conversion for one brand, split by a dimension.

    ``dimension`` is a column on the enriched outcomes (``platform``, ``persona``,
    ``journey_stage``, ``question_cluster``). Returns ``<dimension>``, ``recommended``,
    ``mentioned``, ``conversion_rate``.
    """
    out_cols = [dimension, "recommended", "mentioned", "conversion_rate"]
    enriched = _enrich_outcomes(outcomes, response_runs, prompts)
    if enriched.empty or dimension not in enriched.columns:
        return pd.DataFrame(columns=out_cols)
    sub = enriched[enriched["brand_name"] == brand_name]
    sub = sub[sub["outcome"] != OUT_NOT_MENTIONED]
    if sub.empty:
        return pd.DataFrame(columns=out_cols)
    rows = []
    for val, grp in sub.groupby(dimension):
        mentioned = len(grp)
        rec = int((grp["outcome"] == OUT_RECOMMENDED).sum())
        rows.append({dimension: val, "recommended": rec, "mentioned": mentioned,
                     "conversion_rate": (rec / mentioned) if mentioned else 0.0})
    return pd.DataFrame(rows, columns=out_cols).sort_values("conversion_rate", ascending=False).reset_index(drop=True)


def rejection_reasons_by(
    outcomes: pd.DataFrame, response_runs: pd.DataFrame, prompts: pd.DataFrame,
    dimension: str, brand_name: str,
) -> pd.DataFrame:
    """Counts of rejection reasons for one brand, split by a dimension (persona/stage).

    Returns long-form: ``<dimension>``, ``reason``, ``count``.
    """
    out_cols = [dimension, "reason", "count"]
    enriched = _enrich_outcomes(outcomes, response_runs, prompts)
    if enriched.empty or dimension not in enriched.columns:
        return pd.DataFrame(columns=out_cols)
    sub = enriched[(enriched["brand_name"] == brand_name) & (enriched["outcome"] == OUT_REJECTED)]
    if sub.empty:
        return pd.DataFrame(columns=out_cols)
    counts: dict[tuple, int] = {}
    for _, row in sub.iterrows():
        for reason in _split(row["reason_categories"]):
            key = (row[dimension], reason)
            counts[key] = counts.get(key, 0) + 1
    rows = [{dimension: k[0], "reason": k[1], "count": v} for k, v in counts.items()]
    return pd.DataFrame(rows, columns=out_cols).sort_values("count", ascending=False).reset_index(drop=True)


def rejection_reason_totals(outcomes: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    """Overall rejection-reason counts for one brand. Columns: ``reason``, ``count``."""
    cols = ["reason", "count"]
    if outcomes.empty:
        return pd.DataFrame(columns=cols)
    sub = outcomes[(outcomes["brand_name"] == brand_name) & (outcomes["outcome"] == OUT_REJECTED)]
    counts: dict[str, int] = {}
    for _, row in sub.iterrows():
        for reason in _split(row["reason_categories"]):
            counts[reason] = counts.get(reason, 0) + 1
    rows = [{"reason": r, "count": c} for r, c in counts.items()]
    return pd.DataFrame(rows, columns=cols).sort_values("count", ascending=False).reset_index(drop=True)
