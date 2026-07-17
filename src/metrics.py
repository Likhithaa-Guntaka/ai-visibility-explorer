"""Visibility metrics for AI Visibility Explorer.

Every metric is a small, pure function over the canonical DataFrames so it can be
unit-tested in isolation and reused by both the dashboard and the readout. Each
metric also has a plain-language definition in :data:`METRIC_DEFINITIONS`, which the
UI renders as tooltips / an info panel — the definitions are never hidden.

Vocabulary
----------
run
    One AI response (one row of ``response_runs``).
mentioned run
    A run in which at least one tracked brand was found.
focal brand
    The brand the customer cares about (chosen in the UI). Most per-brand tables are
    returned for *all* brands so the UI can filter.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Plain-language definitions surfaced as tooltips / an info section in the UI.
# ---------------------------------------------------------------------------
METRIC_DEFINITIONS: dict[str, str] = {
    "brand_mention_rate": (
        "Brand mention rate — the share of AI responses that mention a brand at "
        "least once. Example: mentioned in 12 of 20 responses = 60%."
    ),
    "share_of_voice": (
        "Share of voice — a brand's total mentions divided by the total mentions of "
        "all tracked brands. It shows how much of the conversation a brand owns."
    ),
    "first_mention_share": (
        "First mention share — among responses that mention any tracked brand, the "
        "share in which this brand is mentioned first. Being first often signals the "
        "model's default or top-of-mind answer."
    ),
    "recommendation_rate": (
        "Recommendation rate — the share of responses in which a brand is actively "
        "recommended (appears next to a recommendation cue such as 'recommend', "
        "'best', or heads the list). This is a transparent heuristic, not a guarantee."
    ),
    "citation_rate": (
        "Citation rate — the share of responses that include at least one source URL. "
        "Higher citation rates mean the platform is grounding answers in linkable pages."
    ),
    "source_domain_share": (
        "Source domain share — for each cited website, the share of all citations that "
        "point to it. Reveals which third-party sources influence AI answers most."
    ),
    "prompt_category_performance": (
        "Prompt category performance — brand mention rate broken down by prompt "
        "category (e.g. comparison vs. purchase intent). Shows where a brand is strong "
        "or weak by question type."
    ),
    "persona_performance": (
        "Persona performance — brand mention rate broken down by the customer persona a "
        "prompt targets. Shows which audiences a brand shows up for."
    ),
    "platform_comparison": (
        "Platform comparison — brand mention rate and share of voice by AI platform. "
        "The same brand can be far more visible on one platform than another."
    ),
    "competitor_visibility": (
        "Competitor visibility — a leaderboard of mention rate and share of voice "
        "across all tracked brands, so a customer can see where they rank."
    ),
    "content_coverage_gaps": (
        "Content coverage gaps — prompt topics where the focal brand is absent or weak "
        "while competitors appear. These are candidate areas for new or improved content."
    ),
    "response_consistency": (
        "Response consistency — how stable results are across repeated runs of the same "
        "prompt, measured with simple, explainable overlap metrics. Low consistency "
        "means a single run should not be over-interpreted."
    ),
}

# Below this many responses, a metric is statistically fragile; the UI warns.
SMALL_SAMPLE_THRESHOLD: int = 5


# ---------------------------------------------------------------------------
# Foundational helpers.
# ---------------------------------------------------------------------------


def total_runs(response_runs: pd.DataFrame) -> int:
    """Total number of AI responses in scope."""
    return int(len(response_runs))


def enrich_mentions(
    brand_mentions: pd.DataFrame,
    response_runs: pd.DataFrame,
    prompts: pd.DataFrame,
) -> pd.DataFrame:
    """Join mentions -> runs -> prompts so each mention row carries context columns.

    Adds ``platform``, ``run_date``, ``run_number``, ``prompt_id`` and the prompt's
    ``prompt_category``, ``topic``, ``persona``, ``journey_stage``. Returns an empty
    frame (with expected columns) if there are no mentions.
    """
    context_cols = [
        "run_id", "brand_name", "mention_count", "first_mention_position", "is_recommended",
        "prompt_id", "platform", "run_date", "run_number",
        "prompt_category", "topic", "persona", "journey_stage",
    ]
    if brand_mentions.empty:
        return pd.DataFrame(columns=context_cols)
    runs = response_runs[["run_id", "prompt_id", "platform", "run_date", "run_number"]]
    prom = prompts[["prompt_id", "prompt_category", "topic", "persona", "journey_stage"]]
    merged = brand_mentions.merge(runs, on="run_id", how="left").merge(prom, on="prompt_id", how="left")
    return merged.reindex(columns=context_cols)


# ---------------------------------------------------------------------------
# 1. Brand mention rate
# ---------------------------------------------------------------------------


def brand_mention_rate(brand_mentions: pd.DataFrame, response_runs: pd.DataFrame) -> pd.DataFrame:
    """Per-brand share of responses that mention the brand at least once.

    Returns columns: ``brand_name``, ``mentioned_runs``, ``total_runs``, ``mention_rate``.
    """
    n = total_runs(response_runs)
    if brand_mentions.empty or n == 0:
        return pd.DataFrame(columns=["brand_name", "mentioned_runs", "total_runs", "mention_rate"])
    mentioned = (
        brand_mentions.groupby("brand_name")["run_id"].nunique().rename("mentioned_runs").reset_index()
    )
    mentioned["total_runs"] = n
    mentioned["mention_rate"] = mentioned["mentioned_runs"] / n
    return mentioned.sort_values("mention_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Share of voice
# ---------------------------------------------------------------------------


def share_of_voice(brand_mentions: pd.DataFrame) -> pd.DataFrame:
    """Per-brand share of *total mentions* across all tracked brands.

    Returns columns: ``brand_name``, ``mentions``, ``share_of_voice``.
    """
    if brand_mentions.empty:
        return pd.DataFrame(columns=["brand_name", "mentions", "share_of_voice"])
    grouped = brand_mentions.groupby("brand_name")["mention_count"].sum().rename("mentions").reset_index()
    total = grouped["mentions"].sum()
    grouped["share_of_voice"] = grouped["mentions"] / total if total else 0.0
    return grouped.sort_values("share_of_voice", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. First mention share
# ---------------------------------------------------------------------------


def first_mention_share(brand_mentions: pd.DataFrame) -> pd.DataFrame:
    """Per-brand share of mentioned runs in which the brand is mentioned first.

    "First" = smallest ``first_mention_position`` in that run. Ties are broken by
    giving each tied brand a fractional credit so shares still sum to ~1.
    Returns columns: ``brand_name``, ``first_mentions``, ``mentioned_runs``, ``first_mention_share``.
    """
    empty = pd.DataFrame(columns=["brand_name", "first_mentions", "mentioned_runs", "first_mention_share"])
    if brand_mentions.empty:
        return empty
    valid = brand_mentions[brand_mentions["first_mention_position"] >= 0]
    if valid.empty:
        return empty
    credits: dict[str, float] = {}
    runs_with_any = valid["run_id"].nunique()
    for _run_id, grp in valid.groupby("run_id"):
        min_pos = grp["first_mention_position"].min()
        firsts = grp[grp["first_mention_position"] == min_pos]["brand_name"].tolist()
        share = 1.0 / len(firsts)
        for b in firsts:
            credits[b] = credits.get(b, 0.0) + share
    out = pd.DataFrame([{"brand_name": b, "first_mentions": c} for b, c in credits.items()])
    out["mentioned_runs"] = runs_with_any
    out["first_mention_share"] = out["first_mentions"] / runs_with_any
    return out.sort_values("first_mention_share", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Recommendation rate
# ---------------------------------------------------------------------------


def recommendation_rate(brand_mentions: pd.DataFrame, response_runs: pd.DataFrame) -> pd.DataFrame:
    """Per-brand share of all responses in which the brand is recommended.

    Returns columns: ``brand_name``, ``recommended_runs``, ``total_runs``, ``recommendation_rate``.
    """
    n = total_runs(response_runs)
    if brand_mentions.empty or n == 0:
        return pd.DataFrame(columns=["brand_name", "recommended_runs", "total_runs", "recommendation_rate"])
    rec = brand_mentions[brand_mentions["is_recommended"].astype(bool)]
    grouped = rec.groupby("brand_name")["run_id"].nunique().rename("recommended_runs").reset_index()
    grouped["total_runs"] = n
    grouped["recommendation_rate"] = grouped["recommended_runs"] / n
    return grouped.sort_values("recommendation_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Citation rate
# ---------------------------------------------------------------------------


def citation_rate(citations: pd.DataFrame, response_runs: pd.DataFrame) -> dict[str, float]:
    """Overall share of responses containing at least one source URL.

    Returns ``{"runs_with_citations", "total_runs", "citation_rate"}``.
    """
    n = total_runs(response_runs)
    runs_with = int(citations["run_id"].nunique()) if not citations.empty else 0
    return {
        "runs_with_citations": runs_with,
        "total_runs": n,
        "citation_rate": (runs_with / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# 6. Source domain share
# ---------------------------------------------------------------------------


def source_domain_share(citations: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    """Per-domain citation counts and share of all citations.

    Returns columns: ``citation_domain``, ``citations``, ``runs``, ``domain_share``.
    ``runs`` is the number of distinct responses that cited the domain.
    """
    cols = ["citation_domain", "citations", "runs", "domain_share"]
    if citations.empty:
        return pd.DataFrame(columns=cols)
    grouped = (
        citations.groupby("citation_domain")
        .agg(citations=("citation_url", "size"), runs=("run_id", "nunique"))
        .reset_index()
    )
    total = grouped["citations"].sum()
    grouped["domain_share"] = grouped["citations"] / total if total else 0.0
    grouped = grouped.sort_values(["citations", "runs"], ascending=False).reset_index(drop=True)
    return grouped.head(top_n) if top_n else grouped


# ---------------------------------------------------------------------------
# 7 & 8. Breakdown by a prompt attribute (category, persona, etc.)
# ---------------------------------------------------------------------------


def visibility_by_attribute(
    enriched_mentions: pd.DataFrame,
    response_runs: pd.DataFrame,
    prompts: pd.DataFrame,
    attribute: str,
    brand_name: str,
) -> pd.DataFrame:
    """Mention rate for one brand broken down by a prompt attribute.

    ``attribute`` is a column on ``prompts`` such as ``prompt_category`` or
    ``persona``. Denominator is the number of *responses* whose prompt has each
    attribute value (not the number of prompts), so rates are per-response.
    Returns columns: ``<attribute>``, ``mentioned_runs``, ``total_runs``, ``mention_rate``.
    """
    out_cols = [attribute, "mentioned_runs", "total_runs", "mention_rate"]
    if response_runs.empty or prompts.empty:
        return pd.DataFrame(columns=out_cols)
    # Responses per attribute value (denominator).
    runs_ctx = response_runs.merge(prompts[["prompt_id", attribute]], on="prompt_id", how="left")
    denom = runs_ctx.groupby(attribute)["run_id"].nunique().rename("total_runs").reset_index()
    # Responses mentioning the focal brand per attribute value (numerator).
    if enriched_mentions.empty:
        numer = pd.DataFrame(columns=[attribute, "mentioned_runs"])
    else:
        brand_rows = enriched_mentions[enriched_mentions["brand_name"] == brand_name]
        numer = brand_rows.groupby(attribute)["run_id"].nunique().rename("mentioned_runs").reset_index()
    merged = denom.merge(numer, on=attribute, how="left")
    merged["mentioned_runs"] = merged["mentioned_runs"].fillna(0).astype(int)
    merged["mention_rate"] = merged.apply(
        lambda r: (r["mentioned_runs"] / r["total_runs"]) if r["total_runs"] else 0.0, axis=1
    )
    return merged.sort_values("mention_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 9. Platform comparison
# ---------------------------------------------------------------------------


def platform_comparison(
    enriched_mentions: pd.DataFrame, response_runs: pd.DataFrame, brand_name: str
) -> pd.DataFrame:
    """Mention rate for one brand per AI platform.

    Returns columns: ``platform``, ``mentioned_runs``, ``total_runs``, ``mention_rate``.
    """
    out_cols = ["platform", "mentioned_runs", "total_runs", "mention_rate"]
    if response_runs.empty:
        return pd.DataFrame(columns=out_cols)
    denom = response_runs.groupby("platform")["run_id"].nunique().rename("total_runs").reset_index()
    if enriched_mentions.empty:
        numer = pd.DataFrame(columns=["platform", "mentioned_runs"])
    else:
        brand_rows = enriched_mentions[enriched_mentions["brand_name"] == brand_name]
        numer = brand_rows.groupby("platform")["run_id"].nunique().rename("mentioned_runs").reset_index()
    merged = denom.merge(numer, on="platform", how="left")
    merged["mentioned_runs"] = merged["mentioned_runs"].fillna(0).astype(int)
    merged["mention_rate"] = merged.apply(
        lambda r: (r["mentioned_runs"] / r["total_runs"]) if r["total_runs"] else 0.0, axis=1
    )
    return merged.sort_values("mention_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 10. Competitor visibility leaderboard
# ---------------------------------------------------------------------------


def competitor_visibility(brand_mentions: pd.DataFrame, response_runs: pd.DataFrame) -> pd.DataFrame:
    """Combined leaderboard of mention rate + share of voice for all brands.

    Returns columns: ``brand_name``, ``mention_rate``, ``share_of_voice``,
    ``recommended_runs`` ... sorted by share of voice.
    """
    mr = brand_mention_rate(brand_mentions, response_runs)
    sov = share_of_voice(brand_mentions)
    rec = recommendation_rate(brand_mentions, response_runs)
    if mr.empty and sov.empty:
        return pd.DataFrame(columns=["brand_name", "mention_rate", "share_of_voice", "recommendation_rate"])
    out = (
        mr[["brand_name", "mention_rate"]]
        .merge(sov[["brand_name", "share_of_voice"]], on="brand_name", how="outer")
        .merge(rec[["brand_name", "recommendation_rate"]], on="brand_name", how="outer")
        .fillna(0.0)
    )
    return out.sort_values("share_of_voice", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 11. Content coverage gaps
# ---------------------------------------------------------------------------


def content_coverage_gaps(
    enriched_mentions: pd.DataFrame,
    response_runs: pd.DataFrame,
    prompts: pd.DataFrame,
    focal_brand: str,
    dimension: str = "topic",
) -> pd.DataFrame:
    """Topics/categories where the focal brand trails the best competitor.

    For each value of ``dimension`` (default ``topic``), compares the focal brand's
    mention rate against the strongest competitor's mention rate. A positive ``gap``
    means competitors are winning that topic — a candidate content gap.
    Returns columns: ``<dimension>``, ``focal_rate``, ``top_competitor``,
    ``competitor_rate``, ``gap``, ``total_runs``.
    """
    out_cols = [dimension, "focal_rate", "top_competitor", "competitor_rate", "gap", "total_runs"]
    if response_runs.empty or prompts.empty:
        return pd.DataFrame(columns=out_cols)
    runs_ctx = response_runs.merge(prompts[["prompt_id", dimension]], on="prompt_id", how="left")
    denom = runs_ctx.groupby(dimension)["run_id"].nunique().to_dict()

    rows = []
    for dim_value, n in denom.items():
        if not n:
            continue
        sub = enriched_mentions[enriched_mentions[dimension] == dim_value] if not enriched_mentions.empty else enriched_mentions
        focal_runs = sub[sub["brand_name"] == focal_brand]["run_id"].nunique() if not sub.empty else 0
        focal_rate = focal_runs / n
        # Best competitor in this dimension value.
        comp_rate, comp_name = 0.0, None
        if not sub.empty:
            comp = sub[sub["brand_name"] != focal_brand]
            if not comp.empty:
                by_comp = comp.groupby("brand_name")["run_id"].nunique() / n
                comp_name = by_comp.idxmax()
                comp_rate = float(by_comp.max())
        rows.append(
            {
                dimension: dim_value,
                "focal_rate": focal_rate,
                "top_competitor": comp_name,
                "competitor_rate": comp_rate,
                "gap": comp_rate - focal_rate,
                "total_runs": int(n),
            }
        )
    result = pd.DataFrame(rows, columns=out_cols)
    return result.sort_values("gap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 12. Response consistency across repeated runs
# ---------------------------------------------------------------------------


def response_consistency(
    brand_mentions: pd.DataFrame,
    citations: pd.DataFrame,
    response_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Per-prompt consistency across repeated runs, using simple overlap measures.

    Only prompts with 2+ responses are included. For each such prompt we compute:

    * ``brand_overlap`` — mean pairwise Jaccard of the *sets of brands* mentioned.
    * ``recommendation_agreement`` — mean pairwise Jaccard of the sets of
      *recommended* brands (1.0 if all runs recommend the same set; NaN if no run
      recommends anything).
    * ``citation_domain_overlap`` — mean pairwise Jaccard of cited domain sets
      (NaN if no run in the group had citations).
    * ``mention_count_variation`` — coefficient of variation (std/mean) of the total
      mention count per run. 0 = identical volume; higher = noisier.

    Returns one row per prompt plus the number of runs. Aggregate with
    :func:`consistency_summary`.
    """
    cols = [
        "prompt_id", "num_runs", "brand_overlap", "recommendation_agreement",
        "citation_domain_overlap", "mention_count_variation",
    ]
    if response_runs.empty:
        return pd.DataFrame(columns=cols)

    runs_by_prompt = response_runs.groupby("prompt_id")["run_id"].apply(list)
    rows = []
    for prompt_id, run_ids in runs_by_prompt.items():
        if len(run_ids) < 2:
            continue
        brand_sets = {r: _brand_set(brand_mentions, r) for r in run_ids}
        rec_sets = {r: _recommended_set(brand_mentions, r) for r in run_ids}
        domain_sets = {r: _domain_set(citations, r) for r in run_ids}
        counts = {r: _mention_total(brand_mentions, r) for r in run_ids}

        rows.append(
            {
                "prompt_id": prompt_id,
                "num_runs": len(run_ids),
                "brand_overlap": _mean_pairwise_jaccard(list(brand_sets.values())),
                "recommendation_agreement": _mean_pairwise_jaccard(
                    list(rec_sets.values()), skip_all_empty=True
                ),
                "citation_domain_overlap": _mean_pairwise_jaccard(
                    list(domain_sets.values()), skip_all_empty=True
                ),
                "mention_count_variation": _coefficient_of_variation(list(counts.values())),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def consistency_summary(consistency_df: pd.DataFrame) -> dict[str, Optional[float]]:
    """Aggregate per-prompt consistency into headline averages.

    Returns means (ignoring NaN) for each measure plus ``prompts_with_repeats``.
    """
    if consistency_df.empty:
        return {
            "prompts_with_repeats": 0,
            "avg_brand_overlap": None,
            "avg_recommendation_agreement": None,
            "avg_citation_domain_overlap": None,
            "avg_mention_count_variation": None,
        }
    return {
        "prompts_with_repeats": int(len(consistency_df)),
        "avg_brand_overlap": _nan_mean(consistency_df["brand_overlap"]),
        "avg_recommendation_agreement": _nan_mean(consistency_df["recommendation_agreement"]),
        "avg_citation_domain_overlap": _nan_mean(consistency_df["citation_domain_overlap"]),
        "avg_mention_count_variation": _nan_mean(consistency_df["mention_count_variation"]),
    }


# ---------------------------------------------------------------------------
# Internal set/statistics helpers for consistency.
# ---------------------------------------------------------------------------


def _brand_set(brand_mentions: pd.DataFrame, run_id: str) -> set[str]:
    if brand_mentions.empty:
        return set()
    return set(brand_mentions[brand_mentions["run_id"] == run_id]["brand_name"].tolist())


def _recommended_set(brand_mentions: pd.DataFrame, run_id: str) -> set[str]:
    if brand_mentions.empty:
        return set()
    sub = brand_mentions[(brand_mentions["run_id"] == run_id) & (brand_mentions["is_recommended"].astype(bool))]
    return set(sub["brand_name"].tolist())


def _domain_set(citations: pd.DataFrame, run_id: str) -> set[str]:
    if citations.empty:
        return set()
    return set(citations[citations["run_id"] == run_id]["citation_domain"].tolist())


def _mention_total(brand_mentions: pd.DataFrame, run_id: str) -> int:
    if brand_mentions.empty:
        return 0
    return int(brand_mentions[brand_mentions["run_id"] == run_id]["mention_count"].sum())


def _jaccard(a: set, b: set) -> Optional[float]:
    """Jaccard similarity of two sets. Returns NaN-like None if both are empty."""
    if not a and not b:
        return None
    union = a | b
    return len(a & b) / len(union) if union else None


def _mean_pairwise_jaccard(sets: list[set], skip_all_empty: bool = False) -> float:
    """Mean Jaccard over all unordered pairs of sets.

    If ``skip_all_empty`` is True, pairs where both sets are empty are ignored
    (used for recommendation/citation agreement, where "nobody recommended anything"
    should not be scored as perfect agreement). Returns NaN if no scorable pairs.
    """
    scores = []
    for a, b in combinations(sets, 2):
        j = _jaccard(a, b)
        if j is None:
            if skip_all_empty:
                continue
            j = 1.0  # both empty counts as identical when not skipping
        scores.append(j)
    return float(sum(scores) / len(scores)) if scores else float("nan")


def _coefficient_of_variation(values: list[float]) -> float:
    """std/mean of a list; 0 if all equal; NaN if mean is 0."""
    s = pd.Series(values, dtype="float64")
    mean = s.mean()
    if mean == 0:
        return float("nan")
    return float(s.std(ddof=0) / mean)


def _nan_mean(series: pd.Series) -> Optional[float]:
    val = series.dropna().mean()
    return None if pd.isna(val) else float(val)


def is_small_sample(n: int) -> bool:
    """True when a metric is based on fewer than the small-sample threshold of runs."""
    return n < SMALL_SAMPLE_THRESHOLD
