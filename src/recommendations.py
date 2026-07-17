"""Deterministic, metric-grounded customer-facing readout.

Every sentence produced here is derived from a computed metric — the templates never
invent findings. This is the honest default for the MVP. An optional LLM narrative
adapter is provided separately (:func:`generate_llm_narrative`); it is clearly
labelled as AI-generated and is instructed to stay grounded in the same metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import metrics as M
from .database import AnalysisData


@dataclass
class Readout:
    """Structured customer-facing summary. Each field is a list of plain sentences."""

    executive_summary: list[str] = field(default_factory=list)
    strongest_areas: list[str] = field(default_factory=list)
    weakest_areas: list[str] = field(default_factory=list)
    competitors_gaining: list[str] = field(default_factory=list)
    frequent_sources: list[str] = field(default_factory=list)
    content_gaps: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    limitations_confidence: list[str] = field(default_factory=list)


def _pct(x: float) -> str:
    """Format a 0-1 fraction as a percentage string."""
    return f"{round(x * 100)}%"


def build_readout(data: AnalysisData, focal_brand: str) -> Readout:
    """Assemble the full deterministic readout for ``focal_brand``.

    All numbers come from :mod:`src.metrics`. If there is no data, the readout still
    returns safe, honest placeholder sentences rather than raising.
    """
    r = Readout()
    n = M.total_runs(data.response_runs)

    if n == 0:
        r.executive_summary.append(
            "No AI responses have been added yet, so no visibility can be measured. "
            "Add responses (paste or CSV) to generate findings."
        )
        r.limitations_confidence.append("There is no data to analyze yet.")
        return r

    mr = M.brand_mention_rate(data.brand_mentions, data.response_runs)
    sov = M.share_of_voice(data.brand_mentions)
    fms = M.first_mention_share(data.brand_mentions)
    rec = M.recommendation_rate(data.brand_mentions, data.response_runs)
    cite = M.citation_rate(data.citations, data.response_runs)
    domains = M.source_domain_share(data.citations, top_n=5)
    leaderboard = M.competitor_visibility(data.brand_mentions, data.response_runs)
    enriched = M.enrich_mentions(data.brand_mentions, data.response_runs, data.prompts)
    gaps = M.content_coverage_gaps(enriched, data.response_runs, data.prompts, focal_brand, dimension="topic")
    consistency = M.consistency_summary(
        M.response_consistency(data.brand_mentions, data.citations, data.response_runs)
    )

    focal_mr = _lookup(mr, "brand_name", focal_brand, "mention_rate", 0.0)
    focal_sov = _lookup(sov, "brand_name", focal_brand, "share_of_voice", 0.0)
    focal_rank = _rank(leaderboard, "brand_name", focal_brand)
    n_brands = len(leaderboard)

    # -- Executive summary ---------------------------------------------------
    rank_txt = f"ranks #{focal_rank} of {n_brands}" if focal_rank else "was not ranked"
    r.executive_summary.append(
        f"Across {n} synthetic AI responses, {focal_brand} was mentioned in "
        f"{_pct(focal_mr)} of answers and holds {_pct(focal_sov)} share of voice, and {rank_txt} "
        f"among tracked brands."
    )
    if not leaderboard.empty:
        leader = leaderboard.iloc[0]
        if leader["brand_name"] != focal_brand:
            r.executive_summary.append(
                f"The most visible brand is {leader['brand_name']} "
                f"({_pct(leader['share_of_voice'])} share of voice), which {focal_brand} trails."
            )
        else:
            r.executive_summary.append(
                f"{focal_brand} is currently the most visible tracked brand by share of voice."
            )
    r.executive_summary.append(
        f"{_pct(cite['citation_rate'])} of responses included at least one source link."
    )

    # -- Strongest areas -----------------------------------------------------
    cat_perf = M.visibility_by_attribute(enriched, data.response_runs, data.prompts, "prompt_category", focal_brand)
    strong = cat_perf[cat_perf["mention_rate"] > 0].head(2)
    for _, row in strong.iterrows():
        r.strongest_areas.append(
            f"{focal_brand} appears in {_pct(row['mention_rate'])} of '{row['prompt_category']}' "
            f"responses ({int(row['mentioned_runs'])} of {int(row['total_runs'])})."
        )
    focal_fms = _lookup(fms, "brand_name", focal_brand, "first_mention_share", 0.0)
    if focal_fms > 0:
        r.strongest_areas.append(
            f"{focal_brand} is mentioned first in {_pct(focal_fms)} of answers that name any tracked brand."
        )
    if not r.strongest_areas:
        r.strongest_areas.append(f"No clear strong areas for {focal_brand} in this sample.")

    # -- Weakest areas -------------------------------------------------------
    weak = cat_perf.sort_values("mention_rate").head(2)
    for _, row in weak.iterrows():
        r.weakest_areas.append(
            f"In '{row['prompt_category']}' prompts, {focal_brand} appears in only "
            f"{_pct(row['mention_rate'])} of responses."
        )
    if not r.weakest_areas:
        r.weakest_areas.append("Not enough category coverage to identify weak areas.")

    # -- Competitors gaining / ahead ----------------------------------------
    ahead = leaderboard[leaderboard["brand_name"] != focal_brand]
    ahead = ahead[ahead["share_of_voice"] >= focal_sov].head(3)
    for _, row in ahead.iterrows():
        r.competitors_gaining.append(
            f"{row['brand_name']} leads {focal_brand} with {_pct(row['share_of_voice'])} share of voice "
            f"and a {_pct(row['mention_rate'])} mention rate."
        )
    if not r.competitors_gaining:
        r.competitors_gaining.append(f"No tracked competitor currently outranks {focal_brand} by share of voice.")

    # -- Frequently cited sources -------------------------------------------
    for _, row in domains.iterrows():
        r.frequent_sources.append(
            f"{row['citation_domain']} — {int(row['citations'])} citations "
            f"({_pct(row['domain_share'])} of all sources)."
        )
    if not r.frequent_sources:
        r.frequent_sources.append("No source URLs were found in the responses.")

    # -- Content gaps --------------------------------------------------------
    real_gaps = gaps[gaps["gap"] > 0].head(4)
    for _, row in real_gaps.iterrows():
        comp = row["top_competitor"] or "a competitor"
        r.content_gaps.append(
            f"On the topic '{row['topic']}', {focal_brand} appears in {_pct(row['focal_rate'])} of "
            f"responses vs {_pct(row['competitor_rate'])} for {comp} — a gap of "
            f"{_pct(row['gap'])} ({int(row['total_runs'])} responses)."
        )
    if not r.content_gaps:
        r.content_gaps.append(f"No clear topic gaps where competitors beat {focal_brand} in this sample.")

    # -- Next actions (grounded, prioritized) --------------------------------
    if not real_gaps.empty:
        top_gap = real_gaps.iloc[0]
        r.next_actions.append(
            f"Prioritize content for '{top_gap['topic']}', where {top_gap['top_competitor'] or 'a competitor'} "
            f"currently out-appears {focal_brand}."
        )
    weakest_cat = cat_perf.sort_values("mention_rate").head(1)
    if not weakest_cat.empty:
        r.next_actions.append(
            f"Strengthen '{weakest_cat.iloc[0]['prompt_category']}' content, {focal_brand}'s weakest "
            f"prompt category ({_pct(weakest_cat.iloc[0]['mention_rate'])} mention rate)."
        )
    if not domains.empty:
        r.next_actions.append(
            f"Study the top cited sources (e.g. {domains.iloc[0]['citation_domain']}) to understand which "
            f"third-party pages influence recommendations, and seek presence there."
        )
    r.next_actions.append(
        "Re-run the same prompts on a schedule to measure whether visibility improves after content changes."
    )

    # -- Limitations & confidence -------------------------------------------
    r.limitations_confidence.append(
        f"This analysis is based on {n} responses across {data.response_runs['platform'].nunique()} "
        f"platform label(s). Treat it as directional, not definitive."
    )
    if M.is_small_sample(n):
        r.limitations_confidence.append(
            f"⚠ Small sample: fewer than {M.SMALL_SAMPLE_THRESHOLD * 4} responses means individual "
            "percentages can swing a lot with one more answer."
        )
    if consistency["avg_brand_overlap"] is not None:
        r.limitations_confidence.append(
            f"Across repeated runs, brand overlap averaged {_pct(consistency['avg_brand_overlap'])}; "
            "results vary run-to-run, so a single run should not be over-interpreted."
        )
    r.limitations_confidence.append(
        "All responses in the demo are clearly labelled synthetic data, not real outputs from any AI platform. "
        "Technical page traits are associations, not proven causes of citations."
    )
    return r


def readout_to_markdown(readout: Readout, focal_brand: str, project_name: str) -> str:
    """Render a :class:`Readout` as a Markdown document for export."""
    sections = [
        ("Executive summary", readout.executive_summary),
        ("Strongest visibility areas", readout.strongest_areas),
        ("Weakest visibility areas", readout.weakest_areas),
        ("Competitors ahead / gaining", readout.competitors_gaining),
        ("Frequently cited sources", readout.frequent_sources),
        ("Content gaps", readout.content_gaps),
        ("Recommended next actions", readout.next_actions),
        ("Limitations and confidence", readout.limitations_confidence),
    ]
    lines = [
        f"# AI Visibility Readout — {focal_brand}",
        f"_Project: {project_name}. Based on synthetic/user-provided data. Directional, not definitive._",
        "",
    ]
    for title, items in sections:
        lines.append(f"## {title}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional LLM narrative (Phase 2). Deterministic templates remain the default.
# ---------------------------------------------------------------------------


def generate_llm_narrative(readout: Readout, focal_brand: str, api_key: Optional[str]) -> str:
    """Optionally turn the grounded readout into prose using an LLM.

    This does NOT compute new findings — it is given the already-computed, grounded
    bullet points and asked only to rephrase them. If no key is supplied it raises,
    so callers can fall back to the deterministic Markdown. The returned text must be
    labelled as AI-generated by the caller.
    """
    if not api_key:
        raise RuntimeError(
            "No LLM API key provided. The deterministic readout is the default; "
            "set an API key in .env to enable optional narrative generation."
        )
    # Intentionally not implemented in the MVP to avoid requiring a paid API.
    # A Phase-2 implementation would send the grounded bullets (below) to the model
    # with a strict instruction: rephrase only, invent nothing.
    raise NotImplementedError(
        "LLM narrative generation is a Phase 2 feature. The grounded bullets are "
        "available via build_readout(); wire them to your provider of choice."
    )


# ---------------------------------------------------------------------------
# Small lookup helpers.
# ---------------------------------------------------------------------------


def _lookup(df: pd.DataFrame, key_col: str, key: str, value_col: str, default: float) -> float:
    if df.empty:
        return default
    hit = df[df[key_col] == key]
    if hit.empty:
        return default
    return float(hit.iloc[0][value_col])


def _rank(df: pd.DataFrame, key_col: str, key: str) -> Optional[int]:
    if df.empty or key not in set(df[key_col]):
        return None
    return int(df.reset_index(drop=True).index[df[key_col].values == key][0]) + 1
