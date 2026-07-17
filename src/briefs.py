"""Content Action Briefs (feature: Content Action Briefs).

For each content-coverage gap (a topic where competitors out-appear the focal brand),
generate a deterministic, editable content brief. Every field is grounded in calculated
gaps, the actual prompts behind the topic, and the citation-opportunity analysis — the
briefs invent nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import metrics as M
from .citation_quality import citation_opportunities
from .database import AnalysisData

# Deterministic mappings from prompt category to content guidance.
CATEGORY_FORMAT: dict[str, str] = {
    "Product comparison": "Comparison / “vs.” page",
    "Purchase intent": "Pricing & plans comparison page",
    "Informational": "Educational guide / explainer",
    "Problem based": "How-to / solution guide",
    "Customer persona": "Use-case / persona landing page",
    "Brand specific": "Product capability page + FAQ",
    "Nonbrand discovery": "Best-of / category roundup page",
}

CATEGORY_SCHEMA: dict[str, str] = {
    "Product comparison": "FAQPage + Product",
    "Purchase intent": "Product + Offer",
    "Informational": "Article",
    "Problem based": "HowTo",
    "Customer persona": "Article + FAQPage",
    "Brand specific": "FAQPage + Organization",
    "Nonbrand discovery": "Article + ItemList",
}

CATEGORY_EVIDENCE: dict[str, str] = {
    "Product comparison": "Feature-by-feature table, pricing, screenshots, migration notes",
    "Purchase intent": "Transparent pricing tiers, total-cost examples, free-plan limits",
    "Informational": "Clear definitions, examples, and a concise direct answer up top",
    "Problem based": "Step-by-step instructions, before/after outcomes, a checklist",
    "Customer persona": "Persona-specific workflows, testimonials, time-saved examples",
    "Brand specific": "Capability details, honest limits, FAQ answering objections",
    "Nonbrand discovery": "Objective selection criteria, a shortlist with pros/cons, sources",
}

_DEFAULT_FORMAT = "Educational guide / explainer"
_DEFAULT_SCHEMA = "Article"
_DEFAULT_EVIDENCE = "A concise direct answer, concrete examples, and cited sources"


@dataclass
class ContentBrief:
    """One deterministic content brief for a single coverage gap."""

    topic: str
    focal_brand: str
    gap_summary: str
    target_persona: str
    journey_stage: str
    prompt_category: str
    questions_to_answer: list[str] = field(default_factory=list)
    suggested_format: str = _DEFAULT_FORMAT
    suggested_title: str = ""
    suggested_headings: list[str] = field(default_factory=list)
    recommended_schema: str = _DEFAULT_SCHEMA
    evidence_needed: str = _DEFAULT_EVIDENCE
    source_opportunities: list[str] = field(default_factory=list)


def _mode(series: pd.Series, default: str) -> str:
    """Most common non-empty value in a series, or a default."""
    vals = series.dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    if vals.empty:
        return default
    return vals.value_counts().index[0]


def _title_for(category: str, topic: str, brand: str, persona: str) -> str:
    if category == "Product comparison":
        return f"{brand} vs. the Alternatives for {topic}"
    if category == "Purchase intent":
        return f"{brand} Pricing & Plans: Is It Right for {topic}?"
    if category == "Nonbrand discovery":
        return f"Best Tools for {topic} (and Where {brand} Fits)"
    if category == "Problem based":
        return f"How to Solve {topic} with {brand}"
    if category == "Customer persona":
        return f"{brand} for {persona}: {topic}"
    return f"{topic}: A Practical Guide (Featuring {brand})"


def _headings_for(category: str, questions: list[str], brand: str) -> list[str]:
    base = [f"What to know about {category.lower()} for this topic"]
    # Turn up to four prompts into question-style H2s (AI answers favour these).
    base.extend(q if q.endswith("?") else q + "?" for q in questions[:4])
    if category == "Product comparison":
        base.append(f"Feature and pricing comparison table")
    if category in ("Purchase intent",):
        base.append("Plans, pricing, and total cost")
    base.append(f"When {brand} is (and isn’t) the right choice")
    base.append("Key takeaways")
    # De-dupe preserving order.
    seen, out = set(), []
    for h in base:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def build_briefs(
    data: AnalysisData,
    focal_brand: str,
    max_briefs: int = 8,
    min_gap: float = 0.0,
) -> list[ContentBrief]:
    """Build content briefs for the focal brand's top coverage gaps (by topic).

    Only topics where a competitor out-appears the focal brand (``gap`` > ``min_gap``)
    become briefs. Returns at most ``max_briefs`` briefs, most severe gap first.
    """
    if data.response_runs.empty or data.prompts.empty:
        return []

    enriched = M.enrich_mentions(data.brand_mentions, data.response_runs, data.prompts)
    gaps = M.content_coverage_gaps(enriched, data.response_runs, data.prompts, focal_brand, dimension="topic")
    gaps = gaps[gaps["gap"] > min_gap].head(max_briefs)
    if gaps.empty:
        return []

    opps = citation_opportunities(data.citations, data.brand_mentions, data.brands, focal_brand, top_n=None)

    briefs: list[ContentBrief] = []
    for _, g in gaps.iterrows():
        topic = g["topic"]
        topic_prompts = data.prompts[data.prompts["topic"] == topic]
        category = _mode(topic_prompts["prompt_category"], "Informational")
        persona = _mode(topic_prompts["persona"], "Team Lead")
        stage = _mode(topic_prompts["journey_stage"], "Consideration")
        questions = [q for q in topic_prompts["prompt_text"].dropna().astype(str).tolist() if q.strip()]

        comp = g["top_competitor"] or "a competitor"
        gap_summary = (
            f"{focal_brand} appears in {round(g['focal_rate']*100)}% of '{topic}' responses vs "
            f"{round(g['competitor_rate']*100)}% for {comp} (gap {round(g['gap']*100)} pts, "
            f"{int(g['total_runs'])} responses)."
        )

        # Third-party source opportunities relevant to this brand (review/news/forum).
        source_ops = []
        if not opps.empty:
            relevant = opps[opps["source_type"].isin(["Review site", "News or media", "Forum or community"])]
            for _, o in relevant.head(3).iterrows():
                source_ops.append(
                    f"{o['citation_domain']} ({o['source_type']}) — cited alongside competitors in "
                    f"{int(o['runs_with_competitor'])} responses vs {int(o['runs_with_focal'])} mentioning {focal_brand}."
                )

        briefs.append(
            ContentBrief(
                topic=topic,
                focal_brand=focal_brand,
                gap_summary=gap_summary,
                target_persona=persona,
                journey_stage=stage,
                prompt_category=category,
                questions_to_answer=questions,
                suggested_format=CATEGORY_FORMAT.get(category, _DEFAULT_FORMAT),
                suggested_title=_title_for(category, topic, focal_brand, persona),
                suggested_headings=_headings_for(category, questions, focal_brand),
                recommended_schema=CATEGORY_SCHEMA.get(category, _DEFAULT_SCHEMA),
                evidence_needed=CATEGORY_EVIDENCE.get(category, _DEFAULT_EVIDENCE),
                source_opportunities=source_ops,
            )
        )
    return briefs


def brief_to_markdown(brief: ContentBrief) -> str:
    """Render one brief as Markdown for export."""
    lines = [
        f"### Content brief — {brief.topic}",
        f"_{brief.gap_summary}_",
        "",
        f"- **Target persona:** {brief.target_persona}",
        f"- **Journey stage:** {brief.journey_stage}",
        f"- **Prompt topic:** {brief.topic}",
        f"- **Prompt category:** {brief.prompt_category}",
        f"- **Suggested content format:** {brief.suggested_format}",
        f"- **Suggested title:** {brief.suggested_title}",
        f"- **Recommended schema type:** {brief.recommended_schema}",
        f"- **Evidence / examples needed:** {brief.evidence_needed}",
        "",
        "**Questions to answer:**",
    ]
    if brief.questions_to_answer:
        lines.extend(f"- {q}" for q in brief.questions_to_answer)
    else:
        lines.append("- (no prompts found for this topic)")
    lines.append("")
    lines.append("**Suggested headings:**")
    lines.extend(f"- {h}" for h in brief.suggested_headings)
    lines.append("")
    lines.append("**Relevant third-party source opportunities:**")
    if brief.source_opportunities:
        lines.extend(f"- {s}" for s in brief.source_opportunities)
    else:
        lines.append("- None identified for this topic (association-based; investigate manually).")
    return "\n".join(lines)
