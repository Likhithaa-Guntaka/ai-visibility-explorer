"""Brand truth & freshness monitor (AI Decision Influence Lab).

Compares extracted AI claims against **authoritative brand facts the user provides**.
This is an *authoritative-source comparison*, NOT a verification of absolute truth — the
app cannot know ground truth; it only flags where AI answers agree, disagree, look
outdated, or lack evidence relative to the facts you entered.

Matching is deterministic and every verdict is editable.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .database import BRAND_FACTS_COLUMNS

# Verdicts (also in database.TRUTH_VERDICTS).
V_SUPPORTED = "Supported"
V_PARTIAL = "Partially supported"
V_CONFLICTING = "Conflicting"
V_OUTDATED = "Outdated"
V_UNVERIFIABLE = "Unverifiable"
V_MISSING = "Missing from AI responses"

# Which claim types are relevant to each authoritative fact type.
_FACT_TO_CLAIM_TYPES: dict[str, list[str]] = {
    "Pricing": ["Pricing claim"],
    "Feature": ["Product capability", "Positioning claim"],
    "Product name": ["Product capability", "Positioning claim"],
    "Integration": ["Product capability"],
    "Customer segment": ["Customer suitability claim"],
    "Company description": ["Positioning claim"],
    "Supported location": ["Positioning claim", "Customer suitability claim"],
    "Launch date": ["Product capability", "Positioning claim"],
    "Discontinued feature": ["Product capability"],
    "Official source URL": [],
}

# Antonym cues signalling a conflict for pricing facts.
_PRICING_CONFLICTS = {
    "free": ["no free", "paid only", "no free plan", "expensive"],
    "affordable": ["expensive", "pricey", "costly"],
    "cheap": ["expensive", "pricey"],
}

_STOP = {"the", "a", "an", "and", "or", "of", "for", "to", "is", "are", "with", "plan", "plans", "available"}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _STOP and len(w) > 2]


def _brand_claim_text(claims: pd.DataFrame, brand: str, claim_types: list[str]) -> str:
    if claims.empty or not claim_types:
        return ""
    sub = claims[(claims["brand_name"] == brand) & (claims["claim_type"].isin(claim_types))]
    if sub.empty:
        return ""
    return " ".join((sub["claim_text"].fillna("") + " " + sub["evidence_text"].fillna("")).tolist()).lower()


def _run_evidence(claims: pd.DataFrame, citations: pd.DataFrame, brand: str, claim_types: list[str]) -> tuple[str, str]:
    """Return (a representative AI response snippet, a citation domain) for the brand's related claims."""
    if claims.empty:
        return "", ""
    sub = claims[(claims["brand_name"] == brand) & (claims["claim_type"].isin(claim_types))]
    if sub.empty:
        return "", ""
    evidence = str(sub.iloc[0].get("evidence_text", ""))[:200]
    run_ids = set(sub["run_id"])
    citation = ""
    if not citations.empty:
        c = citations[citations["run_id"].isin(run_ids)]
        if not c.empty:
            citation = str(c.iloc[0]["citation_domain"])
    return evidence, citation


_RISK_ACTION = {
    V_CONFLICTING: ("High", "Publish an authoritative page and seek corrections where third-party sources disagree."),
    V_OUTDATED: ("High", "Clarify the current state; request updates where a removed/old detail is still described."),
    V_UNVERIFIABLE: ("Medium", "Add citable evidence (docs, pricing page) so answers can ground this claim."),
    V_PARTIAL: ("Medium", "Strengthen the exact detail on an authoritative, citable page."),
    V_MISSING: ("Medium", "Create clear, citable content stating this fact so AI answers can pick it up."),
    V_SUPPORTED: ("Low", "Maintain the authoritative page and monitor for drift over time."),
}


def compare_facts(
    brand_facts: pd.DataFrame,
    claims: pd.DataFrame,
    response_runs: pd.DataFrame,
    citations: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each authoritative fact with related AI claims and assign a verdict.

    Returns columns: ``brand_name``, ``fact_type``, ``fact_value``, ``source_url``,
    ``as_of_date``, ``verdict``, ``ai_claim``, ``ai_response_text``, ``ai_citation``,
    ``business_risk``, ``recommended_action``. Deterministic and editable.
    """
    cols = ["brand_name", "fact_type", "fact_value", "source_url", "as_of_date", "verdict",
            "ai_claim", "ai_response_text", "ai_citation", "business_risk", "recommended_action"]
    if brand_facts is None or brand_facts.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, fact in brand_facts.iterrows():
        brand = str(fact["brand_name"])
        ftype = str(fact["fact_type"])
        fvalue = str(fact.get("fact_value", ""))
        claim_types = _FACT_TO_CLAIM_TYPES.get(ftype, [])
        related_text = _brand_claim_text(claims, brand, claim_types)
        ai_evidence, ai_citation = _run_evidence(claims, citations, brand, claim_types)

        verdict = _classify_fact(ftype, fvalue, related_text)
        risk, action = _RISK_ACTION.get(verdict, ("Medium", "Review manually."))
        rows.append({
            "brand_name": brand, "fact_type": ftype, "fact_value": fvalue,
            "source_url": str(fact.get("source_url", "")), "as_of_date": str(fact.get("as_of_date", "")),
            "verdict": verdict, "ai_claim": related_text[:160],
            "ai_response_text": ai_evidence, "ai_citation": ai_citation,
            "business_risk": risk, "recommended_action": action,
        })
    return pd.DataFrame(rows, columns=cols)


def _classify_fact(fact_type: str, fact_value: str, related_text: str) -> str:
    """Deterministic verdict for one fact vs the brand's related AI claim text."""
    if not related_text:
        return V_MISSING

    # Discontinued features: if AI still describes the feature, it's Outdated.
    if fact_type == "Discontinued feature":
        feat_tokens = _tokens(fact_value)
        if feat_tokens and any(t in related_text for t in feat_tokens):
            return V_OUTDATED
        return V_SUPPORTED  # AI does not tout the removed feature -> no contradiction

    # Pricing conflicts (free/affordable vs expensive/paid-only).
    if fact_type == "Pricing":
        low = fact_value.lower()
        for anchor, opposites in _PRICING_CONFLICTS.items():
            if anchor in low and any(op in related_text for op in opposites):
                return V_CONFLICTING

    value_tokens = _tokens(fact_value)
    if not value_tokens:
        return V_UNVERIFIABLE
    hits = sum(1 for t in set(value_tokens) if t in related_text)
    ratio = hits / len(set(value_tokens))
    if ratio >= 0.6:
        return V_SUPPORTED
    if ratio > 0:
        return V_PARTIAL
    # Topic claims exist but none of the specific value tokens appear.
    return V_UNVERIFIABLE


def truth_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Verdict counts. Columns: ``verdict``, ``count``, ``share``."""
    cols = ["verdict", "count", "share"]
    if comparison.empty:
        return pd.DataFrame(columns=cols)
    grp = comparison.groupby("verdict").size().rename("count").reset_index()
    total = grp["count"].sum()
    grp["share"] = grp["count"] / total if total else 0.0
    return grp.sort_values("count", ascending=False).reset_index(drop=True)


def demo_brand_facts() -> pd.DataFrame:
    """A small set of SYNTHETIC authoritative facts for the demo brands.

    These are invented for demonstration — not verified real facts. They are chosen to
    exercise every verdict (a pricing conflict for ClickUp, an outdated discontinued
    feature for Trello, a missing fact, etc.).
    """
    rows = [
        {"brand_name": "Notion", "fact_type": "Pricing", "fact_value": "Free plan available for individuals",
         "source_url": "https://www.notion.so/pricing", "as_of_date": "2026-06-01"},
        {"brand_name": "Notion", "fact_type": "Feature", "fact_value": "Databases and wikis",
         "source_url": "https://www.notion.so/product", "as_of_date": "2026-06-01"},
        {"brand_name": "ClickUp", "fact_type": "Pricing", "fact_value": "Affordable free forever plan",
         "source_url": "https://clickup.com/pricing", "as_of_date": "2026-06-01"},
        {"brand_name": "Trello", "fact_type": "Discontinued feature", "fact_value": "Legacy dashboard view",
         "source_url": "https://trello.com/changelog", "as_of_date": "2026-05-15"},
        {"brand_name": "Asana", "fact_type": "Customer segment", "fact_value": "Marketing teams and larger teams",
         "source_url": "https://asana.com/uses", "as_of_date": "2026-06-01"},
        {"brand_name": "Monday.com", "fact_type": "Integration", "fact_value": "Slack and Google Drive integrations",
         "source_url": "https://monday.com/integrations", "as_of_date": "2026-06-01"},
        {"brand_name": "Notion", "fact_type": "Supported location", "fact_value": "Available worldwide in many languages",
         "source_url": "https://www.notion.so", "as_of_date": "2026-06-01"},
    ]
    return pd.DataFrame(rows, columns=BRAND_FACTS_COLUMNS)
