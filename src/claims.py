"""Claim extraction + provenance (AI Decision Influence Lab).

Extracts typed claims about each tracked brand from response text using transparent
keyword lexicons, then connects each claim to its response, prompt, platform, citations
and recommendation outcome so a reviewer can trace provenance.

Responsible-language note: a claim "appearing alongside" a citation or a recommendation
is an *association*. This module never asserts a citation caused a recommendation.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .database import BRAND_CLAIMS_COLUMNS
from .entities import _brand_context, _split
from .extraction import _iter_term_matches, _unique_terms

# Claim type -> signalling phrases (lowercase). Ordered as database.CLAIM_TYPES.
CLAIM_LEXICON: dict[str, list[str]] = {
    "Product capability": ["docs", "wikis", "databases", "boards", "dashboards", "automations",
                            "timelines", "task tracking", "goals", "integrations", "reporting", "templates"],
    "Pricing claim": ["free plan", "affordable", "expensive", "pricing", "per user", "pricey", "budget", "cost"],
    "Positioning claim": ["all-in-one", "all in one", "workspace", "platform", "alternative", "lightweight", "kanban"],
    "Performance claim": ["powerful", "fast", "reliable", "robust", "scales", "feature-rich", "mature"],
    "Ease of use claim": ["easy to use", "easy to learn", "intuitive", "approachable", "simple", "user-friendly"],
    "Customer suitability claim": ["small teams", "startups", "enterprise", "agencies", "non-technical",
                                   "freelancers", "marketing teams", "larger teams", "remote teams", "engineering teams"],
    "Limitation": ["overwhelming", "limited", "lacks", "steep learning curve", "cluttered", "expensive", "not ideal"],
    "Comparative claim": ["better than", "compared to", " vs ", "alternative to", "unlike", "outperforms", "as an alternative"],
}


def _matched_phrases(context: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if next(_iter_term_matches(context, p), None) is not None]


def extract_all_claims(
    response_runs: pd.DataFrame,
    brand_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    """Extract typed claims for every (response, mentioned brand).

    Emits one row per (run, brand, claim_type) that has at least one matched phrase.
    ``claim_text`` is the matched phrases; ``evidence_text`` is the brand's context block.
    Returns a DataFrame matching ``BRAND_CLAIMS_COLUMNS``.
    """
    if response_runs.empty or not brand_aliases:
        return pd.DataFrame(columns=BRAND_CLAIMS_COLUMNS)
    all_terms = {b: _unique_terms([b, *aliases]) for b, aliases in brand_aliases.items()}
    rows: list[dict] = []
    for _, run in response_runs.iterrows():
        run_id = str(run["run_id"])
        text = str(run.get("response_text", ""))
        lowered_full = text.lower()
        for brand, terms in all_terms.items():
            if not any(next(_iter_term_matches(lowered_full, t.lower()), None) is not None for t in terms):
                continue  # brand not mentioned -> no claims
            context = _brand_context(text, terms)
            if not context:
                continue
            for claim_type, phrases in CLAIM_LEXICON.items():
                matched = _matched_phrases(context, phrases)
                if not matched:
                    continue
                rows.append({
                    "claim_id": f"c_{run_id}_{brand}_{claim_type}".replace(" ", "_"),
                    "run_id": run_id,
                    "brand_name": brand,
                    "claim_type": claim_type,
                    "claim_text": ", ".join(matched),
                    "evidence_text": context[:240],
                })
    return pd.DataFrame(rows, columns=BRAND_CLAIMS_COLUMNS)


# ---------------------------------------------------------------------------
# Provenance + analyses.
# ---------------------------------------------------------------------------


def claims_with_provenance(
    claims: pd.DataFrame,
    response_runs: pd.DataFrame,
    citations: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Join each claim to platform, run_date, prompt_id, outcome and citation info.

    Adds ``platform``, ``run_date``, ``prompt_id``, ``outcome``, ``has_citation`` and
    ``citation_domains`` (a comma list). This is the provenance relationship table.
    """
    if claims.empty:
        return claims.assign(platform=pd.Series(dtype=str), outcome=pd.Series(dtype=str),
                             has_citation=pd.Series(dtype=bool), citation_domains=pd.Series(dtype=str))
    runs = response_runs[["run_id", "prompt_id", "platform", "run_date"]] if not response_runs.empty else pd.DataFrame(columns=["run_id", "prompt_id", "platform", "run_date"])
    out = claims.merge(runs, on="run_id", how="left")

    # Outcome per (run, brand).
    if not outcomes.empty:
        out = out.merge(outcomes[["run_id", "brand_name", "outcome"]], on=["run_id", "brand_name"], how="left")
    else:
        out["outcome"] = None

    # Citation presence + domains per run.
    if not citations.empty:
        dom = citations.groupby("run_id")["citation_domain"].apply(lambda s: ", ".join(sorted(set(s)))).rename("citation_domains").reset_index()
        out = out.merge(dom, on="run_id", how="left")
    else:
        out["citation_domains"] = None
    out["has_citation"] = out["citation_domains"].notna() & (out["citation_domains"].astype(str) != "")
    return out


def claim_frequency(claims: pd.DataFrame, brand_name: Optional[str] = None) -> pd.DataFrame:
    """Most common claim types. Columns: ``claim_type``, ``claims``, ``runs``, ``share``."""
    cols = ["claim_type", "claims", "runs", "share"]
    if claims.empty:
        return pd.DataFrame(columns=cols)
    sub = claims if brand_name is None else claims[claims["brand_name"] == brand_name]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    grp = sub.groupby("claim_type").agg(claims=("claim_id", "size"), runs=("run_id", "nunique")).reset_index()
    total = grp["claims"].sum()
    grp["share"] = grp["claims"] / total if total else 0.0
    return grp.sort_values("claims", ascending=False).reset_index(drop=True)


def claims_by_outcome(provenance: pd.DataFrame, brand_name: str, outcome_value: str) -> pd.DataFrame:
    """Claim types appearing in responses with a given outcome for a brand.

    Columns: ``claim_type``, ``claims``.
    """
    cols = ["claim_type", "claims"]
    if provenance.empty or "outcome" not in provenance.columns:
        return pd.DataFrame(columns=cols)
    sub = provenance[(provenance["brand_name"] == brand_name) & (provenance["outcome"] == outcome_value)]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    grp = sub.groupby("claim_type").size().rename("claims").reset_index()
    return grp.sort_values("claims", ascending=False).reset_index(drop=True)


def claims_citation_support(provenance: pd.DataFrame, brand_name: Optional[str] = None) -> pd.DataFrame:
    """Per claim type, how many claim instances appear with vs without a citation.

    Columns: ``claim_type``, ``with_citation``, ``without_citation``, ``supported_share``.
    "Supported" here means a citation appeared in the same response — association, not proof.
    """
    cols = ["claim_type", "with_citation", "without_citation", "supported_share"]
    if provenance.empty or "has_citation" not in provenance.columns:
        return pd.DataFrame(columns=cols)
    sub = provenance if brand_name is None else provenance[provenance["brand_name"] == brand_name]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for ct, grp in sub.groupby("claim_type"):
        withc = int(grp["has_citation"].astype(bool).sum())
        total = len(grp)
        rows.append({"claim_type": ct, "with_citation": withc, "without_citation": total - withc,
                     "supported_share": withc / total if total else 0.0})
    return pd.DataFrame(rows, columns=cols).sort_values("supported_share").reset_index(drop=True)


# Antonym cues that make a same-type claim internally contradictory.
_CLAIM_CONTRADICTIONS: dict[str, tuple[list[str], list[str]]] = {
    "Pricing claim": (["affordable", "free plan", "budget"], ["expensive", "pricey"]),
    "Ease of use claim": (["easy to use", "easy to learn", "intuitive", "simple", "approachable"], []),
}
_EASE_NEGATIVE = ["steep learning curve", "overwhelming", "cluttered", "complex"]


def conflicting_claims(claims: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    """Detect brands described with contradictory claims across responses.

    Columns: ``dimension``, ``positive``, ``negative``, ``positive_count``, ``negative_count``.
    (e.g. a brand called both 'affordable' and 'expensive' across different responses.)
    """
    cols = ["dimension", "positive", "negative", "positive_count", "negative_count"]
    if claims.empty:
        return pd.DataFrame(columns=cols)
    sub = claims[claims["brand_name"] == brand_name]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    text_all = " ; ".join((sub["claim_text"].fillna("") + " " + sub["evidence_text"].fillna("")).tolist()).lower()

    rows = []
    # Pricing: affordable/free vs expensive/pricey.
    pos_terms, neg_terms = _CLAIM_CONTRADICTIONS["Pricing claim"]
    pos = sum(text_all.count(t) for t in pos_terms)
    neg = sum(text_all.count(t) for t in neg_terms)
    if pos and neg:
        rows.append({"dimension": "Pricing", "positive": "affordable/free", "negative": "expensive/pricey",
                     "positive_count": pos, "negative_count": neg})
    # Ease of use: easy/intuitive vs steep/complex.
    ease_pos = sum(text_all.count(t) for t in _CLAIM_CONTRADICTIONS["Ease of use claim"][0])
    ease_neg = sum(text_all.count(t) for t in _EASE_NEGATIVE)
    if ease_pos and ease_neg:
        rows.append({"dimension": "Ease of use", "positive": "easy/intuitive", "negative": "steep/complex",
                     "positive_count": ease_pos, "negative_count": ease_neg})
    return pd.DataFrame(rows, columns=cols)


def claims_by_platform(claims: pd.DataFrame, response_runs: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    """Claim-type counts per platform for a brand (claims that differ by platform).

    Columns: ``platform``, ``claim_type``, ``claims``.
    """
    cols = ["platform", "claim_type", "claims"]
    if claims.empty or response_runs.empty:
        return pd.DataFrame(columns=cols)
    merged = claims[claims["brand_name"] == brand_name].merge(
        response_runs[["run_id", "platform"]], on="run_id", how="left")
    if merged.empty:
        return pd.DataFrame(columns=cols)
    grp = merged.groupby(["platform", "claim_type"]).size().rename("claims").reset_index()
    return grp.sort_values(["platform", "claims"], ascending=[True, False]).reset_index(drop=True)


def claims_over_time(claims: pd.DataFrame, response_runs: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    """Claim-type counts per collection date for a brand (claims that change over time).

    Columns: ``run_date``, ``claim_type``, ``claims``.
    """
    cols = ["run_date", "claim_type", "claims"]
    if claims.empty or response_runs.empty:
        return pd.DataFrame(columns=cols)
    merged = claims[claims["brand_name"] == brand_name].merge(
        response_runs[["run_id", "run_date"]], on="run_id", how="left")
    if merged.empty:
        return pd.DataFrame(columns=cols)
    grp = merged.groupby(["run_date", "claim_type"]).size().rename("claims").reset_index()
    return grp.sort_values(["run_date", "claims"], ascending=[True, False]).reset_index(drop=True)
