"""Citation quality & opportunity analysis (feature: Citation Quality and Opportunities).

Classifies each cited source into a transparent source *type*, then computes diversity,
concentration, brand-owned vs third-party share, top source types, and — grounded in
run-level co-occurrence — which third-party domains show up alongside competitors but
not the focal brand (a citation *opportunity*).

Responsible use: we describe associations only. We never claim that a source *caused*
a brand's visibility or an AI recommendation.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Source-type lexicons matched against a normalized domain (www stripped, lowercased).
# Matching is by exact domain, suffix, or substring as noted per set.
REVIEW_SITES = {"g2.com", "capterra.com", "trustradius.com", "softwareadvice.com", "getapp.com"}
NEWS_MEDIA = {
    "nytimes.com", "forbes.com", "techcrunch.com", "theverge.com", "wired.com",
    "businessinsider.com", "pcmag.com", "zdnet.com", "cnbc.com", "cnn.com",
}
FORUM_COMMUNITY = {"reddit.com", "quora.com", "stackexchange.com", "stackoverflow.com", "news.ycombinator.com"}
SOCIAL_PLATFORMS = {"twitter.com", "x.com", "linkedin.com", "facebook.com", "youtube.com", "instagram.com", "tiktok.com"}
# Documentation is detected structurally (subdomain / path-like), see _is_docs.
DOC_SUBDOMAINS = ("docs.", "developer.", "support.", "help.")

SOURCE_TYPES = [
    "Brand owned",
    "Competitor owned",
    "Review site",
    "Forum or community",
    "News or media",
    "Documentation",
    "Social platform",
    "Other third party",
]


def _is_docs(domain: str) -> bool:
    return any(domain.startswith(pfx) for pfx in DOC_SUBDOMAINS)


def _matches(domain: str, domain_set: set[str]) -> bool:
    """True if domain equals or is a subdomain of any domain in ``domain_set``."""
    return any(domain == d or domain.endswith("." + d) for d in domain_set)


def classify_domain(
    domain: str,
    focal_domains: set[str],
    competitor_domains: set[str],
) -> str:
    """Return the source type of a domain relative to the focal brand.

    Precedence: brand owned > competitor owned > review > forum > news > docs > social
    > other third party.
    """
    if not domain:
        return "Other third party"
    if _matches(domain, focal_domains):
        return "Brand owned"
    if _matches(domain, competitor_domains):
        return "Competitor owned"
    if _matches(domain, REVIEW_SITES):
        return "Review site"
    if _matches(domain, FORUM_COMMUNITY):
        return "Forum or community"
    if _matches(domain, NEWS_MEDIA):
        return "News or media"
    if _is_docs(domain):
        return "Documentation"
    if _matches(domain, SOCIAL_PLATFORMS):
        return "Social platform"
    return "Other third party"


def _brand_domain_sets(brands_df: pd.DataFrame, focal_brand: str) -> tuple[set[str], set[str]]:
    """Return (focal_domains, competitor_domains) from the brands table."""
    focal, comp = set(), set()
    if brands_df.empty:
        return focal, comp
    for _, row in brands_df.iterrows():
        dom = str(row.get("brand_domain", "") or "").strip().lower()
        if not dom:
            continue
        if dom.startswith("www."):
            dom = dom[4:]
        (focal if row["brand_name"] == focal_brand else comp).add(dom)
    return focal, comp


def classify_citations(citations: pd.DataFrame, brands_df: pd.DataFrame, focal_brand: str) -> pd.DataFrame:
    """Add a ``source_type`` column to the citations frame (relative to focal brand)."""
    if citations.empty:
        return citations.assign(source_type=pd.Series(dtype=str))
    focal_domains, competitor_domains = _brand_domain_sets(brands_df, focal_brand)
    out = citations.copy()
    out["source_type"] = out["citation_domain"].map(
        lambda d: classify_domain(str(d), focal_domains, competitor_domains)
    )
    return out


# ---------------------------------------------------------------------------
# Quality metrics.
# ---------------------------------------------------------------------------


def citation_diversity(citations: pd.DataFrame) -> dict[str, float]:
    """Diversity of cited sources.

    Returns ``unique_domains``, ``total_citations``, ``diversity`` (unique/total, where
    1.0 means every citation is a different domain), and ``shannon_evenness`` (0-1, how
    evenly citations are spread across domains).
    """
    import math

    if citations.empty:
        return {"unique_domains": 0, "total_citations": 0, "diversity": 0.0, "shannon_evenness": 0.0}
    total = len(citations)
    counts = citations["citation_domain"].value_counts()
    unique = int(len(counts))
    shares = counts / total
    entropy = float(-(shares * shares.apply(math.log)).sum())
    max_entropy = math.log(unique) if unique > 1 else 0.0
    evenness = (entropy / max_entropy) if max_entropy > 0 else (1.0 if unique == 1 else 0.0)
    return {
        "unique_domains": unique,
        "total_citations": total,
        "diversity": unique / total,
        "shannon_evenness": evenness,
    }


def citation_concentration(citations: pd.DataFrame) -> dict[str, float]:
    """Concentration of citations in a few domains.

    Returns ``hhi`` (Herfindahl index, sum of squared domain shares; 1.0 = one domain
    owns everything), ``top1_share`` and ``top3_share``.
    """
    if citations.empty:
        return {"hhi": 0.0, "top1_share": 0.0, "top3_share": 0.0}
    total = len(citations)
    shares = (citations["citation_domain"].value_counts() / total).sort_values(ascending=False)
    hhi = float((shares ** 2).sum())
    return {
        "hhi": hhi,
        "top1_share": float(shares.iloc[0]),
        "top3_share": float(shares.iloc[:3].sum()),
    }


def source_type_breakdown(classified: pd.DataFrame) -> pd.DataFrame:
    """Counts and share of citations by source type.

    Returns columns: ``source_type``, ``citations``, ``share``.
    """
    cols = ["source_type", "citations", "share"]
    if classified.empty or "source_type" not in classified.columns:
        return pd.DataFrame(columns=cols)
    grp = classified.groupby("source_type").size().rename("citations").reset_index()
    total = grp["citations"].sum()
    grp["share"] = grp["citations"] / total if total else 0.0
    return grp.sort_values("citations", ascending=False).reset_index(drop=True)


def brand_owned_vs_third_party(classified: pd.DataFrame) -> dict[str, float]:
    """Share of citations that are brand-owned vs everything else (third party).

    Returns ``brand_owned``, ``competitor_owned``, ``third_party`` counts and their shares.
    """
    if classified.empty or "source_type" not in classified.columns:
        return {"brand_owned": 0, "competitor_owned": 0, "third_party": 0,
                "brand_owned_share": 0.0, "competitor_owned_share": 0.0, "third_party_share": 0.0}
    total = len(classified)
    brand_owned = int((classified["source_type"] == "Brand owned").sum())
    competitor_owned = int((classified["source_type"] == "Competitor owned").sum())
    third_party = total - brand_owned - competitor_owned
    return {
        "brand_owned": brand_owned,
        "competitor_owned": competitor_owned,
        "third_party": third_party,
        "brand_owned_share": brand_owned / total if total else 0.0,
        "competitor_owned_share": competitor_owned / total if total else 0.0,
        "third_party_share": third_party / total if total else 0.0,
    }


def citation_opportunities(
    citations: pd.DataFrame,
    brand_mentions: pd.DataFrame,
    brands_df: pd.DataFrame,
    focal_brand: str,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Third-party domains cited alongside competitors more than the focal brand.

    For each third-party cited domain we count the responses citing it that mention the
    focal brand vs at least one competitor. Domains with high competitor co-occurrence
    and low focal co-occurrence are *opportunities worth investigating* — places where
    competitors show up in cited sources but you do not. This is association, not proof.

    Returns columns: ``citation_domain``, ``source_type``, ``runs_citing``,
    ``runs_with_focal``, ``runs_with_competitor``, ``opportunity_gap``.
    """
    cols = ["citation_domain", "source_type", "runs_citing", "runs_with_focal",
            "runs_with_competitor", "opportunity_gap"]
    if citations.empty:
        return pd.DataFrame(columns=cols)

    classified = classify_citations(citations, brands_df, focal_brand)
    third_party = classified[~classified["source_type"].isin(["Brand owned"])]
    if third_party.empty:
        return pd.DataFrame(columns=cols)

    # Per-run focal / competitor mention flags.
    focal_runs = set(brand_mentions[brand_mentions["brand_name"] == focal_brand]["run_id"]) if not brand_mentions.empty else set()
    competitor_runs = set(brand_mentions[brand_mentions["brand_name"] != focal_brand]["run_id"]) if not brand_mentions.empty else set()

    rows = []
    for domain, grp in third_party.groupby("citation_domain"):
        runs = set(grp["run_id"])
        with_focal = len(runs & focal_runs)
        with_comp = len(runs & competitor_runs)
        rows.append(
            {
                "citation_domain": domain,
                "source_type": grp["source_type"].iloc[0],
                "runs_citing": len(runs),
                "runs_with_focal": with_focal,
                "runs_with_competitor": with_comp,
                "opportunity_gap": with_comp - with_focal,
            }
        )
    out = pd.DataFrame(rows, columns=cols).sort_values(
        ["opportunity_gap", "runs_citing"], ascending=False
    ).reset_index(drop=True)
    return out.head(top_n) if top_n else out
