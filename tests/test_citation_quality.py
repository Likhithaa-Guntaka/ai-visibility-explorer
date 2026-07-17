"""Tests for citation source classification and quality metrics."""

from __future__ import annotations

import math

import pandas as pd

from src import citation_quality as CQ


BRANDS = pd.DataFrame([
    {"brand_id": "b1", "project_id": "p", "brand_name": "Notion", "brand_domain": "notion.so"},
    {"brand_id": "b2", "project_id": "p", "brand_name": "Asana", "brand_domain": "asana.com"},
])


def test_classify_domain_precedence():
    focal = {"notion.so"}
    comp = {"asana.com"}
    assert CQ.classify_domain("notion.so", focal, comp) == "Brand owned"
    assert CQ.classify_domain("asana.com", focal, comp) == "Competitor owned"
    assert CQ.classify_domain("g2.com", focal, comp) == "Review site"
    assert CQ.classify_domain("reddit.com", focal, comp) == "Forum or community"
    assert CQ.classify_domain("nytimes.com", focal, comp) == "News or media"
    assert CQ.classify_domain("docs.example.com", focal, comp) == "Documentation"
    assert CQ.classify_domain("twitter.com", focal, comp) == "Social platform"
    assert CQ.classify_domain("random-blog.io", focal, comp) == "Other third party"


def test_subdomain_matches_brand_owned():
    assert CQ.classify_domain("help.notion.so", {"notion.so"}, set()) == "Brand owned"


def _citations():
    return pd.DataFrame([
        {"run_id": "r1", "citation_url": "https://notion.so/a", "citation_domain": "notion.so", "citation_position": 1},
        {"run_id": "r1", "citation_url": "https://g2.com/x", "citation_domain": "g2.com", "citation_position": 2},
        {"run_id": "r2", "citation_url": "https://asana.com/b", "citation_domain": "asana.com", "citation_position": 1},
        {"run_id": "r2", "citation_url": "https://g2.com/x", "citation_domain": "g2.com", "citation_position": 2},
    ])


def test_classify_citations_adds_source_type():
    cls = CQ.classify_citations(_citations(), BRANDS, "Notion")
    types = dict(zip(cls["citation_domain"], cls["source_type"]))
    assert types["notion.so"] == "Brand owned"
    assert types["asana.com"] == "Competitor owned"
    assert types["g2.com"] == "Review site"


def test_diversity():
    d = CQ.citation_diversity(_citations())
    assert d["unique_domains"] == 3
    assert d["total_citations"] == 4
    assert math.isclose(d["diversity"], 3 / 4)


def test_concentration():
    c = CQ.citation_concentration(_citations())
    # g2.com appears twice out of four -> top1_share = 0.5
    assert math.isclose(c["top1_share"], 0.5)
    assert 0 < c["hhi"] <= 1


def test_source_type_breakdown_shares_sum_to_one():
    cls = CQ.classify_citations(_citations(), BRANDS, "Notion")
    stb = CQ.source_type_breakdown(cls)
    assert math.isclose(stb["share"].sum(), 1.0)


def test_brand_owned_vs_third_party():
    cls = CQ.classify_citations(_citations(), BRANDS, "Notion")
    res = CQ.brand_owned_vs_third_party(cls)
    assert res["brand_owned"] == 1       # notion.so
    assert res["competitor_owned"] == 1  # asana.com
    assert res["third_party"] == 2       # two g2.com


def test_citation_opportunities_flags_competitor_cooccurrence():
    citations = _citations()
    mentions = pd.DataFrame([
        {"run_id": "r1", "brand_name": "Notion", "mention_count": 1, "first_mention_position": 0, "is_recommended": True},
        {"run_id": "r2", "brand_name": "Asana", "mention_count": 1, "first_mention_position": 0, "is_recommended": True},
    ])
    # Focal = Notion. g2.com is cited in r1 (Notion) and r2 (Asana competitor).
    opps = CQ.citation_opportunities(citations, mentions, BRANDS, "Notion")
    g2 = opps[opps["citation_domain"] == "g2.com"].iloc[0]
    assert g2["runs_with_competitor"] == 1
    assert g2["runs_with_focal"] == 1
    assert g2["opportunity_gap"] == 0


def test_empty_citations_safe():
    empty = pd.DataFrame(columns=["run_id", "citation_url", "citation_domain", "citation_position"])
    assert CQ.citation_diversity(empty)["unique_domains"] == 0
    assert CQ.source_type_breakdown(CQ.classify_citations(empty, BRANDS, "Notion")).empty
