"""Tests for the visibility metrics.

Uses a small, hand-checked fixture so every expected number is obvious by inspection.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src import metrics as M


@pytest.fixture
def response_runs():
    # 4 runs across 3 prompts. Prompt p1 has two repeated runs (r1, r2).
    return pd.DataFrame(
        [
            {"run_id": "r1", "prompt_id": "p1", "platform": "ChatGPT", "run_date": "2026-07-10", "run_number": 1},
            {"run_id": "r2", "prompt_id": "p1", "platform": "ChatGPT", "run_date": "2026-07-10", "run_number": 2},
            {"run_id": "r3", "prompt_id": "p2", "platform": "Claude", "run_date": "2026-07-10", "run_number": 1},
            {"run_id": "r4", "prompt_id": "p3", "platform": "Claude", "run_date": "2026-07-10", "run_number": 1},
        ]
    )


@pytest.fixture
def prompts():
    return pd.DataFrame(
        [
            {"prompt_id": "p1", "prompt_category": "Nonbrand discovery", "topic": "Best tools", "persona": "Team Lead", "journey_stage": "Consideration"},
            {"prompt_id": "p2", "prompt_category": "Product comparison", "topic": "Notion vs Asana", "persona": "Team Lead", "journey_stage": "Consideration"},
            {"prompt_id": "p3", "prompt_category": "Purchase intent", "topic": "Pricing", "persona": "Founder", "journey_stage": "Decision"},
        ]
    )


@pytest.fixture
def brand_mentions():
    # Notion mentioned in r1, r2, r3 (3 of 4 runs); Asana in r1, r3; Trello in r4.
    return pd.DataFrame(
        [
            {"run_id": "r1", "brand_name": "Notion", "mention_count": 2, "first_mention_position": 0, "is_recommended": True},
            {"run_id": "r1", "brand_name": "Asana", "mention_count": 1, "first_mention_position": 20, "is_recommended": False},
            {"run_id": "r2", "brand_name": "Notion", "mention_count": 1, "first_mention_position": 5, "is_recommended": True},
            {"run_id": "r3", "brand_name": "Notion", "mention_count": 1, "first_mention_position": 10, "is_recommended": False},
            {"run_id": "r3", "brand_name": "Asana", "mention_count": 2, "first_mention_position": 0, "is_recommended": True},
            {"run_id": "r4", "brand_name": "Trello", "mention_count": 1, "first_mention_position": 0, "is_recommended": False},
        ]
    )


@pytest.fixture
def citations():
    return pd.DataFrame(
        [
            {"run_id": "r1", "citation_url": "https://g2.com/a", "citation_domain": "g2.com", "citation_position": 1},
            {"run_id": "r1", "citation_url": "https://notion.so/x", "citation_domain": "notion.so", "citation_position": 2},
            {"run_id": "r2", "citation_url": "https://g2.com/a", "citation_domain": "g2.com", "citation_position": 1},
            {"run_id": "r3", "citation_url": "https://capterra.com/y", "citation_domain": "capterra.com", "citation_position": 1},
        ]
    )


def test_total_runs(response_runs):
    assert M.total_runs(response_runs) == 4


def test_brand_mention_rate(brand_mentions, response_runs):
    df = M.brand_mention_rate(brand_mentions, response_runs)
    rates = dict(zip(df["brand_name"], df["mention_rate"]))
    assert rates["Notion"] == 0.75  # r1, r2, r3
    assert rates["Asana"] == 0.5  # r1, r3
    assert rates["Trello"] == 0.25  # r4


def test_share_of_voice_sums_to_one(brand_mentions):
    df = M.share_of_voice(brand_mentions)
    assert math.isclose(df["share_of_voice"].sum(), 1.0, rel_tol=1e-9)
    # Total mentions: Notion 4, Asana 3, Trello 1 => 8 total.
    sov = dict(zip(df["brand_name"], df["share_of_voice"]))
    assert math.isclose(sov["Notion"], 4 / 8)
    assert math.isclose(sov["Asana"], 3 / 8)


def test_first_mention_share(brand_mentions):
    df = M.first_mention_share(brand_mentions)
    shares = dict(zip(df["brand_name"], df["first_mention_share"]))
    # r1 first: Notion(0). r2 first: Notion(5). r3 first: Asana(0). r4 first: Trello(0).
    # 4 runs mention a brand. Notion first in 2 => 0.5; Asana 1 => 0.25; Trello 1 => 0.25.
    assert shares["Notion"] == 0.5
    assert shares["Asana"] == 0.25
    assert shares["Trello"] == 0.25


def test_recommendation_rate(brand_mentions, response_runs):
    df = M.recommendation_rate(brand_mentions, response_runs)
    rates = dict(zip(df["brand_name"], df["recommendation_rate"]))
    # Notion recommended in r1, r2 => 2/4 = 0.5; Asana in r3 => 0.25.
    assert rates["Notion"] == 0.5
    assert rates["Asana"] == 0.25


def test_citation_rate(citations, response_runs):
    res = M.citation_rate(citations, response_runs)
    # r1, r2, r3 have citations => 3/4.
    assert res["runs_with_citations"] == 3
    assert res["citation_rate"] == 0.75


def test_source_domain_share(citations):
    df = M.source_domain_share(citations)
    row = df[df["citation_domain"] == "g2.com"].iloc[0]
    # g2.com cited twice out of 4 total citations.
    assert row["citations"] == 2
    assert math.isclose(row["domain_share"], 2 / 4)
    assert row["runs"] == 2


def test_visibility_by_category(brand_mentions, response_runs, prompts):
    enriched = M.enrich_mentions(brand_mentions, response_runs, prompts)
    df = M.visibility_by_attribute(enriched, response_runs, prompts, "prompt_category", "Notion")
    rates = dict(zip(df["prompt_category"], df["mention_rate"]))
    # Nonbrand discovery = p1 (r1, r2): Notion in both => 2/2 = 1.0
    assert rates["Nonbrand discovery"] == 1.0
    # Purchase intent = p3 (r4): Notion absent => 0.0
    assert rates["Purchase intent"] == 0.0


def test_platform_comparison(brand_mentions, response_runs, prompts):
    enriched = M.enrich_mentions(brand_mentions, response_runs, prompts)
    df = M.platform_comparison(enriched, response_runs, "Notion")
    rates = dict(zip(df["platform"], df["mention_rate"]))
    # ChatGPT = r1, r2 (both Notion) => 1.0; Claude = r3 (Notion), r4 (no) => 0.5.
    assert rates["ChatGPT"] == 1.0
    assert rates["Claude"] == 0.5


def test_competitor_visibility_ordered_by_sov(brand_mentions, response_runs):
    df = M.competitor_visibility(brand_mentions, response_runs)
    assert list(df["brand_name"])[0] == "Notion"  # highest SoV


def test_content_coverage_gaps_flags_competitor_lead(brand_mentions, response_runs, prompts):
    enriched = M.enrich_mentions(brand_mentions, response_runs, prompts)
    # Focal = Trello, which only appears on the Pricing topic; competitors dominate elsewhere.
    df = M.content_coverage_gaps(enriched, response_runs, prompts, "Trello", dimension="topic")
    best_tools = df[df["topic"] == "Best tools"].iloc[0]
    assert best_tools["focal_rate"] == 0.0
    assert best_tools["gap"] > 0  # a competitor leads Trello here


def test_response_consistency_perfect_and_summary(response_runs, citations):
    # Build mentions where p1's two runs (r1, r2) mention exactly the same brand set.
    mentions = pd.DataFrame(
        [
            {"run_id": "r1", "brand_name": "Notion", "mention_count": 2, "first_mention_position": 0, "is_recommended": True},
            {"run_id": "r1", "brand_name": "Asana", "mention_count": 1, "first_mention_position": 5, "is_recommended": False},
            {"run_id": "r2", "brand_name": "Notion", "mention_count": 2, "first_mention_position": 0, "is_recommended": True},
            {"run_id": "r2", "brand_name": "Asana", "mention_count": 1, "first_mention_position": 5, "is_recommended": False},
        ]
    )
    df = M.response_consistency(mentions, citations, response_runs)
    # Only p1 has repeated runs.
    assert list(df["prompt_id"]) == ["p1"]
    row = df.iloc[0]
    assert row["brand_overlap"] == 1.0  # identical brand sets
    assert row["mention_count_variation"] == 0.0  # identical totals
    summary = M.consistency_summary(df)
    assert summary["prompts_with_repeats"] == 1
    assert summary["avg_brand_overlap"] == 1.0


def test_is_small_sample():
    assert M.is_small_sample(3) is True
    assert M.is_small_sample(10) is False


def test_metrics_handle_empty_inputs():
    empty_runs = pd.DataFrame(columns=["run_id", "prompt_id", "platform", "run_date", "run_number"])
    empty_mentions = pd.DataFrame(
        columns=["run_id", "brand_name", "mention_count", "first_mention_position", "is_recommended"]
    )
    empty_citations = pd.DataFrame(columns=["run_id", "citation_url", "citation_domain", "citation_position"])
    assert M.brand_mention_rate(empty_mentions, empty_runs).empty
    assert M.share_of_voice(empty_mentions).empty
    assert M.citation_rate(empty_citations, empty_runs)["citation_rate"] == 0.0
    assert M.consistency_summary(M.response_consistency(empty_mentions, empty_citations, empty_runs))[
        "prompts_with_repeats"
    ] == 0
