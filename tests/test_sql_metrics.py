"""DuckDB SQL metrics must equal the pandas reference implementations, value-for-value.

These tests are the contract that lets the app compute headline metrics in SQL while
keeping src/metrics.py as the definition of record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import appkit
from src import metrics as M
from src import sql_metrics as S
from src.database import AnalysisData


def _assert_frames_equal(pandas_df: pd.DataFrame, sql_df: pd.DataFrame, key: str, value_cols: list[str]):
    """Compare two metric frames after sorting by ``key`` (ignores row order)."""
    a = pandas_df.sort_values(key).reset_index(drop=True)
    b = sql_df.sort_values(key).reset_index(drop=True)
    assert list(a[key].astype(str)) == list(b[key].astype(str)), f"key mismatch on {key}"
    for c in value_cols:
        assert np.allclose(a[c].astype(float), b[c].astype(float)), f"value mismatch in {c}"


# -- fixtures (mirror tests/test_metrics.py so expected values are hand-checked) -----


@pytest.fixture
def response_runs():
    return pd.DataFrame([
        {"run_id": "r1", "prompt_id": "p1", "platform": "ChatGPT", "run_date": "2026-06-05", "run_number": 1},
        {"run_id": "r2", "prompt_id": "p1", "platform": "ChatGPT", "run_date": "2026-06-05", "run_number": 2},
        {"run_id": "r3", "prompt_id": "p2", "platform": "Claude", "run_date": "2026-06-05", "run_number": 1},
        {"run_id": "r4", "prompt_id": "p3", "platform": "Claude", "run_date": "2026-06-05", "run_number": 1},
    ])


@pytest.fixture
def prompts():
    return pd.DataFrame([
        {"prompt_id": "p1", "prompt_category": "Nonbrand discovery", "topic": "Best tools", "persona": "Team Lead", "journey_stage": "Consideration"},
        {"prompt_id": "p2", "prompt_category": "Product comparison", "topic": "Notion vs Asana", "persona": "Team Lead", "journey_stage": "Consideration"},
        {"prompt_id": "p3", "prompt_category": "Purchase intent", "topic": "Pricing", "persona": "Founder", "journey_stage": "Decision"},
    ])


@pytest.fixture
def brand_mentions():
    return pd.DataFrame([
        {"run_id": "r1", "brand_name": "Notion", "mention_count": 2, "first_mention_position": 0, "is_recommended": True},
        {"run_id": "r1", "brand_name": "Asana", "mention_count": 1, "first_mention_position": 20, "is_recommended": False},
        {"run_id": "r2", "brand_name": "Notion", "mention_count": 1, "first_mention_position": 5, "is_recommended": True},
        {"run_id": "r3", "brand_name": "Notion", "mention_count": 1, "first_mention_position": 10, "is_recommended": False},
        {"run_id": "r3", "brand_name": "Asana", "mention_count": 2, "first_mention_position": 0, "is_recommended": True},
        {"run_id": "r4", "brand_name": "Trello", "mention_count": 1, "first_mention_position": 0, "is_recommended": False},
    ])


@pytest.fixture
def citations():
    return pd.DataFrame([
        {"run_id": "r1", "citation_url": "https://g2.com/a", "citation_domain": "g2.com", "citation_position": 1},
        {"run_id": "r1", "citation_url": "https://notion.so/x", "citation_domain": "notion.so", "citation_position": 2},
        {"run_id": "r2", "citation_url": "https://g2.com/a", "citation_domain": "g2.com", "citation_position": 1},
        {"run_id": "r3", "citation_url": "https://capterra.com/y", "citation_domain": "capterra.com", "citation_position": 1},
    ])


# -- equivalence on the hand-checked fixture ----------------------------------------


def test_sql_brand_mention_rate_matches(brand_mentions, response_runs):
    _assert_frames_equal(
        M.brand_mention_rate(brand_mentions, response_runs),
        S.brand_mention_rate(brand_mentions, response_runs),
        "brand_name", ["mentioned_runs", "total_runs", "mention_rate"],
    )


def test_sql_share_of_voice_matches(brand_mentions):
    _assert_frames_equal(
        M.share_of_voice(brand_mentions), S.share_of_voice(brand_mentions),
        "brand_name", ["mentions", "share_of_voice"],
    )


def test_sql_first_mention_share_matches(brand_mentions):
    _assert_frames_equal(
        M.first_mention_share(brand_mentions), S.first_mention_share(brand_mentions),
        "brand_name", ["first_mentions", "mentioned_runs", "first_mention_share"],
    )


def test_sql_recommendation_rate_matches(brand_mentions, response_runs):
    _assert_frames_equal(
        M.recommendation_rate(brand_mentions, response_runs),
        S.recommendation_rate(brand_mentions, response_runs),
        "brand_name", ["recommended_runs", "total_runs", "recommendation_rate"],
    )


def test_sql_citation_rate_matches(citations, response_runs):
    assert M.citation_rate(citations, response_runs) == S.citation_rate(citations, response_runs)


def test_sql_source_domain_share_matches(citations):
    _assert_frames_equal(
        M.source_domain_share(citations), S.source_domain_share(citations),
        "citation_domain", ["citations", "runs", "domain_share"],
    )


def test_sql_competitor_visibility_matches(brand_mentions, response_runs):
    _assert_frames_equal(
        M.competitor_visibility(brand_mentions, response_runs),
        S.competitor_visibility(brand_mentions, response_runs),
        "brand_name", ["mention_rate", "share_of_voice", "recommendation_rate"],
    )


@pytest.mark.parametrize("attribute", ["prompt_category", "persona"])
def test_sql_visibility_by_attribute_matches(attribute, brand_mentions, response_runs, prompts):
    enriched = M.enrich_mentions(brand_mentions, response_runs, prompts)
    _assert_frames_equal(
        M.visibility_by_attribute(enriched, response_runs, prompts, attribute, "Notion"),
        S.visibility_by_attribute(brand_mentions, response_runs, prompts, attribute, "Notion"),
        attribute, ["mentioned_runs", "total_runs", "mention_rate"],
    )


def test_sql_platform_comparison_matches(brand_mentions, response_runs, prompts):
    enriched = M.enrich_mentions(brand_mentions, response_runs, prompts)
    _assert_frames_equal(
        M.platform_comparison(enriched, response_runs, "Notion"),
        S.platform_comparison(brand_mentions, response_runs, "Notion"),
        "platform", ["mentioned_runs", "total_runs", "mention_rate"],
    )


# -- equivalence on the full synthetic demo dataset ---------------------------------


def test_sql_equals_pandas_on_full_demo():
    data = appkit.load_demo_analysis()
    with S.SqlMetrics(data) as m:
        _assert_frames_equal(M.brand_mention_rate(data.brand_mentions, data.response_runs),
                             m.brand_mention_rate(), "brand_name", ["mentioned_runs", "mention_rate"])
        _assert_frames_equal(M.share_of_voice(data.brand_mentions), m.share_of_voice(),
                             "brand_name", ["mentions", "share_of_voice"])
        _assert_frames_equal(M.competitor_visibility(data.brand_mentions, data.response_runs),
                             m.competitor_visibility(), "brand_name",
                             ["mention_rate", "share_of_voice", "recommendation_rate"])
        assert M.citation_rate(data.citations, data.response_runs) == m.citation_rate()


# -- edge cases ---------------------------------------------------------------------


def test_sql_empty_inputs_return_empty():
    empty = AnalysisData()
    with S.SqlMetrics(empty) as m:
        assert m.brand_mention_rate().empty
        assert m.share_of_voice().empty
        assert m.first_mention_share().empty
        assert m.competitor_visibility().empty
        assert m.source_domain_share().empty
        assert m.citation_rate()["citation_rate"] == 0.0


def test_sql_filtering_is_respected():
    """SQL metrics run over whatever AnalysisData is passed, so app-level filtering
    (which produces a scoped AnalysisData) flows through unchanged."""
    data = appkit.load_demo_analysis()
    scoped = appkit.filter_data(data, question_clusters=["Pricing & value"])
    with S.SqlMetrics(scoped) as m:
        sql_rate = m.citation_rate()
    pandas_rate = M.citation_rate(scoped.citations, scoped.response_runs)
    assert sql_rate == pandas_rate
    # And the scoped total is smaller than the whole dataset.
    assert sql_rate["total_runs"] < len(data.response_runs)


def test_visibility_by_attribute_rejects_unknown_column():
    data = appkit.load_demo_analysis()
    with S.SqlMetrics(data) as m:
        with pytest.raises(ValueError):
            m.visibility_by_attribute("response_text", "Notion")  # not allow-listed


def test_top_n_limits_domain_rows():
    data = appkit.load_demo_analysis()
    with S.SqlMetrics(data) as m:
        assert len(m.source_domain_share(top_n=3)) == 3
