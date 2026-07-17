"""Tests for deterministic brand + citation extraction."""

from __future__ import annotations

import pandas as pd

from src.extraction import (
    DeterministicExtractor,
    build_alias_map,
    extract_all,
    normalize_domain,
)


def _brands_df():
    return pd.DataFrame(
        [
            {"brand_id": "b1", "project_id": "p", "brand_name": "Notion", "brand_domain": "notion.so"},
            {"brand_id": "b2", "project_id": "p", "brand_name": "Monday.com", "brand_domain": "monday.com"},
            {"brand_id": "b3", "project_id": "p", "brand_name": "Trello", "brand_domain": "trello.com"},
        ]
    )


def test_normalize_domain_strips_www_port_and_lowercases():
    assert normalize_domain("https://www.G2.com:443/best") == "g2.com"
    assert normalize_domain("http://Blog.Example.com/path") == "blog.example.com"
    assert normalize_domain("not a url") == ""


def test_case_insensitive_matching_and_counts():
    ex = DeterministicExtractor()
    text = "notion is great. I love Notion. NOTION rocks."
    result = ex.extract_run("r1", text, {"Notion": []})
    assert len(result.mentions) == 1
    assert result.mentions[0]["mention_count"] == 3
    assert result.mentions[0]["brand_name"] == "Notion"


def test_alias_matching_for_monday():
    ex = DeterministicExtractor()
    text = "We recommend Monday for teams that like colorful boards."
    result = ex.extract_run("r1", text, {"Monday.com": ["Monday"]})
    assert len(result.mentions) == 1
    assert result.mentions[0]["mention_count"] == 1


def test_word_boundaries_prevent_substring_false_positives():
    ex = DeterministicExtractor()
    # "Trellos" should not match the brand "Trello".
    text = "The Trellos family is unrelated to any board tool."
    result = ex.extract_run("r1", text, {"Trello": []})
    assert result.mentions == []


def test_first_mention_position_identifies_earliest_brand():
    ex = DeterministicExtractor()
    text = "Asana leads, then Notion, then Trello."
    result = ex.extract_run("r1", text, {"Notion": [], "Asana": [], "Trello": []})
    positions = {m["brand_name"]: m["first_mention_position"] for m in result.mentions}
    assert positions["Asana"] < positions["Notion"] < positions["Trello"]


def test_recommendation_heuristic_detects_cue_word():
    ex = DeterministicExtractor()
    text = "For most teams, we recommend Notion as the best starting point."
    result = ex.extract_run("r1", text, {"Notion": []})
    assert result.mentions[0]["is_recommended"] is True


def test_recommendation_heuristic_negative_when_no_cue():
    ex = DeterministicExtractor()
    text = "Notion exists. Some people have heard of it."
    result = ex.extract_run("r1", text, {"Notion": []})
    assert result.mentions[0]["is_recommended"] is False


def test_url_extraction_normalizes_domains_and_positions():
    ex = DeterministicExtractor()
    text = "Sources: https://www.g2.com/x and https://clickup.com/features."
    result = ex.extract_run("r1", text, {})
    domains = [c["citation_domain"] for c in result.citations]
    positions = [c["citation_position"] for c in result.citations]
    assert domains == ["g2.com", "clickup.com"]
    assert positions == [1, 2]


def test_url_trailing_punctuation_is_trimmed():
    ex = DeterministicExtractor()
    text = "See https://example.com/page."
    result = ex.extract_run("r1", text, {})
    assert result.citations[0]["citation_url"] == "https://example.com/page"


def test_duplicate_urls_deduped_within_run():
    ex = DeterministicExtractor()
    text = "https://a.com/x and again https://a.com/x"
    result = ex.extract_run("r1", text, {})
    assert len(result.citations) == 1


def test_build_alias_map_adds_monday_default_alias():
    alias_map = build_alias_map(_brands_df())
    assert "Monday" in alias_map["Monday.com"]


def test_extract_all_returns_schema_shaped_frames():
    runs = pd.DataFrame(
        [
            {"run_id": "r1", "response_text": "Notion and Trello. https://g2.com/x"},
            {"run_id": "r2", "response_text": "Only Asana here."},
        ]
    )
    alias_map = {"Notion": [], "Trello": [], "Asana": []}
    mentions, citations = extract_all(runs, alias_map)
    assert set(mentions["brand_name"]) == {"Notion", "Trello", "Asana"}
    assert list(citations["citation_domain"]) == ["g2.com"]
