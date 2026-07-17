"""Tests for Answer Extractability analysis (no network required)."""

from __future__ import annotations

from src import page_audit as PA

CLUSTER_QS = [
    "Which tools do agencies use to manage client projects?",
    "What is the most affordable project management software for a 10-person team?",
]


def _good_row():
    return {
        "audit_status": "ok",
        "schema_types": "Article, FAQPage, Organization",
        "h1_count": 1, "h2_count": 4, "h3_count": 6, "word_count": 1500,
        "external_link_count": 5, "question_heading_count": 3, "answer_upfront": True,
        "has_author": True, "published_date": "2026-01-10", "modified_date": "2026-06-01",
        "canonical_url": "https://x.com/p", "robots_accessible": True, "sitemap_found": True,
        "page_title": "Best PM Tools for Agencies",
        "list_count": 4, "table_count": 2, "comparison_table_count": 1, "short_answer_count": 3,
        "headings_text": "Which tools do agencies use to manage client projects? | Pricing",
    }


def test_all_twelve_extractability_factors_present():
    facs = PA.extractability_factors(_good_row(), CLUSTER_QS)
    assert len(facs) == 12
    names = {f["factor"] for f in facs}
    for expected in ["Direct answer near the beginning", "Question-based headings",
                     "Short answer sections", "Lists", "Comparison tables",
                     "Clear brand, product & category entities",
                     "Supporting evidence & outbound sources",
                     "Answer schema (FAQ/Product/Organization/Article/HowTo)",
                     "Cluster question coverage", "Published & modified dates",
                     "Canonical URL", "Crawlability (robots & sitemap)"]:
        assert expected in names


def test_every_factor_exposes_its_rule_and_weight():
    for f in PA.extractability_factors(_good_row(), CLUSTER_QS):
        assert f["rule"], f"{f['factor']} is missing its rule text"
        assert f["weight"] > 0


def test_summary_is_transparent_not_opaque():
    res = PA.extractability_summary(_good_row(), CLUSTER_QS)
    assert res["formula"]
    assert len(res["components"]) == 12
    assert res["rules"] == PA.EXTRACTABILITY_RULES
    # Points reconcile with the stated formula.
    earned = sum(c["points"] for c in res["components"] if c["points"] is not None)
    considered = sum(c["weight"] for c in res["components"] if c["credit"] is not None)
    assert abs(res["score"] - (earned / considered * 100)) < 1e-9


def test_lists_and_tables_tiers():
    row = dict(_good_row())
    row.update({"list_count": 1, "comparison_table_count": 0, "table_count": 1})
    facs = {f["factor"]: f["status"] for f in PA.extractability_factors(row, CLUSTER_QS)}
    assert facs["Lists"] == "partial"
    assert facs["Comparison tables"] == "partial"

    row.update({"list_count": 0, "table_count": 0, "comparison_table_count": 0})
    facs = {f["factor"]: f["status"] for f in PA.extractability_factors(row, CLUSTER_QS)}
    assert facs["Lists"] == "fail"
    assert facs["Comparison tables"] == "fail"


def test_short_answer_sections_tier():
    row = dict(_good_row())
    row["short_answer_count"] = 1
    facs = {f["factor"]: f["status"] for f in PA.extractability_factors(row, CLUSTER_QS)}
    assert facs["Short answer sections"] == "partial"
    row["short_answer_count"] = 0
    facs = {f["factor"]: f["status"] for f in PA.extractability_factors(row, CLUSTER_QS)}
    assert facs["Short answer sections"] == "fail"


def test_cluster_coverage_unknown_without_cluster():
    facs = {f["factor"]: f["status"] for f in PA.extractability_factors(_good_row(), None)}
    assert facs["Cluster question coverage"] == "unknown"


def test_cluster_coverage_partial_when_half_matched():
    facs = {f["factor"]: f["status"] for f in PA.extractability_factors(_good_row(), CLUSTER_QS)}
    # Headings match the agency question but not the pricing question -> 50% -> partial.
    assert facs["Cluster question coverage"] == "partial"


def test_blocked_page_scores_none_not_zero():
    res = PA.extractability_summary({"audit_status": "blocked"}, CLUSTER_QS)
    assert res["score"] is None


def test_question_coverage_helper():
    cov = PA.question_coverage(
        "Which tools do agencies use to manage client projects?", "Agencies guide", CLUSTER_QS
    )
    assert cov["total"] == 2
    assert cov["covered"] == 1
    assert cov["coverage"] == 0.5
    assert len(cov["detail"]) == 2


def test_question_coverage_no_questions():
    cov = PA.question_coverage("anything", "title", [])
    assert cov["coverage"] is None
    assert cov["total"] == 0


def test_readiness_audit_still_works():
    """The original readiness audit must keep working alongside extractability."""
    res = PA.readiness_score(_good_row(), reference_year=2026)
    assert res["score"] == 100.0
    assert len(PA.readiness_factors(_good_row(), reference_year=2026)) == 12
