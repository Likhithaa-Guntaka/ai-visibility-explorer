"""Tests for the AI Answer Readiness audit logic (no network required)."""

from __future__ import annotations

from src import page_audit as PA


def _good_row():
    return {
        "audit_status": "ok",
        "schema_types": "Article, FAQPage, Organization",
        "h1_count": 1, "h2_count": 4, "h3_count": 6, "word_count": 1500,
        "external_link_count": 5, "question_heading_count": 3, "answer_upfront": True,
        "has_author": True, "published_date": "2026-01-10", "modified_date": "2026-06-01",
        "canonical_url": "https://x.com/p", "robots_accessible": True, "sitemap_found": True,
        "page_title": "Best PM Tools",
    }


def test_all_twelve_factors_present():
    facs = PA.readiness_factors(_good_row(), reference_year=2026)
    assert len(facs) == 12
    names = {f["factor"] for f in facs}
    assert "Direct answer near the beginning" in names
    assert "Content freshness" in names


def test_good_page_scores_full():
    res = PA.readiness_score(_good_row(), reference_year=2026)
    assert res["score"] == 100.0
    assert res["points_considered"] == 100.0


def test_thin_old_page_scores_low_but_transparent():
    row = dict(_good_row())
    row.update({"word_count": 120, "modified_date": "2015-01-01", "published_date": "2015-01-01",
                "external_link_count": 0, "question_heading_count": 0, "answer_upfront": False,
                "schema_types": "", "has_author": False, "h1_count": 3, "h2_count": 0, "h3_count": 0})
    res = PA.readiness_score(row, reference_year=2026)
    assert res["score"] < 50
    # Formula and components are always exposed (no opaque score).
    assert "credit" in res["formula"]
    assert len(res["components"]) == 12


def test_blocked_page_is_unknown_not_zero():
    res = PA.readiness_score({"audit_status": "blocked"}, reference_year=2026)
    # Everything unobservable -> excluded -> score is None (not a misleading 0).
    assert res["score"] is None
    assert all(c["status"] == "unknown" for c in res["components"])


def test_partial_credit_for_moderate_depth():
    row = dict(_good_row())
    row["word_count"] = 500  # moderate
    facs = {f["factor"]: f["status"] for f in PA.readiness_factors(row, reference_year=2026)}
    assert facs["Topic coverage (depth)"] == "partial"


def test_freshness_uses_reference_year():
    row = dict(_good_row())
    row["modified_date"] = "2020-01-01"
    facs = {f["factor"]: f["status"] for f in PA.readiness_factors(row, reference_year=2026)}
    assert facs["Content freshness"] == "fail"


def test_question_heading_detection():
    assert PA._is_question("How do I pick a tool?") is True
    assert PA._is_question("What is Notion") is True
    assert PA._is_question("Our pricing") is False
