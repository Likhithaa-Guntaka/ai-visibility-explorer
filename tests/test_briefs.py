"""Tests for deterministic content action briefs."""

from __future__ import annotations

import pandas as pd

from src import appkit
from src import briefs as B
from src.database import AnalysisData


def test_briefs_generated_for_gaps():
    data = appkit.load_demo_analysis()
    briefs = B.build_briefs(data, "Trello", max_briefs=3)
    assert len(briefs) >= 1
    b = briefs[0]
    # Every required field is populated and grounded.
    assert b.topic
    assert b.target_persona
    assert b.journey_stage
    assert b.prompt_category
    assert b.suggested_title
    assert b.suggested_format
    assert b.recommended_schema
    assert b.evidence_needed
    assert isinstance(b.suggested_headings, list) and b.suggested_headings
    assert "Trello" in b.suggested_title


def test_brief_questions_come_from_real_prompts():
    data = appkit.load_demo_analysis()
    briefs = B.build_briefs(data, "Trello", max_briefs=5)
    all_prompt_texts = set(data.prompts["prompt_text"])
    for b in briefs:
        for q in b.questions_to_answer:
            assert q in all_prompt_texts  # grounded, not invented


def test_brief_markdown_has_all_headers():
    data = appkit.load_demo_analysis()
    briefs = B.build_briefs(data, "Trello", max_briefs=1)
    md = B.brief_to_markdown(briefs[0])
    for label in ["Target persona", "Journey stage", "Prompt topic", "Suggested content format",
                  "Suggested title", "Recommended schema type", "Evidence", "Questions to answer",
                  "Suggested headings", "third-party source opportunities"]:
        assert label in md


def test_no_briefs_when_no_data():
    assert B.build_briefs(AnalysisData(), "Notion") == []


def test_schema_and_format_are_category_grounded():
    data = appkit.load_demo_analysis()
    briefs = B.build_briefs(data, "Trello", max_briefs=8)
    for b in briefs:
        assert b.recommended_schema == B.CATEGORY_SCHEMA.get(b.prompt_category, B._DEFAULT_SCHEMA)
        assert b.suggested_format == B.CATEGORY_FORMAT.get(b.prompt_category, B._DEFAULT_FORMAT)
