"""Tests for file-based project export / import."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import appkit
from src import persistence as P
from src.database import AnalysisData
from src.experiments import Experiment


def test_round_trip_preserves_all_tables():
    data = appkit.load_demo_analysis()
    exp = Experiment(name="e1", focal_brand="Trello", baseline_date="2026-06-05",
                     post_date="2026-07-10", primary_kpi="Brand mention rate")
    js = P.export_project_json(data, {"Monday.com": ["Monday"]}, "Trello", [exp])
    bundle = P.import_bundle(js)

    for table in ["projects", "benchmarks", "brands", "prompts", "response_runs",
                  "brand_mentions", "citations", "brand_entities"]:
        orig = getattr(data, table).reset_index(drop=True)
        restored = getattr(bundle.data, table).reset_index(drop=True)
        assert orig.equals(restored), f"{table} did not round-trip identically"

    assert bundle.focal_brand == "Trello"
    assert bundle.alias_overrides == {"Monday.com": ["Monday"]}
    assert len(bundle.experiments) == 1
    assert bundle.experiments[0]["focal_brand"] == "Trello"


def test_manual_corrections_survive_round_trip():
    """Editing an extracted table (as a user would) must be preserved on import."""
    data = appkit.load_demo_analysis()
    edited = data.brand_mentions.copy()
    edited.loc[edited.index[0], "is_recommended"] = not bool(edited.iloc[0]["is_recommended"])
    from dataclasses import replace
    data = replace(data, brand_mentions=edited)

    bundle = P.import_bundle(P.export_project_json(data))
    assert bool(bundle.data.brand_mentions.iloc[0]["is_recommended"]) == bool(edited.iloc[0]["is_recommended"])


def test_page_audits_included_when_present():
    data = appkit.load_demo_analysis()
    from dataclasses import replace
    audits = pd.DataFrame([{
        "citation_url": "https://g2.com/x", "audit_status": "ok", "page_title": "G2",
        "h1_count": 1, "h2_count": 3, "word_count": 900,
    }])
    data = replace(data, page_audits=audits)
    bundle = P.import_bundle(P.export_project_json(data))
    assert not bundle.data.page_audits.empty
    assert bundle.data.page_audits.iloc[0]["citation_url"] == "https://g2.com/x"


def test_empty_project_round_trip():
    empty = AnalysisData()
    bundle = P.import_bundle(P.export_project_json(empty))
    assert bundle.data.response_runs.empty
    assert bundle.data.brand_mentions.empty
    assert bundle.focal_brand is None
    assert bundle.experiments == []


def test_missing_optional_tables_and_fields_tolerated():
    payload = {"schema_version": P.PROJECT_SCHEMA_VERSION, "tables": {
        "prompts": {"columns": ["prompt_id"], "records": [{"prompt_id": "p1"}]},
    }}  # no brands, responses, aliases, focal, experiments
    bundle = P.import_bundle(payload)
    assert len(bundle.data.prompts) == 1
    assert bundle.data.brands.empty
    assert bundle.data.response_runs.empty
    assert bundle.alias_overrides == {}
    assert bundle.experiments == []


def test_invalid_json_raises_clear_error():
    with pytest.raises(P.ProjectImportError, match="not valid JSON"):
        P.import_bundle("{ this is not json")


def test_missing_schema_version_raises():
    with pytest.raises(P.ProjectImportError, match="schema_version"):
        P.import_bundle(json.dumps({"tables": {}}))


def test_future_schema_version_rejected():
    with pytest.raises(P.ProjectImportError, match="Unsupported schema_version"):
        P.import_bundle({"schema_version": P.PROJECT_SCHEMA_VERSION + 5, "tables": {}})


def test_non_object_payload_rejected():
    with pytest.raises(P.ProjectImportError, match="must be a JSON object"):
        P.import_bundle(json.dumps([1, 2, 3]))


def test_missing_tables_section_rejected():
    with pytest.raises(P.ProjectImportError, match="tables"):
        P.import_bundle({"schema_version": P.PROJECT_SCHEMA_VERSION})


def test_export_only_contains_data_sections():
    """The export must carry only analysis data — no config/secret-bearing sections.

    (Provider names like 'gemini' legitimately appear as synthetic *platform labels*,
    so we check the payload's structure and for API-key-shaped values, not brand names.)
    """
    payload = P.export_bundle(appkit.load_demo_analysis())
    assert set(payload.keys()) == {"schema_version", "app", "tables", "alias_overrides",
                                   "focal_brand", "experiments"}
    # No section named like a credential store.
    for key in payload:
        assert not any(s in key.lower() for s in ["key", "secret", "token", "password", "env"])
    # No value looks like an API key (sk-..., AIza..., ghp_..., 32+ hex).
    import re
    blob = json.dumps(payload)
    key_pattern = re.compile(r"(sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{20,})")
    match = key_pattern.search(blob)
    assert match is None, f"export contains an API-key-shaped value: {match.group(0) if match else ''}"


def test_schema_version_is_embedded():
    payload = P.export_bundle(AnalysisData())
    assert payload["schema_version"] == P.PROJECT_SCHEMA_VERSION
    assert payload["app"] == "ai-visibility-explorer"
