"""Tests for deterministic entity & narrative extraction."""

from __future__ import annotations

import pandas as pd

from src import entities as E


def _runs(texts: dict[str, str]) -> pd.DataFrame:
    rows = []
    for i, (run_id, text) in enumerate(texts.items()):
        rows.append({"run_id": run_id, "platform": "P" + str(i % 2), "run_number": 1, "response_text": text})
    return pd.DataFrame(rows)


ALIASES = {"Notion": [], "Trello": [], "Asana": []}


def test_extracts_features_and_strengths_and_weaknesses():
    text = "1. **Notion** is a flexible all-in-one workspace with docs, wikis, and databases, but it can be overwhelming."
    runs = _runs({"r1": text})
    ent = E.extract_all_entities(runs, ALIASES)
    notion = ent[ent["brand_name"] == "Notion"].iloc[0]
    assert "Flexible" in notion["strengths"]
    assert "All-in-one" in notion["strengths"]
    assert "Docs" in notion["features"]
    assert "Overwhelming" in notion["weaknesses"]


def test_competitors_alongside_lists_other_brands():
    text = "1. **Notion** is great. 2. **Asana** is reliable. 3. **Trello** is simple."
    runs = _runs({"r1": text})
    ent = E.extract_all_entities(runs, ALIASES)
    notion = ent[ent["brand_name"] == "Notion"].iloc[0]
    assert "Asana" in notion["competitors_alongside"]
    assert "Trello" in notion["competitors_alongside"]
    assert "Notion" not in notion["competitors_alongside"]


def test_brand_not_mentioned_is_skipped():
    runs = _runs({"r1": "1. **Asana** only."})
    ent = E.extract_all_entities(runs, ALIASES)
    assert "Notion" not in set(ent["brand_name"])
    assert "Asana" in set(ent["brand_name"])


def test_pronoun_clause_in_same_block_is_attributed():
    # The 'pricey' clause uses 'It', but stays in the same block/line as ClickUp.
    text = "1. **ClickUp** is affordable. It can get pricey on higher tiers."
    runs = _runs({"r1": text})
    ent = E.extract_all_entities(runs, {"ClickUp": []})
    row = ent[ent["brand_name"] == "ClickUp"].iloc[0]
    assert "Affordable" in row["strengths"]
    assert "Expensive" in row["weaknesses"]


def test_conflicting_descriptions_detected():
    runs = _runs({
        "r1": "1. **ClickUp** is affordable and easy to use.",
        "r2": "1. **ClickUp** can get pricey and has a steep learning curve.",
    })
    ent = E.extract_all_entities(runs, {"ClickUp": []})
    conflicts = E.conflicting_descriptions(ent, "ClickUp")
    pairs = set(zip(conflicts["descriptor_a"], conflicts["descriptor_b"]))
    assert ("Affordable", "Expensive") in pairs or ("Affordable", "Premium / expensive") in pairs
    assert ("Easy to use", "Steep learning curve") in pairs


def test_descriptor_frequency_and_share():
    runs = _runs({
        "r1": "1. **Notion** is flexible.",
        "r2": "1. **Notion** is flexible and powerful.",
    })
    ent = E.extract_all_entities(runs, {"Notion": []})
    freq = E.descriptor_frequency(ent, "Notion", "strengths")
    flex = freq[freq["descriptor"] == "Flexible"].iloc[0]
    assert flex["count"] == 2
    assert flex["share"] == 1.0


def test_narrative_consistency_identical_runs_is_one():
    runs = _runs({
        "r1": "1. **Notion** is flexible and powerful.",
        "r2": "1. **Notion** is flexible and powerful.",
    })
    ent = E.extract_all_entities(runs, {"Notion": []})
    nc = E.narrative_consistency(ent, runs, "Notion")
    assert nc["runs"] == 2
    assert nc["consistency"] == 1.0


def test_narrative_consistency_none_when_single_run():
    runs = _runs({"r1": "1. **Notion** is flexible."})
    ent = E.extract_all_entities(runs, {"Notion": []})
    assert E.narrative_consistency(ent, runs, "Notion")["consistency"] is None


def test_attribute_coverage_flags_missing_fields():
    # No persona / category language -> those attributes have 0 coverage.
    runs = _runs({"r1": "1. **Notion** is flexible with docs."})
    ent = E.extract_all_entities(runs, {"Notion": []})
    cov = E.attribute_coverage(ent, "Notion")
    coverage = dict(zip(cov["attribute"], cov["coverage"]))
    assert coverage["features"] == 1.0
    assert coverage["personas"] == 0.0


def test_platform_descriptions_groups_by_platform():
    runs = pd.DataFrame([
        {"run_id": "r1", "platform": "ChatGPT", "run_number": 1, "response_text": "1. **Notion** is flexible."},
        {"run_id": "r2", "platform": "Claude", "run_number": 1, "response_text": "1. **Notion** is powerful."},
    ])
    ent = E.extract_all_entities(runs, {"Notion": []})
    pd_desc = E.platform_descriptions(ent, runs, "Notion", "strengths")
    by_platform = dict(zip(pd_desc["platform"], pd_desc["descriptors"]))
    assert "Flexible" in by_platform["ChatGPT"]
    assert "Powerful" in by_platform["Claude"]


def test_empty_inputs_return_empty():
    empty = pd.DataFrame(columns=["run_id", "platform", "run_number", "response_text"])
    ent = E.extract_all_entities(empty, {"Notion": []})
    assert ent.empty
    assert E.conflicting_descriptions(ent, "Notion").empty
    assert E.narrative_consistency(ent, empty, "Notion")["consistency"] is None
