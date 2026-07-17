"""Tests for before/after AEO experiments."""

from __future__ import annotations

import pandas as pd
import pytest

from src import appkit
from src import experiments as X
from src.database import AnalysisData


def _row(run_id, prompt_id, date, text, platform="P1"):
    return {
        "run_id": run_id, "prompt_id": prompt_id, "platform": platform, "model_name": "",
        "run_date": date, "run_number": 1, "response_text": text, "has_citations": "http" in text,
        "dataset_kind": "Real", "benchmark_name": "B", "collection_date": date, "collection_notes": "",
    }


def _data() -> AnalysisData:
    prompts = pd.DataFrame([
        {"prompt_id": "p1", "project_id": "x", "prompt_text": "best tool?",
         "prompt_category": "Nonbrand discovery", "topic": "Best", "persona": "Team Lead",
         "journey_stage": "Consideration", "is_brand_prompt": False,
         "search_intent": "Commercial investigation", "question_cluster": "Discovery"},
        {"prompt_id": "p2", "project_id": "x", "prompt_text": "cheapest tool?",
         "prompt_category": "Purchase intent", "topic": "Pricing", "persona": "Founder",
         "journey_stage": "Decision", "is_brand_prompt": False,
         "search_intent": "Transactional", "question_cluster": "Pricing"},
    ])
    runs = pd.DataFrame([
        # Baseline: Trello absent from both questions.
        _row("b1", "p1", "2026-07-01", "1. **Notion** is flexible."),
        _row("b2", "p2", "2026-07-01", "1. **Notion** is affordable."),
        # Post: Trello now appears in both.
        _row("a1", "p1", "2026-08-01", "1. **Trello** is simple. 2. **Notion** is flexible."),
        _row("a2", "p2", "2026-08-01", "1. **Trello** is affordable. https://g2.com/x"),
    ])
    brands = pd.DataFrame([
        {"brand_id": "b1", "project_id": "x", "brand_name": "Notion", "brand_domain": "notion.so"},
        {"brand_id": "b2", "project_id": "x", "brand_name": "Trello", "brand_domain": "trello.com"},
    ])
    return appkit.run_extraction(AnalysisData(prompts=prompts, response_runs=runs, brands=brands))


@pytest.fixture
def exp():
    return X.Experiment(
        name="test", focal_brand="Trello", baseline_date="2026-07-01", post_date="2026-08-01",
        cluster_dimension="question_cluster", cluster_value=None,
        change_made="added pages", hypothesis="visibility rises",
        primary_kpi="Brand mention rate", secondary_kpis=["Share of voice"],
    )


def test_available_dates():
    assert X.available_dates(_data()) == ["2026-07-01", "2026-08-01"]


def test_slice_arm_splits_by_date():
    data = _data()
    base = X.slice_arm(data, "2026-07-01")
    post = X.slice_arm(data, "2026-08-01")
    assert set(base.response_runs["run_id"]) == {"b1", "b2"}
    assert set(post.response_runs["run_id"]) == {"a1", "a2"}
    # Derived tables follow the slice.
    assert set(base.brand_mentions["run_id"]) <= {"b1", "b2"}


def test_slice_arm_respects_cluster():
    data = _data()
    arm = X.slice_arm(data, "2026-07-01", "question_cluster", "Pricing")
    assert set(arm.response_runs["run_id"]) == {"b2"}


def test_mention_rate_change_is_computed(exp):
    res = X.compare_experiment(_data(), exp)
    comp = res["comparison"]
    mr = comp[comp["metric"] == "Brand mention rate"].iloc[0]
    assert mr["baseline"] == 0.0   # Trello absent at baseline
    assert mr["post"] == 1.0       # Trello in both post responses
    assert mr["absolute_change"] == 1.0
    assert mr["pp_change"] == 100.0
    assert bool(mr["is_primary"]) is True


def test_sample_sizes_reported(exp):
    res = X.compare_experiment(_data(), exp)
    assert res["baseline_n"] == 2
    assert res["post_n"] == 2


def test_all_required_metrics_present(exp):
    res = X.compare_experiment(_data(), exp)
    metrics = set(res["comparison"]["metric"])
    for required in ["Brand mention rate", "Share of voice", "First mention share",
                     "Recommendation rate", "Citation rate", "Source coverage",
                     "Narrative consistency", "Top competitor share of voice"]:
        assert required in metrics


def test_count_metric_has_no_pp_change(exp):
    res = X.compare_experiment(_data(), exp)
    src = res["comparison"]
    row = src[src["metric"] == "Source coverage"].iloc[0]
    assert row["unit"] == "count"
    # A percentage-point change is meaningless for a count; pandas stores the
    # module's None as NaN in the float column.
    assert pd.isna(row["pp_change"])


def test_platform_and_prompt_level_results(exp):
    res = X.compare_experiment(_data(), exp)
    assert not res["platform"].empty
    prompts = res["prompt_level"]
    p1 = prompts[prompts["prompt_id"] == "p1"].iloc[0]
    assert p1["baseline_rate"] == 0.0
    assert p1["post_rate"] == 1.0
    assert p1["pp_change"] == 100.0


def test_limitations_always_use_associational_language(exp):
    res = X.compare_experiment(_data(), exp)
    joined = " ".join(res["limitations"]).lower()
    assert "not proof" in joined or "not a controlled experiment" in joined
    assert "association" in joined
    # Small samples are flagged.
    assert "small sample" in joined


def test_empty_arm_flagged(exp):
    bad = X.Experiment(**{**exp.__dict__, "post_date": "2099-01-01"})
    res = X.compare_experiment(_data(), bad)
    assert res["post_n"] == 0
    assert any("no responses" in item.lower() for item in res["limitations"])


def test_demo_two_waves_compare(exp):
    data = appkit.load_demo_analysis()
    dates = X.available_dates(data)
    assert len(dates) >= 2
    demo_exp = X.Experiment(
        name="demo", focal_brand="Trello", baseline_date=dates[0], post_date=dates[-1],
        primary_kpi="Brand mention rate",
    )
    res = X.compare_experiment(data, demo_exp)
    assert res["baseline_n"] > 0 and res["post_n"] > 0
    assert len(res["comparison"]) == len(X.KPI_OPTIONS)


def test_format_change_rate_and_count():
    rate_row = pd.Series({"baseline": 0.1, "post": 0.2, "unit": "rate", "pp_change": 10.0, "absolute_change": 0.1})
    count_row = pd.Series({"baseline": 3, "post": 5, "unit": "count", "pp_change": None, "absolute_change": 2})
    assert X.format_change(rate_row) == "+10.0 pp"
    assert X.format_change(count_row) == "+2"
