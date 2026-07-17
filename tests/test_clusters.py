"""Tests for AEO question-cluster analysis."""

from __future__ import annotations

import pandas as pd

from src import appkit
from src import clusters as C
from src.database import AnalysisData, ensure_prompt_cluster_columns


def _data() -> AnalysisData:
    prompts = pd.DataFrame([
        # Same intent + stage + persona -> should recommend ONE comprehensive page.
        {"prompt_id": "p1", "project_id": "x", "prompt_text": "best tool for agencies?",
         "prompt_category": "Nonbrand discovery", "topic": "Agency", "persona": "Agency Owner",
         "journey_stage": "Consideration", "is_brand_prompt": False,
         "search_intent": "Commercial investigation", "question_cluster": "Agency"},
        {"prompt_id": "p2", "project_id": "x", "prompt_text": "which tool do agencies use?",
         "prompt_category": "Nonbrand discovery", "topic": "Agency", "persona": "Agency Owner",
         "journey_stage": "Consideration", "is_brand_prompt": False,
         "search_intent": "Commercial investigation", "question_cluster": "Agency"},
        # Mixed intents + stages + personas -> should recommend SEPARATE pages.
        {"prompt_id": "p3", "project_id": "x", "prompt_text": "what is pm software?",
         "prompt_category": "Informational", "topic": "Basics", "persona": "Team Lead",
         "journey_stage": "Awareness", "is_brand_prompt": False,
         "search_intent": "Informational", "question_cluster": "Mixed"},
        {"prompt_id": "p4", "project_id": "x", "prompt_text": "cheapest plan?",
         "prompt_category": "Purchase intent", "topic": "Pricing", "persona": "Founder",
         "journey_stage": "Decision", "is_brand_prompt": False,
         "search_intent": "Transactional", "question_cluster": "Mixed"},
    ])
    runs = pd.DataFrame([
        {"run_id": "r1", "prompt_id": "p1", "platform": "P1", "model_name": "", "run_date": "2026-07-01",
         "run_number": 1, "response_text": "1. **Notion** is flexible.", "has_citations": False,
         "dataset_kind": "Synthetic", "benchmark_name": "B", "collection_date": "2026-07-01", "collection_notes": ""},
        {"run_id": "r2", "prompt_id": "p2", "platform": "P1", "model_name": "", "run_date": "2026-07-01",
         "run_number": 1, "response_text": "1. **Trello** is simple.", "has_citations": False,
         "dataset_kind": "Synthetic", "benchmark_name": "B", "collection_date": "2026-07-01", "collection_notes": ""},
        {"run_id": "r3", "prompt_id": "p3", "platform": "P1", "model_name": "", "run_date": "2026-07-01",
         "run_number": 1, "response_text": "No tools named here.", "has_citations": False,
         "dataset_kind": "Synthetic", "benchmark_name": "B", "collection_date": "2026-07-01", "collection_notes": ""},
        {"run_id": "r4", "prompt_id": "p4", "platform": "P1", "model_name": "", "run_date": "2026-07-01",
         "run_number": 1, "response_text": "1. **Notion** is affordable.", "has_citations": False,
         "dataset_kind": "Synthetic", "benchmark_name": "B", "collection_date": "2026-07-01", "collection_notes": ""},
    ])
    brands = pd.DataFrame([
        {"brand_id": "b1", "project_id": "x", "brand_name": "Notion", "brand_domain": "notion.so"},
        {"brand_id": "b2", "project_id": "x", "brand_name": "Trello", "brand_domain": "trello.com"},
    ])
    return appkit.run_extraction(AnalysisData(prompts=prompts, response_runs=runs, brands=brands))


def test_search_intent_derived_from_category_when_missing():
    prompts = pd.DataFrame([
        {"prompt_id": "p1", "prompt_text": "x", "prompt_category": "Product comparison", "topic": "T"},
    ])
    out = ensure_prompt_cluster_columns(prompts)
    assert out.iloc[0]["search_intent"] == "Commercial investigation"
    assert out.iloc[0]["question_cluster"] == "T"  # defaults to topic


def test_user_supplied_intent_and_cluster_are_preserved():
    prompts = pd.DataFrame([
        {"prompt_id": "p1", "prompt_text": "x", "prompt_category": "Product comparison", "topic": "T",
         "search_intent": "Transactional", "question_cluster": "My Cluster"},
    ])
    out = ensure_prompt_cluster_columns(prompts)
    assert out.iloc[0]["search_intent"] == "Transactional"
    assert out.iloc[0]["question_cluster"] == "My Cluster"


def test_brand_type_dimension_derived():
    data = _data()
    p = C.prepare_prompts(data.prompts)
    assert set(p["brand_type"]) == {"Non-brand question"}


def test_cluster_summary_counts_and_rates():
    data = _data()
    s = C.cluster_summary(data, "Trello", "question_cluster")
    agency = s[s["cluster"] == "Agency"].iloc[0]
    assert agency["prompts"] == 2
    assert agency["runs"] == 2
    # Trello appears in 1 of the 2 Agency responses.
    assert agency["focal_mention_rate"] == 0.5
    assert agency["top_competitor"] == "Notion"


def test_competitor_rates_in_cluster_lists_all_brands():
    data = _data()
    comp = C.competitor_rates_in_cluster(data, "question_cluster", "Agency")
    assert set(comp["brand_name"]) == {"Notion", "Trello"}


def test_question_outcomes_classifies_win_loss_and_absent():
    data = _data()
    outcomes = C.question_outcomes(data, "Trello", "question_cluster", "Agency")
    by_prompt = dict(zip(outcomes["prompt_id"], outcomes["outcome"]))
    assert by_prompt["p2"] == C.OUTCOME_FOCAL_WINS       # Trello mentioned, Notion not
    assert by_prompt["p1"] == C.OUTCOME_COMPETITOR_WINS  # Notion mentioned, Trello not

    mixed = C.question_outcomes(data, "Trello", "question_cluster", "Mixed")
    by_prompt2 = dict(zip(mixed["prompt_id"], mixed["outcome"]))
    assert by_prompt2["p3"] == C.OUTCOME_NO_BRAND        # no tracked brand at all


def test_consolidation_recommends_one_page_for_homogeneous_cluster():
    rec = C.page_consolidation_recommendation(_data(), "question_cluster", "Agency")
    assert rec["recommendation"] == "One comprehensive page"
    assert rec["evidence"]["questions"] == 2
    assert rec["rules"]  # rules always exposed


def test_consolidation_recommends_separate_pages_for_mixed_cluster():
    rec = C.page_consolidation_recommendation(_data(), "question_cluster", "Mixed")
    assert rec["recommendation"] == "Separate pages"
    assert rec["split_by"] == "search_intent"
    assert len(rec["evidence"]["search_intents"]) == 2


def test_cluster_questions_returns_prompt_texts():
    qs = C.cluster_questions(_data(), "question_cluster", "Agency")
    assert len(qs) == 2
    assert "best tool for agencies?" in qs


def test_demo_has_clusters_and_all_dimensions_work():
    data = appkit.load_demo_analysis()
    for dim in C.CLUSTER_DIMENSIONS:
        s = C.cluster_summary(data, "Notion", dim)
        assert not s.empty, f"dimension {dim} produced no clusters"


def test_empty_data_is_safe():
    empty = AnalysisData()
    assert C.cluster_summary(empty, "Notion", "question_cluster").empty
    assert C.question_outcomes(empty, "Notion", "question_cluster", "x").empty
    assert C.page_consolidation_recommendation(empty, "question_cluster", "x")["recommendation"] == "Not enough data"
