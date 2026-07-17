"""Tests for Real Benchmark Mode: dataset labels and filtering separation."""

from __future__ import annotations

import pandas as pd

from src import appkit
from src.database import AnalysisData, Database, DATASET_KINDS


def _mixed_data() -> AnalysisData:
    runs = pd.DataFrame([
        {"run_id": "r1", "prompt_id": "p1", "platform": "ChatGPT", "model_name": "", "run_date": "2026-07-01",
         "run_number": 1, "response_text": "1. **Notion** is flexible.", "has_citations": False,
         "dataset_kind": "Synthetic", "benchmark_name": "Demo", "collection_date": "", "collection_notes": ""},
        {"run_id": "r2", "prompt_id": "p1", "platform": "ChatGPT", "model_name": "", "run_date": "2026-07-02",
         "run_number": 1, "response_text": "1. **Asana** is reliable.", "has_citations": False,
         "dataset_kind": "Real", "benchmark_name": "July Real", "collection_date": "2026-07-02", "collection_notes": "manual"},
    ])
    prompts = pd.DataFrame([
        {"prompt_id": "p1", "project_id": "x", "prompt_text": "best tool?", "prompt_category": "Nonbrand discovery",
         "topic": "Best tools", "persona": "Team Lead", "journey_stage": "Consideration", "is_brand_prompt": False},
    ])
    brands = pd.DataFrame([
        {"brand_id": "b1", "project_id": "x", "brand_name": "Notion", "brand_domain": "notion.so"},
        {"brand_id": "b2", "project_id": "x", "brand_name": "Asana", "brand_domain": "asana.com"},
    ])
    data = AnalysisData(prompts=prompts, response_runs=runs, brands=brands)
    return appkit.run_extraction(data)


def test_dataset_kinds_constant():
    assert DATASET_KINDS == ["Synthetic", "Real", "User Collected"]


def test_filter_by_dataset_kind_separates_real_and_synthetic():
    data = _mixed_data()
    synth = appkit.filter_data(data, dataset_kinds=["Synthetic"])
    real = appkit.filter_data(data, dataset_kinds=["Real"])
    assert set(synth.response_runs["run_id"]) == {"r1"}
    assert set(real.response_runs["run_id"]) == {"r2"}
    # Derived tables (mentions/entities/citations) are subset too.
    assert set(synth.brand_mentions["run_id"]) <= {"r1"}
    assert set(real.brand_entities["run_id"]) <= {"r2"}


def test_filter_by_benchmark_name():
    data = _mixed_data()
    july = appkit.filter_data(data, benchmark_names=["July Real"])
    assert set(july.response_runs["run_id"]) == {"r2"}


def test_extended_schema_loads_into_duckdb():
    data = _mixed_data()
    with Database.from_analysis_data(data) as db:
        df = db.query("SELECT dataset_kind, COUNT(*) n FROM response_runs GROUP BY dataset_kind ORDER BY dataset_kind")
        counts = dict(zip(df["dataset_kind"], df["n"]))
        assert counts["Real"] == 1
        assert counts["Synthetic"] == 1
        # benchmarks + brand_entities tables exist and are queryable
        assert db.query("SELECT COUNT(*) c FROM brand_entities")["c"].iloc[0] >= 1


def test_demo_is_labelled_synthetic():
    data = appkit.load_demo_analysis()
    assert set(data.response_runs["dataset_kind"].unique()) == {"Synthetic"}
    assert not data.benchmarks.empty
    assert data.benchmarks.iloc[0]["dataset_kind"] == "Synthetic"
