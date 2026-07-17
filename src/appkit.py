"""Streamlit glue: session-state management, demo loading, extraction, filtering.

Keeping this logic out of the page files makes the pages short and readable, and lets
us unit-test the non-UI parts (loading, filtering) without a running Streamlit server.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import pandas as pd

from .database import (
    AnalysisData,
    BENCHMARKS_COLUMNS,
    ensure_prompt_cluster_columns,
    load_analysis_from_csvs,
)
from .entities import extract_all_entities
from .extraction import build_alias_map, extract_all

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEMO_PROMPTS = os.path.join(DATA_DIR, "demo_prompts.csv")
DEMO_RESPONSES = os.path.join(DATA_DIR, "demo_responses.csv")

# The five demo brands and their domains (synthetic scenario).
DEMO_BRANDS = [
    ("Notion", "notion.so"),
    ("Asana", "asana.com"),
    ("ClickUp", "clickup.com"),
    ("Monday.com", "monday.com"),
    ("Trello", "trello.com"),
]


def demo_brands_df(project_id: str = "demo") -> pd.DataFrame:
    """Return the demo brands as a schema-shaped DataFrame."""
    rows = [
        {"brand_id": f"b{i+1}", "project_id": project_id, "brand_name": name, "brand_domain": domain}
        for i, (name, domain) in enumerate(DEMO_BRANDS)
    ]
    return pd.DataFrame(rows, columns=["brand_id", "project_id", "brand_name", "brand_domain"])


def load_demo_analysis() -> AnalysisData:
    """Load the synthetic demo dataset and run deterministic extraction on it.

    Returns a fully-populated :class:`AnalysisData` (raw tables + extracted
    ``brand_mentions`` and ``citations``). Page audits start empty.
    """
    data = load_analysis_from_csvs(
        DEMO_PROMPTS,
        DEMO_RESPONSES,
        project_name="Demo — Productivity Software (synthetic)",
        industry="Productivity Software",
    )
    data.brands = demo_brands_df()
    # Register the synthetic benchmark so real vs synthetic separation is explicit.
    data.benchmarks = pd.DataFrame(
        [{
            "benchmark_name": "Demo Synthetic Benchmark",
            "dataset_kind": "Synthetic",
            # The benchmark starts at the baseline wave.
            "created_at": "2026-06-05",
            "notes": (
                "Script-generated demo data (baseline 2026-06-05, post-change 2026-07-10). "
                "Synthetic demonstration scenario — not real AI platform output."
            ),
        }],
        columns=BENCHMARKS_COLUMNS,
    )
    return run_extraction(data)


def run_extraction(data: AnalysisData, alias_overrides: Optional[dict[str, list[str]]] = None) -> AnalysisData:
    """(Re)compute brand_mentions, citations, and narrative entities from response_runs.

    Returns a new :class:`AnalysisData`. Entity/narrative extraction (feature 2) runs
    alongside the original brand-mention and citation extraction so all deterministic
    signals are refreshed together.
    """
    alias_map = build_alias_map(data.brands, alias_overrides)
    mentions, citations = extract_all(data.response_runs, alias_map)
    entities = extract_all_entities(data.response_runs, alias_map)
    # Guarantee AEO clustering columns exist for hand-entered prompts too.
    prompts = ensure_prompt_cluster_columns(data.prompts)
    return replace(
        data, prompts=prompts, brand_mentions=mentions, citations=citations, brand_entities=entities
    )


# ---------------------------------------------------------------------------
# Session-state helpers (import streamlit lazily so this module stays testable).
# ---------------------------------------------------------------------------


def ensure_state() -> None:
    """Initialize Streamlit session state on first use."""
    import streamlit as st

    if "data" not in st.session_state:
        st.session_state["data"] = AnalysisData()
    if "focal_brand" not in st.session_state:
        st.session_state["focal_brand"] = None
    if "alias_overrides" not in st.session_state:
        st.session_state["alias_overrides"] = {}
    if "experiments" not in st.session_state:
        st.session_state["experiments"] = []
    # Page audits live on AnalysisData.page_audits (a single source of truth), so they
    # persist with the project — no separate session slot is needed.


def get_data() -> AnalysisData:
    """Return the active AnalysisData from session state."""
    import streamlit as st

    ensure_state()
    return st.session_state["data"]


def set_data(data: AnalysisData) -> None:
    import streamlit as st

    st.session_state["data"] = data


def has_data() -> bool:
    """True when there is at least one response loaded."""
    return not get_data().response_runs.empty


def focal_brand() -> Optional[str]:
    """The currently-selected focal brand, defaulting to the first configured brand."""
    import streamlit as st

    ensure_state()
    fb = st.session_state.get("focal_brand")
    brands = get_data().brand_names()
    if fb in brands:
        return fb
    return brands[0] if brands else None


# ---------------------------------------------------------------------------
# Filtering — returns a NEW AnalysisData scoped to the selected slice.
# ---------------------------------------------------------------------------


def filter_data(
    data: AnalysisData,
    platforms: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    topics: Optional[list[str]] = None,
    personas: Optional[list[str]] = None,
    journey_stages: Optional[list[str]] = None,
    run_dates: Optional[list[str]] = None,
    dataset_kinds: Optional[list[str]] = None,
    benchmark_names: Optional[list[str]] = None,
    search_intents: Optional[list[str]] = None,
    question_clusters: Optional[list[str]] = None,
) -> AnalysisData:
    """Filter an analysis by response and prompt attributes.

    A value of ``None`` (or empty list) means "no filter on that dimension". The
    returned object contains only the responses matching every active filter, plus
    the mentions/citations/entities belonging to those responses and the prompts they
    use. ``dataset_kinds`` and ``benchmark_names`` are the key controls that keep real
    and synthetic results separated.
    """
    runs = data.response_runs.copy()
    # Normalize so AEO cluster columns are always available to filter on.
    prompts = ensure_prompt_cluster_columns(data.prompts)

    # Prompt-level filters restrict the set of prompt_ids.
    if categories:
        prompts = prompts[prompts["prompt_category"].isin(categories)]
    if topics:
        prompts = prompts[prompts["topic"].isin(topics)]
    if personas:
        prompts = prompts[prompts["persona"].isin(personas)]
    if journey_stages:
        prompts = prompts[prompts["journey_stage"].isin(journey_stages)]
    if search_intents:
        prompts = prompts[prompts["search_intent"].isin(search_intents)]
    if question_clusters:
        prompts = prompts[prompts["question_cluster"].isin(question_clusters)]
    allowed_prompt_ids = set(prompts["prompt_id"])

    # Response-level filters.
    if not runs.empty:
        runs = runs[runs["prompt_id"].isin(allowed_prompt_ids)]
    if platforms and not runs.empty:
        runs = runs[runs["platform"].isin(platforms)]
    if run_dates and not runs.empty:
        runs = runs[runs["run_date"].isin(run_dates)]
    if dataset_kinds and not runs.empty and "dataset_kind" in runs.columns:
        runs = runs[runs["dataset_kind"].isin(dataset_kinds)]
    if benchmark_names and not runs.empty and "benchmark_name" in runs.columns:
        runs = runs[runs["benchmark_name"].isin(benchmark_names)]

    allowed_run_ids = set(runs["run_id"]) if not runs.empty else set()

    def _subset(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["run_id"].isin(allowed_run_ids)] if not df.empty else df

    return replace(
        data,
        prompts=prompts.reset_index(drop=True),
        response_runs=runs.reset_index(drop=True),
        brand_mentions=_subset(data.brand_mentions).reset_index(drop=True),
        citations=_subset(data.citations).reset_index(drop=True),
        brand_entities=_subset(data.brand_entities).reset_index(drop=True),
    )


def unique_values(df: pd.DataFrame, column: str) -> list[str]:
    """Sorted unique non-null string values of a column (safe on empty frames)."""
    if df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())
