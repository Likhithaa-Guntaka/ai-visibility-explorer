"""Small shared Streamlit UI helpers (filters, guards) used across pages."""

from __future__ import annotations

from typing import Optional

from .database import AnalysisData
from . import appkit


def require_data(st) -> Optional[AnalysisData]:
    """Guard used at the top of analysis pages.

    Returns the loaded :class:`AnalysisData`, or shows a friendly prompt and returns
    ``None`` when nothing is loaded yet.
    """
    if not appkit.has_data():
        st.info("No data loaded. Go to the **home page** or **Data Input** and load data first.")
        st.page_link("app.py", label="→ Home / load demo data", icon="🏠")
        return None
    data = appkit.get_data()
    if data.brand_mentions.empty:
        st.warning(
            "Responses are loaded but extraction hasn't run yet. Open **Data Input → "
            "Review & Correct** and click *Run extraction*."
        )
        return None
    return data


def sidebar_filters(st, data: AnalysisData) -> AnalysisData:
    """Render sidebar filter widgets and return a filtered copy of ``data``.

    Filters: brand (focal), platform, prompt category, topic, persona, journey stage,
    and experiment (run) date. Empty selections mean "all".
    """
    st.sidebar.header("Filters")

    brands = data.brand_names()
    if brands:
        current = appkit.focal_brand()
        idx = brands.index(current) if current in brands else 0
        focal = st.sidebar.selectbox("Focal brand", brands, index=idx, help="The brand most metrics are computed for.")
        st.session_state["focal_brand"] = focal

    # Dataset-kind and benchmark filters keep REAL and SYNTHETIC data separated.
    kind_options = appkit.unique_values(data.response_runs, "dataset_kind")
    dataset_kinds = st.sidebar.multiselect(
        "Dataset type", kind_options,
        help="Keep Real, User Collected, and Synthetic data separate. Leave empty to view all (a warning is shown if mixed).",
    )
    benchmark_options = appkit.unique_values(data.response_runs, "benchmark_name")
    benchmark_options = [b for b in benchmark_options if b]
    benchmark_names = st.sidebar.multiselect("Benchmark", benchmark_options) if benchmark_options else []

    platforms = st.sidebar.multiselect("AI platform", appkit.unique_values(data.response_runs, "platform"))
    categories = st.sidebar.multiselect("Prompt category", appkit.unique_values(data.prompts, "prompt_category"))
    topics = st.sidebar.multiselect("Topic", appkit.unique_values(data.prompts, "topic"))
    personas = st.sidebar.multiselect("Persona", appkit.unique_values(data.prompts, "persona"))
    stages = st.sidebar.multiselect("Journey stage", appkit.unique_values(data.prompts, "journey_stage"))
    dates = st.sidebar.multiselect("Experiment date", appkit.unique_values(data.response_runs, "run_date"))

    filtered = appkit.filter_data(
        data,
        platforms=platforms or None,
        categories=categories or None,
        topics=topics or None,
        personas=personas or None,
        journey_stages=stages or None,
        run_dates=dates or None,
        dataset_kinds=dataset_kinds or None,
        benchmark_names=benchmark_names or None,
    )
    st.sidebar.caption(f"{len(filtered.response_runs)} of {len(data.response_runs)} responses in view.")

    # Honesty guard: warn whenever the current view mixes real and synthetic data.
    if not filtered.response_runs.empty and "dataset_kind" in filtered.response_runs.columns:
        kinds_in_view = sorted(filtered.response_runs["dataset_kind"].dropna().unique().tolist())
        if len(kinds_in_view) > 1:
            st.sidebar.warning(
                "⚠ This view mixes dataset types: " + ", ".join(kinds_in_view) +
                ". Filter by a single **Dataset type** to keep real and synthetic results separate.",
            )
        elif kinds_in_view:
            st.sidebar.caption(f"Dataset type in view: **{kinds_in_view[0]}**")
    return filtered
