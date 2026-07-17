"""Data Input — create a project, add brands/prompts, import responses, correct extraction."""

from __future__ import annotations

import io
from dataclasses import replace

import pandas as pd
import streamlit as st

from src import appkit
from src.database import (
    AnalysisData,
    DATASET_KINDS,
    JOURNEY_STAGES,
    PROMPT_CATEGORIES,
    _to_bool,
)
from src.validation import validate_prompts, validate_responses

st.set_page_config(page_title="Data Input", page_icon="📥", layout="wide")
appkit.ensure_state()


# ---------------------------------------------------------------------------
# Small local helpers (defined first because Streamlit runs the script top-down).
# ---------------------------------------------------------------------------


def _read_csv(uploaded) -> pd.DataFrame:
    """Read an uploaded CSV into strings, tolerating BOM and blank cells."""
    raw = uploaded.getvalue()
    return pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)


def _show_validation(result) -> None:
    for err in result.errors:
        st.error(err)
    for warn in result.warnings:
        st.warning(warn)
    if result.ok and not result.warnings:
        st.success("File looks valid.")


def _parse_aliases(text: str) -> dict[str, list[str]]:
    """Parse 'Brand=a,b; Other=c' into {'Brand': ['a','b'], 'Other': ['c']}."""
    out: dict[str, list[str]] = {}
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        brand, aliases = chunk.split("=", 1)
        vals = [a.strip() for a in aliases.split(",") if a.strip()]
        if brand.strip() and vals:
            out[brand.strip()] = vals
    return out


st.title("📥 Data Input")
st.caption("Build an analysis by hand, or load the synthetic demo, then run extraction.")

data = appkit.get_data()

# Quick demo shortcut so users don't have to go back to the home page.
if st.button("🚀 Load synthetic demo dataset"):
    appkit.set_data(appkit.load_demo_analysis())
    st.session_state["focal_brand"] = "Notion"
    st.success("Demo data loaded and extracted.")
    st.rerun()

tab_project, tab_prompts, tab_responses, tab_review = st.tabs(
    ["1 · Project & Brands", "2 · Prompts", "3 · Responses", "4 · Review & Correct"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Project & Brands
# ---------------------------------------------------------------------------
with tab_project:
    st.subheader("Project")
    existing_project = data.projects.iloc[0].to_dict() if not data.projects.empty else {}
    with st.form("project_form"):
        pname = st.text_input("Project name", value=existing_project.get("project_name", "My Analysis"))
        industry = st.text_input("Industry", value=existing_project.get("industry", "Productivity Software"))
        if st.form_submit_button("Save project"):
            projects = pd.DataFrame(
                [{"project_id": "user", "project_name": pname, "industry": industry, "created_at": ""}]
            )
            appkit.set_data(replace(appkit.get_data(), projects=projects))
            st.success("Project saved.")

    st.subheader("Brands & competitors")
    st.caption("Add every brand you want to track, including competitors. Aliases help match nicknames (e.g. 'Monday' for 'Monday.com').")
    brands_seed = data.brands if not data.brands.empty else pd.DataFrame(
        columns=["brand_id", "project_id", "brand_name", "brand_domain"]
    )
    edited_brands = st.data_editor(
        brands_seed[["brand_name", "brand_domain"]] if not brands_seed.empty else pd.DataFrame(
            [{"brand_name": "", "brand_domain": ""}]
        ),
        num_rows="dynamic",
        use_container_width=True,
        key="brands_editor",
        column_config={
            "brand_name": st.column_config.TextColumn("Brand name", required=True),
            "brand_domain": st.column_config.TextColumn("Brand domain (optional)"),
        },
    )
    # Optional aliases entered as free text: "Monday.com=Monday,monday; Notion=notion.so"
    alias_text = st.text_input(
        "Brand aliases (optional)",
        value="Monday.com=Monday",
        help="Format: Brand=alias1,alias2; separate brands with semicolons.",
    )
    if st.button("Save brands & aliases"):
        clean = edited_brands.dropna(subset=["brand_name"])
        clean = clean[clean["brand_name"].astype(str).str.strip() != ""]
        brands_df = pd.DataFrame(
            [
                {
                    "brand_id": f"b{i+1}",
                    "project_id": "user",
                    "brand_name": str(row["brand_name"]).strip(),
                    "brand_domain": str(row.get("brand_domain", "") or "").strip(),
                }
                for i, (_, row) in enumerate(clean.iterrows())
            ]
        )
        appkit.set_data(replace(appkit.get_data(), brands=brands_df))
        st.session_state["alias_overrides"] = _parse_aliases(alias_text)
        st.success(f"Saved {len(brands_df)} brand(s). Aliases: {st.session_state['alias_overrides'] or 'none'}")

# ---------------------------------------------------------------------------
# Tab 2 — Prompts
# ---------------------------------------------------------------------------
with tab_prompts:
    st.subheader("Prompts")
    st.caption("Add prompts and classify them. Upload a CSV or edit the table directly.")

    up = st.file_uploader("Upload prompts CSV", type=["csv"], key="prompts_csv")
    if up is not None:
        df = _read_csv(up)
        result = validate_prompts(df)
        _show_validation(result)
        if result.ok:
            df = df.reindex(columns=["prompt_id", "project_id", "prompt_text", "prompt_category", "topic", "persona", "journey_stage", "is_brand_prompt"])
            if "project_id" not in df or df["project_id"].isna().all():
                df["project_id"] = "user"
            df["is_brand_prompt"] = _to_bool(df["is_brand_prompt"].fillna("false"))
            appkit.set_data(replace(appkit.get_data(), prompts=df.fillna("")))
            st.success(f"Loaded {len(df)} prompts.")

    prompts_seed = appkit.get_data().prompts
    if prompts_seed.empty:
        prompts_seed = pd.DataFrame(
            [{"prompt_id": "p01", "prompt_text": "", "prompt_category": PROMPT_CATEGORIES[0],
              "topic": "", "persona": "", "journey_stage": JOURNEY_STAGES[0], "is_brand_prompt": False}]
        )
    edited_prompts = st.data_editor(
        prompts_seed.drop(columns=[c for c in ["project_id"] if c in prompts_seed.columns]),
        num_rows="dynamic",
        use_container_width=True,
        key="prompts_editor",
        column_config={
            "prompt_category": st.column_config.SelectboxColumn("Category", options=PROMPT_CATEGORIES),
            "journey_stage": st.column_config.SelectboxColumn("Journey stage", options=JOURNEY_STAGES),
            "is_brand_prompt": st.column_config.CheckboxColumn("Brand prompt?"),
        },
    )
    if st.button("Save prompts"):
        df = edited_prompts.copy()
        df["project_id"] = "user"
        appkit.set_data(replace(appkit.get_data(), prompts=df.fillna("")))
        st.success(f"Saved {len(df)} prompt(s).")

# ---------------------------------------------------------------------------
# Tab 3 — Responses (paste or CSV)
# ---------------------------------------------------------------------------
with tab_responses:
    st.subheader("AI responses")
    st.warning(
        "Only paste responses you collected yourself or via official APIs. Never scrape "
        "ChatGPT, Claude, Gemini, or Perplexity user interfaces. Label synthetic data clearly.",
        icon="⚠️",
    )

    # -- Benchmark Mode: label this batch as Real / User Collected / Synthetic --
    st.markdown("##### Benchmark & dataset label")
    st.caption(
        "Label every batch you add so **real** and **synthetic** results stay separated in every "
        "chart and export. Real = collected from a live platform; Synthetic = generated/example data."
    )
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        dataset_kind = st.selectbox("Dataset type", DATASET_KINDS, index=1,  # default to "Real" for manual input
                                    help="How this batch was obtained. Synthetic data is never shown as real output.")
        benchmark_name = st.text_input("Benchmark name", value="My Benchmark",
                                       help="A name for this collection, e.g. 'July 2026 – ChatGPT'.")
    with bcol2:
        collection_date = st.date_input("Collection date")
        collection_notes = st.text_input("Collection notes (optional)", value="")

    mode = st.radio("Add responses by", ["Upload CSV", "Paste a single response"], horizontal=True)

    if mode == "Upload CSV":
        up = st.file_uploader("Upload responses CSV", type=["csv"], key="responses_csv")
        if up is not None:
            df = _read_csv(up)
            known = set(appkit.get_data().prompts["prompt_id"]) if not appkit.get_data().prompts.empty else None
            result = validate_responses(df, known)
            _show_validation(result)
            if result.ok:
                df = df.reindex(columns=["run_id", "prompt_id", "platform", "model_name", "run_date",
                                         "run_number", "response_text", "has_citations",
                                         "dataset_kind", "benchmark_name", "collection_date", "collection_notes"])
                df["run_number"] = pd.to_numeric(df["run_number"], errors="coerce").fillna(1).astype(int)
                df["has_citations"] = _to_bool(df["has_citations"].fillna("false"))
                # Apply the batch labels where the CSV didn't supply them.
                df["dataset_kind"] = df["dataset_kind"].replace("", pd.NA).fillna(dataset_kind)
                df["benchmark_name"] = df["benchmark_name"].replace("", pd.NA).fillna(benchmark_name)
                df["collection_date"] = df["collection_date"].replace("", pd.NA).fillna(str(collection_date))
                df["collection_notes"] = df["collection_notes"].replace("", pd.NA).fillna(collection_notes)
                appkit.set_data(replace(appkit.get_data(), response_runs=df.fillna("")))
                st.success(f"Loaded {len(df)} responses as '{dataset_kind}'. Go to tab 4 to run extraction.")
    else:
        with st.form("paste_form"):
            prompt_ids = appkit.get_data().prompts["prompt_id"].tolist() if not appkit.get_data().prompts.empty else []
            pid = st.selectbox("Prompt", prompt_ids) if prompt_ids else st.text_input("Prompt ID")
            platform = st.text_input("Platform label", value="ChatGPT (manual)")
            model_name = st.text_input("Model name", value="")
            run_date = st.date_input("Run date")
            run_number = st.number_input("Run number", min_value=1, value=1)
            text = st.text_area("Paste the AI response text", height=200)
            if st.form_submit_button("Add response"):
                cur = appkit.get_data().response_runs
                new_row = {
                    "run_id": f"m{len(cur) + 1:03d}",
                    "prompt_id": pid,
                    "platform": platform,
                    "model_name": model_name,
                    "run_date": str(run_date),
                    "run_number": int(run_number),
                    "response_text": text,
                    "has_citations": "http" in text.lower(),
                    "dataset_kind": dataset_kind,
                    "benchmark_name": benchmark_name,
                    "collection_date": str(collection_date),
                    "collection_notes": collection_notes,
                }
                updated = pd.concat([cur, pd.DataFrame([new_row])], ignore_index=True)
                appkit.set_data(replace(appkit.get_data(), response_runs=updated))
                st.success(f"Response added as '{dataset_kind}'. Go to tab 4 to run extraction.")

    cur_runs = appkit.get_data().response_runs
    if not cur_runs.empty:
        st.caption(f"{len(cur_runs)} response(s) currently loaded.")
        view_cols = [c for c in ["run_id", "prompt_id", "platform", "run_number", "dataset_kind", "benchmark_name"] if c in cur_runs.columns]
        st.dataframe(cur_runs[view_cols], use_container_width=True, height=200)

# ---------------------------------------------------------------------------
# Tab 4 — Review & Correct
# ---------------------------------------------------------------------------
with tab_review:
    st.subheader("Run extraction")
    st.caption("Deterministic extraction finds brand mentions and citation URLs. You can correct the results below.")

    d = appkit.get_data()
    ready = (not d.brands.empty) and (not d.response_runs.empty)
    if not ready:
        st.info("Add at least one brand and one response first (tabs 1 and 3).")
    else:
        if st.button("▶ Run / re-run extraction", type="primary"):
            appkit.set_data(appkit.run_extraction(d, st.session_state.get("alias_overrides")))
            st.success("Extraction complete.")
            st.rerun()

    d = appkit.get_data()
    if not d.brand_mentions.empty:
        st.markdown("**Extracted brand mentions** (editable — correct any mistakes)")
        edited_m = st.data_editor(d.brand_mentions, num_rows="dynamic", use_container_width=True, key="mentions_editor", height=260)
        if st.button("Save mention corrections"):
            appkit.set_data(replace(appkit.get_data(), brand_mentions=edited_m))
            st.success("Mention corrections saved.")

    if not d.citations.empty:
        st.markdown("**Extracted citations** (editable)")
        edited_c = st.data_editor(d.citations, num_rows="dynamic", use_container_width=True, key="citations_editor", height=260)
        if st.button("Save citation corrections"):
            appkit.set_data(replace(appkit.get_data(), citations=edited_c))
            st.success("Citation corrections saved.")

    if not d.brand_entities.empty:
        st.markdown("**Extracted brand entities & narrative** (editable — correct descriptors)")
        st.caption("Category, products, features, personas, strengths, weaknesses, positioning, and competitors mentioned alongside each brand.")
        edited_e = st.data_editor(d.brand_entities, num_rows="dynamic", use_container_width=True, key="entities_editor", height=280)
        if st.button("Save entity corrections"):
            appkit.set_data(replace(appkit.get_data(), brand_entities=edited_e))
            st.success("Entity corrections saved.")

    if not d.brand_mentions.empty:
        st.success("✅ Analysis is ready. Open the **Visibility Dashboard**, **Entity & Narrative Analysis**, and **Content Action Briefs** from the sidebar.")
