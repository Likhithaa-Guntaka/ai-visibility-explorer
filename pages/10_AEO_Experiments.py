"""AEO Experiments — compare a baseline collection against a post-change collection."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import appkit
from src import clusters as C
from src import experiments as X
from src.ui import require_data

st.set_page_config(page_title="AEO Experiments", page_icon="🧪", layout="wide")
appkit.ensure_state()

st.title("🧪 AEO Experiments")
st.caption(
    "Define a before/after experiment, then compare a baseline collection date with a "
    "post-change collection date across the visibility metrics."
)

data = require_data(st)
if data is None:
    st.stop()

if "experiments" not in st.session_state:
    st.session_state["experiments"] = []

st.warning(
    "**Before/after comparisons on AI-search data are associations, not proof of causation.** "
    "Model updates, personalization, and run-to-run noise can move these numbers on their own. "
    "Read the *Confidence and limitations* section with every result.",
    icon="⚠️",
)

# Honesty note whenever the loaded data is the synthetic demo scenario.
if "dataset_kind" in data.response_runs.columns and set(data.response_runs["dataset_kind"].unique()) == {"Synthetic"}:
    st.info(
        "**Synthetic demonstration scenario.** The baseline and post-change waves are synthetic "
        "demonstration data created to show the experiment workflow. They do not represent real "
        "platform responses or evidence of actual brand performance changes.",
        icon="🧪",
    )

dates = X.available_dates(data)
if len(dates) < 2:
    st.info(
        f"Only {len(dates)} collection date found ({', '.join(dates) or 'none'}). An experiment needs "
        "**two** dates: a baseline and a post-change collection. Add a second collection in "
        "**Data Input → Responses**, or load the demo data (which ships with two synthetic waves)."
    )
    st.stop()

# -- Define the experiment ---------------------------------------------------
st.subheader("1 · Define the experiment")
brands = data.brand_names()
prompts = C.prepare_prompts(data.prompts)

with st.form("experiment_form"):
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Experiment name", value="Content push")
        focal = st.selectbox("Focal brand", brands, index=brands.index(appkit.focal_brand()) if appkit.focal_brand() in brands else 0)
        dimension = st.selectbox("Cluster by", C.CLUSTER_DIMENSIONS, format_func=lambda d: C.DIMENSION_LABELS.get(d, d))
        cluster_options = ["(All questions)"] + sorted(prompts[dimension].dropna().unique().tolist()) if dimension in prompts.columns else ["(All questions)"]
        cluster_choice = st.selectbox("Question cluster", cluster_options)
    with c2:
        baseline_date = st.selectbox("Baseline collection date", dates, index=0)
        post_date = st.selectbox("Post-change collection date", dates, index=len(dates) - 1)
        primary_kpi = st.selectbox("Primary KPI", X.KPI_OPTIONS, index=0)
        secondary_kpis = st.multiselect("Secondary KPIs", [k for k in X.KPI_OPTIONS], default=["Share of voice", "Citation rate"])

    change_made = st.text_area("Content or technical change made", height=70,
                               placeholder="e.g. Published a comparison page and added FAQ schema to 3 pages.")
    hypothesis = st.text_area("Hypothesis", height=70,
                              placeholder="e.g. Adding a comparison page will increase mentions for purchase-intent questions.")
    submitted = st.form_submit_button("Run comparison", type="primary")

if submitted:
    st.session_state["experiments"] = [
        X.Experiment(
            name=name, focal_brand=focal, baseline_date=baseline_date, post_date=post_date,
            cluster_dimension=dimension,
            cluster_value=None if cluster_choice == "(All questions)" else cluster_choice,
            change_made=change_made, hypothesis=hypothesis,
            primary_kpi=primary_kpi, secondary_kpis=secondary_kpis,
        )
    ]

if not st.session_state["experiments"]:
    st.info("Define an experiment above and click **Run comparison**.")
    st.stop()

exp: X.Experiment = st.session_state["experiments"][-1]
result = X.compare_experiment(data, exp)

# -- Results -----------------------------------------------------------------
st.divider()
st.subheader(f"2 · Results — {exp.name}")
st.caption(
    f"Focal brand **{exp.focal_brand}** · scope **{exp.cluster_value or 'all questions'}** · "
    f"baseline **{exp.baseline_date}** vs post **{exp.post_date}**"
)
if exp.change_made:
    st.markdown(f"**Change made:** {exp.change_made}")
if exp.hypothesis:
    st.markdown(f"**Hypothesis:** {exp.hypothesis}")

s1, s2 = st.columns(2)
s1.metric("Baseline responses", result["baseline_n"])
s2.metric("Post-change responses", result["post_n"])

comp = result["comparison"]
if result["baseline_n"] == 0 or result["post_n"] == 0:
    st.error("One arm has no responses — pick two dates that both contain collected responses.")
else:
    # Headline: primary KPI.
    primary = comp[comp["is_primary"]]
    if not primary.empty:
        p = primary.iloc[0]
        delta = X.format_change(p)
        st.metric(
            f"Primary KPI — {p['metric']}",
            (f"{p['post']*100:.0f}%" if p["unit"] == "rate" and p["post"] is not None else f"{p['post']:.0f}"),
            delta=delta,
            help="Change observed alongside the described change — not proof it caused it.",
        )

    st.markdown("##### Metric comparison (baseline → post-change)")
    show = comp.copy()

    def _fmt(v, unit):
        if v is None or pd.isna(v):
            return "n/a"
        return f"{v*100:.0f}%" if unit == "rate" else f"{v:.0f}"

    show["Baseline"] = [_fmt(v, u) for v, u in zip(show["baseline"], show["unit"])]
    show["Post-change"] = [_fmt(v, u) for v, u in zip(show["post"], show["unit"])]
    show["Absolute change"] = [
        ("n/a" if a is None or pd.isna(a) else (f"{a*100:+.1f} pp" if u == "rate" else f"{a:+.0f}"))
        for a, u in zip(show["absolute_change"], show["unit"])
    ]
    show["Percentage-point change"] = [
        ("n/a" if pp is None or pd.isna(pp) else f"{pp:+.1f} pp") for pp in show["pp_change"]
    ]
    show["Role"] = [
        "Primary" if pr else ("Secondary" if sec else "") for pr, sec in zip(show["is_primary"], show["is_secondary"])
    ]
    st.dataframe(
        show[["metric", "Baseline", "Post-change", "Absolute change", "Percentage-point change", "baseline_n", "post_n", "Role"]]
        .rename(columns={"metric": "Metric", "baseline_n": "Baseline n", "post_n": "Post n"}),
        use_container_width=True, hide_index=True,
    )

    rates = comp[(comp["unit"] == "rate") & comp["pp_change"].notna()]
    if not rates.empty:
        fig = px.bar(rates, x="pp_change", y="metric", orientation="h",
                     labels={"pp_change": "Percentage-point change", "metric": ""})
        fig.update_traces(marker_color="#2563eb")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Platform-level.
    st.markdown("##### Platform-level results")
    plat = result["platform"]
    if plat.empty:
        st.write("— no platform data —")
    else:
        p = plat.copy()
        p["baseline_rate"] = (p["baseline_rate"] * 100).round(0).astype(int).astype(str) + "%"
        p["post_rate"] = (p["post_rate"] * 100).round(0).astype(int).astype(str) + "%"
        p["pp_change"] = p["pp_change"].round(1).astype(str) + " pp"
        p.columns = ["Platform", "Baseline", "Post-change", "Change", "Baseline n", "Post n"]
        st.dataframe(p, use_container_width=True, hide_index=True)

    # Prompt-level.
    st.markdown("##### Prompt-level changes")
    pl = result["prompt_level"]
    if pl.empty:
        st.write("— no prompt data —")
    else:
        d = pl.copy()
        d["baseline_rate"] = (d["baseline_rate"] * 100).round(0).astype(int).astype(str) + "%"
        d["post_rate"] = (d["post_rate"] * 100).round(0).astype(int).astype(str) + "%"
        d["pp_change"] = d["pp_change"].round(1).astype(str) + " pp"
        d.columns = ["ID", "Question", "Baseline", "Post-change", "Change", "Baseline n", "Post n"]
        st.dataframe(d, use_container_width=True, hide_index=True, height=320)

# -- Confidence & limitations ------------------------------------------------
st.divider()
st.subheader("3 · Confidence and limitations")
for item in result["limitations"]:
    st.markdown(f"- {item}")

st.caption(
    "Language guide: say “mention rate rose 8 pp **alongside** the content change”, not “the "
    "content change **caused** an 8 pp rise”. Only a controlled design with a holdout of "
    "untouched questions would support a causal claim."
)
