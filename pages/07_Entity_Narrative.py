"""Entity & Narrative Analysis — how AI answers describe the brand, and how consistently."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src import appkit
from src import entities as E
from src.ui import require_data, sidebar_filters

st.set_page_config(page_title="Entity & Narrative Analysis", page_icon="💬", layout="wide")
appkit.ensure_state()

st.title("💬 Entity & Narrative Analysis")
st.caption(
    "What AI answers say about the brand — its category, products, features, personas, "
    "strengths, weaknesses, and positioning — extracted deterministically and editable."
)

data = require_data(st)
if data is None:
    st.stop()

data = sidebar_filters(st, data)
focal = appkit.focal_brand()

ent = data.brand_entities[data.brand_entities["brand_name"] == focal] if not data.brand_entities.empty else data.brand_entities
if ent.empty:
    st.info(f"No entity data for {focal} in the current view. Widen the filters or run extraction.")
    st.stop()

st.info(
    "Extraction is transparent (keyword lexicons) and **editable** in **Data Input → Review & "
    "Correct**. Descriptors are associations found in the text, not verified facts.",
    icon="🔎",
)

# -- 1. How each platform describes the brand -------------------------------
st.subheader(f"How each platform describes {focal}")
field = st.selectbox(
    "Descriptor field",
    ["strengths", "weaknesses", "features", "pricing_positioning", "personas", "products", "brand_category"],
    index=0,
)
plat = E.platform_descriptions(data.brand_entities, data.response_runs, focal, field)
st.dataframe(plat, width="stretch", hide_index=True)

# -- 2. Common descriptors ---------------------------------------------------
st.subheader("Common descriptors")
st.caption("Descriptors used in at least 40% of the brand's responses (aggregated across fields).")
common = E.common_descriptors(data.brand_entities, focal, min_share=0.4)
if not common.empty:
    fig = px.bar(common, x="share", y="descriptor", color="field", orientation="h",
                 labels={"share": "Share of responses", "descriptor": ""})
    fig.update_layout(xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"}, height=360)
    st.plotly_chart(fig, width="stretch")
else:
    st.write("No descriptor reaches the 40% threshold in this view.")

# -- 3. Conflicting descriptions --------------------------------------------
st.subheader("Conflicting descriptions")
st.caption("Cases where the brand is described with opposing attributes across responses.")
conflicts = E.conflicting_descriptions(data.brand_entities, focal)
if not conflicts.empty:
    show = conflicts.rename(columns={
        "descriptor_a": "Attribute A", "descriptor_b": "Attribute B",
        "count_a": "A count", "count_b": "B count",
    })
    st.dataframe(show, width="stretch", hide_index=True)
    st.warning("Mixed messages like these can confuse how AI answers position the brand.", icon="🔀")
else:
    st.success("No opposing-attribute conflicts detected in this view.")

# -- 4. Missing / inconsistent attributes -----------------------------------
st.subheader("Attribute coverage (missing or inconsistent attributes)")
cov = E.attribute_coverage(data.brand_entities, focal)
if not cov.empty:
    show = cov.copy()
    show["coverage"] = (show["coverage"] * 100).round(0).astype(int).astype(str) + "%"
    show.columns = ["Attribute", "Responses with value", "Total responses", "Coverage"]
    st.dataframe(show, width="stretch", hide_index=True)
    low = cov[cov["coverage"] < 0.25]["attribute"].tolist()
    if low:
        st.info("Low-coverage attributes (rarely described): " + ", ".join(low) +
                ". These are candidates to clarify in your own content.")

# -- 5. Narrative consistency across platforms and runs ---------------------
st.subheader("Narrative consistency")
st.caption("Mean overlap of descriptor sets across the brand's runs (1.0 = identical every time).")
rows = []
for b in data.brand_names():
    nc = E.narrative_consistency(data.brand_entities, data.response_runs, b)
    rows.append({"brand": b, "runs": nc["runs"], "consistency": nc["consistency"]})
import pandas as pd
cons_df = pd.DataFrame(rows)
cons_df["consistency_display"] = cons_df["consistency"].apply(lambda v: f"{round(v*100)}%" if v is not None else "n/a")
st.dataframe(
    cons_df[["brand", "runs", "consistency_display"]].rename(
        columns={"brand": "Brand", "runs": "Runs", "consistency_display": "Narrative consistency"}
    ),
    width="stretch", hide_index=True,
)
focal_nc = next((r for r in rows if r["brand"] == focal), None)
if focal_nc and focal_nc["consistency"] is not None and focal_nc["consistency"] < 0.7:
    st.warning(
        f"{focal}'s narrative varies across runs ({round(focal_nc['consistency']*100)}% overlap) — "
        "a single response should not be treated as the definitive description."
    )
