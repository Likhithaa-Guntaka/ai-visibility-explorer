"""Deterministic entity & narrative extraction (feature: Entity and Narrative Analysis).

For every (response, brand) pair we extract *how the answer describes the brand* using
transparent keyword lexicons: category, products, features, personas, strengths,
weaknesses, pricing/positioning language, and which competitors are named alongside it.

Everything here is rule-based and editable — no LLM, no hidden logic. Each descriptor
maps back to a concrete phrase in a controlled lexicon, so a user can see and correct
exactly why a descriptor was assigned. The lexicons live in one place so they are easy
to review and extend.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .database import BRAND_ENTITIES_COLUMNS
from .extraction import _iter_term_matches, _unique_terms

# ---------------------------------------------------------------------------
# Controlled lexicons: canonical label -> surface phrases to match (lowercased).
# ---------------------------------------------------------------------------

CATEGORY_LEXICON: dict[str, list[str]] = {
    "All-in-one workspace": ["all-in-one workspace", "workspace"],
    "Project management platform": ["project management platform", "project management"],
    "Task management tool": ["task management tool", "task management"],
    "Kanban tool": ["kanban tool", "kanban-board", "kanban board", "kanban"],
}

# "Products" here means named modules/sub-products (often absent — which is a useful,
# honest signal of a missing attribute).
PRODUCT_LEXICON: dict[str, list[str]] = {
    "Docs": ["docs", "documents"],
    "Wikis": ["wiki", "wikis"],
    "Databases": ["database", "databases"],
    "Boards": ["boards", "board"],
    "Dashboards": ["dashboard", "dashboards"],
    "Goals": ["goal", "goals"],
    "Timelines": ["timeline", "timelines", "gantt"],
    "Automations": ["automation", "automations"],
}

FEATURE_LEXICON: dict[str, list[str]] = {
    "Task tracking": ["task tracking", "task management", "tasks"],
    "Timelines": ["timeline", "timelines", "gantt"],
    "Automations": ["automation", "automations"],
    "Dashboards": ["dashboard", "dashboards"],
    "Kanban boards": ["kanban", "boards"],
    "Docs": ["docs", "documents"],
    "Wikis": ["wiki", "wikis"],
    "Databases": ["database", "databases"],
    "Goals": ["goal", "goals"],
    "Workload management": ["workload"],
    "Reporting": ["reporting", "reports"],
    "Templates": ["template", "templates"],
}

PERSONA_LEXICON: dict[str, list[str]] = {
    "Non-technical teams": ["non-technical", "non technical"],
    "Large teams": ["larger teams", "large teams", "enterprise"],
    "Startups / small teams": ["startup", "startups", "small team", "small teams"],
    "Beginners": ["beginner", "beginners"],
    "Remote teams": ["remote team", "remote teams"],
    "Agencies": ["agency", "agencies"],
    "Freelancers": ["freelancer", "freelancers"],
    "Marketing teams": ["marketing team", "marketing teams"],
    "Engineering teams": ["engineering team", "engineering teams", "developers"],
}

STRENGTH_LEXICON: dict[str, list[str]] = {
    "Flexible": ["flexible", "flexibility"],
    "Powerful": ["powerful", "feature-rich", "feature rich"],
    "Reliable": ["reliable", "polished", "mature"],
    "Easy to use": ["easy to use", "easy to learn", "intuitive", "approachable", "simple", "lightweight"],
    "Customizable": ["customizable"],
    "Affordable": ["affordable"],
    "Generous free plan": ["generous free", "free plan"],
    "All-in-one": ["all-in-one", "all in one"],
}

WEAKNESS_LEXICON: dict[str, list[str]] = {
    "Overwhelming": ["overwhelming"],
    "Expensive": ["expensive", "gets expensive", "pricey", "can get pricey"],
    "Cluttered interface": ["cluttered"],
    "Steep learning curve": ["steep", "learning curve"],
    "Limited for complex work": ["limited for complex", "feels limited", "limited"],
    "Lacks advanced reporting": ["lacks advanced reporting", "reporting is limited", "lacks"],
}

POSITIONING_LEXICON: dict[str, list[str]] = {
    "All-in-one": ["all-in-one", "all in one"],
    "Free plan": ["free plan"],
    "Affordable": ["affordable"],
    "Premium / expensive": ["expensive", "pricey"],
    "Lightweight / simple": ["lightweight", "simple", "easy to learn"],
}

# Conflicting descriptor pairs. If a brand is described with BOTH members across its
# runs, that is a narrative conflict worth surfacing (not necessarily an error).
CONFLICT_PAIRS: list[tuple[str, str]] = [
    ("Affordable", "Premium / expensive"),
    ("Affordable", "Expensive"),
    ("Easy to use", "Steep learning curve"),
    ("Easy to use", "Overwhelming"),
    ("Lightweight / simple", "Overwhelming"),
    ("Free plan", "Premium / expensive"),
]

# Which entity fields are "narrative descriptor" fields (for consistency/coverage).
DESCRIPTOR_FIELDS: list[str] = [
    "brand_category", "products", "features", "personas",
    "strengths", "weaknesses", "pricing_positioning",
]

_SEP = "; "


# ---------------------------------------------------------------------------
# Core extraction.
# ---------------------------------------------------------------------------


def _match_lexicon(context: str, lexicon: dict[str, list[str]]) -> list[str]:
    """Return canonical labels whose phrases appear in ``context`` (lowercased)."""
    hits: list[str] = []
    for label, phrases in lexicon.items():
        for phrase in phrases:
            if next(_iter_term_matches(context, phrase), None) is not None:
                hits.append(label)
                break
    return hits


def _brand_context(text: str, brand_terms: list[str]) -> str:
    """Extract the text *blocks* that mention the brand, joined together.

    We split on line breaks (and bullets) rather than sentences, because AI answers
    typically describe each brand in one block/list-item where the brand is named once
    and later clauses use pronouns ("It is affordable, but it can get pricey"). Keeping
    the whole block preserves those pronoun clauses while still attributing them to the
    right brand, since brand blocks rarely name two brands.
    """
    blocks = re.split(r"\n+", text)
    lowered_terms = [t.lower() for t in brand_terms]
    kept = []
    for block in blocks:
        low = block.lower()
        if any(next(_iter_term_matches(low, t), None) is not None for t in lowered_terms):
            kept.append(block)
    return " ".join(kept).lower()


def extract_entities_run(
    run_id: str,
    response_text: str,
    focus_brand: str,
    focus_terms: list[str],
    all_brand_terms: dict[str, list[str]],
) -> Optional[dict]:
    """Extract narrative entities for one brand in one response.

    Returns a dict matching ``BRAND_ENTITIES_COLUMNS`` (multi-valued fields joined by
    "; "), or ``None`` if the brand is not mentioned in the response at all.
    """
    text = response_text or ""
    context = _brand_context(text, focus_terms)
    if not context:
        return None

    competitors = [
        other
        for other, terms in all_brand_terms.items()
        if other != focus_brand
        and any(next(_iter_term_matches(text.lower(), t.lower()), None) is not None for t in terms)
    ]

    return {
        "run_id": run_id,
        "brand_name": focus_brand,
        "brand_category": _SEP.join(_match_lexicon(context, CATEGORY_LEXICON)),
        "products": _SEP.join(_match_lexicon(context, PRODUCT_LEXICON)),
        "features": _SEP.join(_match_lexicon(context, FEATURE_LEXICON)),
        "personas": _SEP.join(_match_lexicon(context, PERSONA_LEXICON)),
        "strengths": _SEP.join(_match_lexicon(context, STRENGTH_LEXICON)),
        "weaknesses": _SEP.join(_match_lexicon(context, WEAKNESS_LEXICON)),
        "pricing_positioning": _SEP.join(_match_lexicon(context, POSITIONING_LEXICON)),
        "competitors_alongside": _SEP.join(sorted(competitors)),
    }


def extract_all_entities(
    response_runs: pd.DataFrame,
    brand_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    """Extract narrative entities for every (response, mentioned brand) pair.

    ``brand_aliases`` maps canonical brand -> aliases (as produced by
    :func:`src.extraction.build_alias_map`). Returns a DataFrame matching
    ``BRAND_ENTITIES_COLUMNS``.
    """
    all_terms = {brand: _unique_terms([brand, *aliases]) for brand, aliases in brand_aliases.items()}
    rows: list[dict] = []
    for _, run in response_runs.iterrows():
        run_id = str(run["run_id"])
        text = str(run.get("response_text", ""))
        for brand, terms in all_terms.items():
            row = extract_entities_run(run_id, text, brand, terms, all_terms)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows, columns=BRAND_ENTITIES_COLUMNS)


# ---------------------------------------------------------------------------
# Narrative analysis helpers.
# ---------------------------------------------------------------------------


def _split(value: object) -> list[str]:
    """Split a "; "-joined entity field into a clean list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    return [p.strip() for p in s.split(";") if p.strip()] if s else []


def brand_entities_with_context(
    brand_entities: pd.DataFrame, response_runs: pd.DataFrame
) -> pd.DataFrame:
    """Attach platform / run_number to entity rows for per-platform analysis."""
    if brand_entities.empty:
        return brand_entities.assign(platform=pd.Series(dtype=str), run_number=pd.Series(dtype="Int64"))
    ctx = response_runs[["run_id", "platform", "run_number"]]
    return brand_entities.merge(ctx, on="run_id", how="left")


def descriptor_frequency(
    brand_entities: pd.DataFrame, brand_name: str, field_name: str
) -> pd.DataFrame:
    """Count how often each descriptor value appears for a brand in a given field.

    Returns columns: ``descriptor``, ``count``, ``share`` (share of runs mentioning it).
    """
    cols = ["descriptor", "count", "share"]
    if brand_entities.empty or field_name not in brand_entities.columns:
        return pd.DataFrame(columns=cols)
    sub = brand_entities[brand_entities["brand_name"] == brand_name]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    counts: dict[str, int] = {}
    for val in sub[field_name]:
        for d in _split(val):
            counts[d] = counts.get(d, 0) + 1
    n = len(sub)
    rows = [{"descriptor": k, "count": v, "share": v / n} for k, v in counts.items()]
    return pd.DataFrame(rows, columns=cols).sort_values("count", ascending=False).reset_index(drop=True)


def platform_descriptions(
    brand_entities: pd.DataFrame, response_runs: pd.DataFrame, brand_name: str, field_name: str = "strengths"
) -> pd.DataFrame:
    """For each platform, the descriptors used for a brand in a given field.

    Returns columns: ``platform``, ``descriptors`` (sorted unique list joined by ", "),
    ``runs``.
    """
    cols = ["platform", "descriptors", "runs"]
    ctx = brand_entities_with_context(brand_entities, response_runs)
    if ctx.empty:
        return pd.DataFrame(columns=cols)
    sub = ctx[ctx["brand_name"] == brand_name]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for platform, grp in sub.groupby("platform"):
        descriptors = sorted({d for val in grp[field_name] for d in _split(val)})
        rows.append({"platform": platform, "descriptors": ", ".join(descriptors) or "—", "runs": int(len(grp))})
    return pd.DataFrame(rows, columns=cols)


def common_descriptors(brand_entities: pd.DataFrame, brand_name: str, min_share: float = 0.5) -> pd.DataFrame:
    """Descriptors used consistently (in >= ``min_share`` of the brand's runs).

    Aggregates strengths + features + positioning. Returns columns:
    ``descriptor``, ``field``, ``count``, ``share``.
    """
    frames = []
    for field_name in ["strengths", "features", "pricing_positioning", "personas"]:
        f = descriptor_frequency(brand_entities, brand_name, field_name)
        if not f.empty:
            f = f.assign(field=field_name)
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["descriptor", "field", "count", "share"])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["share"] >= min_share]
    return out[["descriptor", "field", "count", "share"]].sort_values("share", ascending=False).reset_index(drop=True)


def conflicting_descriptions(brand_entities: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    """Detect narrative conflicts: both members of a CONFLICT_PAIR appear for the brand.

    Returns columns: ``descriptor_a``, ``descriptor_b``, ``count_a``, ``count_b``.
    """
    cols = ["descriptor_a", "descriptor_b", "count_a", "count_b"]
    if brand_entities.empty:
        return pd.DataFrame(columns=cols)
    sub = brand_entities[brand_entities["brand_name"] == brand_name]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    # Tally all descriptors across the descriptor fields.
    tally: dict[str, int] = {}
    for field_name in ["strengths", "weaknesses", "pricing_positioning"]:
        for val in sub[field_name]:
            for d in _split(val):
                tally[d] = tally.get(d, 0) + 1
    rows = []
    for a, b in CONFLICT_PAIRS:
        if a in tally and b in tally:
            rows.append({"descriptor_a": a, "descriptor_b": b, "count_a": tally[a], "count_b": tally[b]})
    return pd.DataFrame(rows, columns=cols)


def attribute_coverage(brand_entities: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    """Which entity fields are populated vs missing for a brand.

    Returns columns: ``attribute``, ``runs_with_value``, ``total_runs``, ``coverage``.
    A low coverage flags a *missing or inconsistent* brand attribute.
    """
    cols = ["attribute", "runs_with_value", "total_runs", "coverage"]
    if brand_entities.empty:
        return pd.DataFrame(columns=cols)
    sub = brand_entities[brand_entities["brand_name"] == brand_name]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    n = len(sub)
    rows = []
    for field_name in DESCRIPTOR_FIELDS:
        with_val = int(sub[field_name].apply(lambda v: len(_split(v)) > 0).sum())
        rows.append(
            {
                "attribute": field_name,
                "runs_with_value": with_val,
                "total_runs": n,
                "coverage": with_val / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values("coverage").reset_index(drop=True)


def narrative_consistency(
    brand_entities: pd.DataFrame, response_runs: pd.DataFrame, brand_name: str
) -> dict[str, Optional[float]]:
    """Mean pairwise Jaccard overlap of descriptor sets across the brand's runs.

    Combines strengths + features + positioning into one descriptor set per run, then
    averages pairwise Jaccard. 1.0 = every run describes the brand identically; lower
    means the narrative varies run-to-run / platform-to-platform. Returns a dict with
    ``brand``, ``runs``, ``consistency`` (None if < 2 runs).
    """
    from itertools import combinations

    if brand_entities.empty:
        return {"brand": brand_name, "runs": 0, "consistency": None}
    sub = brand_entities[brand_entities["brand_name"] == brand_name]
    if len(sub) < 2:
        return {"brand": brand_name, "runs": int(len(sub)), "consistency": None}

    sets = []
    for _, row in sub.iterrows():
        desc = set()
        for field_name in ["strengths", "features", "pricing_positioning"]:
            desc.update(_split(row[field_name]))
        sets.append(desc)

    scores = []
    for a, b in combinations(sets, 2):
        union = a | b
        scores.append(len(a & b) / len(union) if union else 1.0)
    return {
        "brand": brand_name,
        "runs": int(len(sub)),
        "consistency": float(sum(scores) / len(scores)) if scores else None,
    }
