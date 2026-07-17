"""Before/after AEO experiments (feature: Before and After AEO Experiments).

Compares a *baseline* collection against a *post-change* collection for one focal brand
and (optionally) one question cluster, across the project's visibility metrics.

Interpretation discipline
-------------------------
A before/after comparison on observational AI-search data is an **association**, not
proof of causation. Model updates, personalization, prompt-set changes, and ordinary
run-to-run noise can all move these numbers on their own. :func:`experiment_limitations`
returns the caveats that must be shown alongside every result, and the language used
throughout says "changed alongside", never "caused".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import entities as ENT
from . import metrics as M
from .clusters import prepare_prompts
from .database import AnalysisData

# KPI choices offered in the UI (primary + secondary).
KPI_OPTIONS: list[str] = [
    "Brand mention rate",
    "Share of voice",
    "First mention share",
    "Recommendation rate",
    "Citation rate",
    "Source coverage",
    "Narrative consistency",
    "Top competitor share of voice",
]

# Metrics reported as 0-1 rates (so a percentage-point change is meaningful).
_RATE_METRICS = {
    "Brand mention rate", "Share of voice", "First mention share", "Recommendation rate",
    "Citation rate", "Narrative consistency", "Top competitor share of voice",
}

# Below this many responses in either arm, the comparison is too fragile to lean on.
MIN_ARM_SAMPLE = 10


@dataclass
class Experiment:
    """A user-defined before/after AEO experiment."""

    name: str
    focal_brand: str
    baseline_date: str
    post_date: str
    cluster_dimension: str = "question_cluster"
    cluster_value: Optional[str] = None  # None = all questions
    change_made: str = ""
    hypothesis: str = ""
    primary_kpi: str = "Brand mention rate"
    secondary_kpis: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Slicing helpers.
# ---------------------------------------------------------------------------


def _run_dates(response_runs: pd.DataFrame) -> pd.Series:
    """Effective collection date per run: ``run_date``, falling back to ``collection_date``."""
    if response_runs.empty:
        return pd.Series(dtype=str)
    dates = response_runs["run_date"].fillna("").astype(str) if "run_date" in response_runs.columns else pd.Series([""] * len(response_runs), index=response_runs.index)
    if "collection_date" in response_runs.columns:
        fallback = response_runs["collection_date"].fillna("").astype(str)
        dates = dates.where(dates.str.strip() != "", fallback)
    return dates


def available_dates(data: AnalysisData) -> list[str]:
    """Sorted list of collection dates present in the data."""
    dates = _run_dates(data.response_runs)
    return sorted({d for d in dates.tolist() if str(d).strip()})


def slice_arm(data: AnalysisData, date: str, dimension: str = "question_cluster",
              cluster_value: Optional[str] = None) -> AnalysisData:
    """Return the analysis restricted to one collection date and (optionally) one cluster."""
    from dataclasses import replace

    prompts = prepare_prompts(data.prompts)
    if cluster_value is not None and dimension in prompts.columns:
        prompts = prompts[prompts[dimension] == cluster_value]
    allowed_pids = set(prompts["prompt_id"]) if not prompts.empty else set()

    runs = data.response_runs
    if not runs.empty:
        mask = (_run_dates(runs) == str(date)) & runs["prompt_id"].isin(allowed_pids)
        runs = runs[mask]
    run_ids = set(runs["run_id"]) if not runs.empty else set()

    def _sub(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["run_id"].isin(run_ids)].reset_index(drop=True) if not df.empty else df

    return replace(
        data,
        prompts=prompts.reset_index(drop=True),
        response_runs=runs.reset_index(drop=True),
        brand_mentions=_sub(data.brand_mentions),
        citations=_sub(data.citations),
        brand_entities=_sub(data.brand_entities),
    )


# ---------------------------------------------------------------------------
# Metric computation for one arm.
# ---------------------------------------------------------------------------


def arm_metrics(arm: AnalysisData, focal_brand: str) -> dict[str, Optional[float]]:
    """Compute every comparable metric for one arm of the experiment."""
    n = M.total_runs(arm.response_runs)
    if n == 0:
        return {k: None for k in KPI_OPTIONS}

    mentions, runs = arm.brand_mentions, arm.response_runs
    mr = M.brand_mention_rate(mentions, runs)
    sov = M.share_of_voice(mentions)
    fms = M.first_mention_share(mentions)
    rec = M.recommendation_rate(mentions, runs)
    cite = M.citation_rate(arm.citations, runs)

    def _val(df: pd.DataFrame, col: str, brand: str) -> float:
        if df.empty or brand not in set(df["brand_name"]):
            return 0.0
        return float(df[df["brand_name"] == brand][col].iloc[0])

    # Top competitor by share of voice (competitor visibility).
    comp_sov = 0.0
    if not sov.empty:
        comp = sov[sov["brand_name"] != focal_brand]
        if not comp.empty:
            comp_sov = float(comp["share_of_voice"].max())

    nc = ENT.narrative_consistency(arm.brand_entities, runs, focal_brand)["consistency"]

    return {
        "Brand mention rate": _val(mr, "mention_rate", focal_brand),
        "Share of voice": _val(sov, "share_of_voice", focal_brand),
        "First mention share": _val(fms, "first_mention_share", focal_brand),
        "Recommendation rate": _val(rec, "recommendation_rate", focal_brand),
        "Citation rate": cite["citation_rate"],
        # Source coverage = breadth of distinct cited domains in the arm.
        "Source coverage": float(arm.citations["citation_domain"].nunique()) if not arm.citations.empty else 0.0,
        "Narrative consistency": nc,
        "Top competitor share of voice": comp_sov,
    }


# ---------------------------------------------------------------------------
# Comparison.
# ---------------------------------------------------------------------------


def compare_experiment(data: AnalysisData, exp: Experiment) -> dict:
    """Compare baseline vs post-change arms for an experiment.

    Returns ``comparison`` (DataFrame), ``baseline_n``/``post_n``, ``platform`` and
    ``prompt_level`` DataFrames, and ``limitations`` (list of strings). All language is
    associational — see the module docstring.
    """
    base = slice_arm(data, exp.baseline_date, exp.cluster_dimension, exp.cluster_value)
    post = slice_arm(data, exp.post_date, exp.cluster_dimension, exp.cluster_value)
    base_n, post_n = M.total_runs(base.response_runs), M.total_runs(post.response_runs)

    bm, pm = arm_metrics(base, exp.focal_brand), arm_metrics(post, exp.focal_brand)

    rows = []
    for metric in KPI_OPTIONS:
        b, p = bm.get(metric), pm.get(metric)
        is_rate = metric in _RATE_METRICS
        abs_change = (p - b) if (b is not None and p is not None) else None
        pp_change = (abs_change * 100) if (abs_change is not None and is_rate) else None
        rows.append({
            "metric": metric,
            "unit": "rate" if is_rate else "count",
            "baseline": b,
            "post": p,
            "absolute_change": abs_change,
            "pp_change": pp_change,
            "baseline_n": base_n,
            "post_n": post_n,
            "is_primary": metric == exp.primary_kpi,
            "is_secondary": metric in exp.secondary_kpis,
        })
    comparison = pd.DataFrame(rows)

    return {
        "comparison": comparison,
        "baseline_n": base_n,
        "post_n": post_n,
        "platform": platform_level_changes(base, post, exp.focal_brand),
        "prompt_level": prompt_level_changes(base, post, exp.focal_brand),
        "limitations": experiment_limitations(exp, base_n, post_n),
    }


def platform_level_changes(base: AnalysisData, post: AnalysisData, focal_brand: str) -> pd.DataFrame:
    """Focal-brand mention rate per platform, baseline vs post.

    Returns columns: ``platform``, ``baseline_rate``, ``post_rate``, ``pp_change``,
    ``baseline_n``, ``post_n``.
    """
    cols = ["platform", "baseline_rate", "post_rate", "pp_change", "baseline_n", "post_n"]

    def _rates(arm: AnalysisData) -> pd.DataFrame:
        enriched = M.enrich_mentions(arm.brand_mentions, arm.response_runs, arm.prompts)
        return M.platform_comparison(enriched, arm.response_runs, focal_brand)

    b, p = _rates(base), _rates(post)
    if b.empty and p.empty:
        return pd.DataFrame(columns=cols)
    merged = b.merge(p, on="platform", how="outer", suffixes=("_base", "_post")).fillna(0)
    merged["pp_change"] = (merged["mention_rate_post"] - merged["mention_rate_base"]) * 100
    out = merged.rename(columns={
        "mention_rate_base": "baseline_rate", "mention_rate_post": "post_rate",
        "total_runs_base": "baseline_n", "total_runs_post": "post_n",
    })
    return out.reindex(columns=cols).sort_values("pp_change", ascending=False).reset_index(drop=True)


def prompt_level_changes(base: AnalysisData, post: AnalysisData, focal_brand: str) -> pd.DataFrame:
    """Focal-brand mention rate per prompt, baseline vs post.

    Returns columns: ``prompt_id``, ``prompt_text``, ``baseline_rate``, ``post_rate``,
    ``pp_change``, ``baseline_n``, ``post_n``.
    """
    cols = ["prompt_id", "prompt_text", "baseline_rate", "post_rate", "pp_change", "baseline_n", "post_n"]

    def _per_prompt(arm: AnalysisData) -> dict[str, tuple[float, int]]:
        out: dict[str, tuple[float, int]] = {}
        if arm.response_runs.empty:
            return out
        for pid, runs in arm.response_runs.groupby("prompt_id"):
            n = len(runs)
            run_ids = set(runs["run_id"])
            mentions = arm.brand_mentions[arm.brand_mentions["run_id"].isin(run_ids)] if not arm.brand_mentions.empty else arm.brand_mentions
            hits = mentions[mentions["brand_name"] == focal_brand]["run_id"].nunique() if not mentions.empty else 0
            out[str(pid)] = (hits / n if n else 0.0, n)
        return out

    b, p = _per_prompt(base), _per_prompt(post)
    texts = {}
    for arm in (base, post):
        if not arm.prompts.empty:
            texts.update(dict(zip(arm.prompts["prompt_id"].astype(str), arm.prompts["prompt_text"].astype(str))))

    rows = []
    for pid in sorted(set(b) | set(p)):
        b_rate, b_n = b.get(pid, (0.0, 0))
        p_rate, p_n = p.get(pid, (0.0, 0))
        rows.append({
            "prompt_id": pid,
            "prompt_text": texts.get(pid, ""),
            "baseline_rate": b_rate,
            "post_rate": p_rate,
            "pp_change": (p_rate - b_rate) * 100,
            "baseline_n": b_n,
            "post_n": p_n,
        })
    return pd.DataFrame(rows, columns=cols).sort_values("pp_change", ascending=False).reset_index(drop=True)


def experiment_limitations(exp: Experiment, baseline_n: int, post_n: int) -> list[str]:
    """The caveats that must accompany every experiment result."""
    out = [
        "This is a **before/after observational comparison, not a controlled experiment**. "
        "Any movement is an association between the change and the metric — it is not proof "
        "that the change caused the movement.",
        "Confounders that can move these numbers on their own: model/platform updates between "
        "the two dates, personalization, ordinary run-to-run variation, and any difference in "
        "the prompt set or collection method.",
    ]
    if baseline_n == 0 or post_n == 0:
        out.append(
            f"⚠ One arm has no responses (baseline n={baseline_n}, post n={post_n}). "
            "Pick dates that both contain collected responses."
        )
    elif baseline_n < MIN_ARM_SAMPLE or post_n < MIN_ARM_SAMPLE:
        out.append(
            f"⚠ Small sample (baseline n={baseline_n}, post n={post_n}; fewer than {MIN_ARM_SAMPLE} in an arm). "
            "A single response can swing a percentage several points — treat this as directional only."
        )
    out.append(
        "To support a stronger claim you would need: more responses per arm, repeated runs per prompt, "
        "an unchanged prompt set, a comparison/holdout group of untouched questions, and ideally repeated "
        "measurement over time rather than two single snapshots."
    )
    if exp.cluster_value:
        out.append(
            f"Scope: only questions in the '{exp.cluster_value}' cluster were compared, so results do not "
            "generalise to the whole prompt set."
        )
    return out


def format_change(row: pd.Series) -> str:
    """Human-readable change string for one comparison row (associational language)."""
    if row["baseline"] is None or row["post"] is None:
        return "n/a"
    if row["unit"] == "rate":
        return f"{row['pp_change']:+.1f} pp"
    return f"{row['absolute_change']:+.0f}"
