"""AEO Question Cluster analysis (feature: AEO Question Cluster Map).

Groups prompts into question clusters using the project's **existing structured
metadata** — topic, search intent, persona, journey stage, brand/non-brand, or a
user-defined cluster label — rather than keyword guessing. For each cluster it reports
visibility for the focal brand and competitors, which questions are won or lost, and a
grounded recommendation about whether the cluster can be served by one comprehensive
page or needs separate pages.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import metrics as M
from .database import AnalysisData, ensure_prompt_cluster_columns

# The dimensions a user can cluster by. ``brand_type`` is derived from is_brand_prompt.
CLUSTER_DIMENSIONS: list[str] = [
    "question_cluster",
    "topic",
    "search_intent",
    "persona",
    "journey_stage",
    "brand_type",
]

DIMENSION_LABELS: dict[str, str] = {
    "question_cluster": "User-defined question cluster",
    "topic": "Topic",
    "search_intent": "Search intent",
    "persona": "Customer persona",
    "journey_stage": "Journey stage",
    "brand_type": "Brand vs non-brand question",
}

# Outcome labels for a single question.
OUTCOME_FOCAL_WINS = "Focal brand wins"
OUTCOME_COMPETITOR_WINS = "Competitor wins"
OUTCOME_NO_BRAND = "No tracked brand appears"


def prepare_prompts(prompts: pd.DataFrame) -> pd.DataFrame:
    """Ensure clustering columns exist, including the derived ``brand_type``."""
    out = ensure_prompt_cluster_columns(prompts)
    if out.empty:
        if "brand_type" not in out.columns:
            out["brand_type"] = pd.Series(dtype=str)
        return out
    if "is_brand_prompt" in out.columns:
        out["brand_type"] = out["is_brand_prompt"].apply(
            lambda v: "Brand question" if bool(v) else "Non-brand question"
        )
    else:
        out["brand_type"] = "Non-brand question"
    return out


def _runs_for_prompts(response_runs: pd.DataFrame, prompt_ids: set[str]) -> pd.DataFrame:
    if response_runs.empty:
        return response_runs
    return response_runs[response_runs["prompt_id"].isin(prompt_ids)]


def cluster_summary(data: AnalysisData, focal_brand: str, dimension: str = "question_cluster") -> pd.DataFrame:
    """Per-cluster visibility summary for the focal brand.

    Returns columns: ``cluster``, ``prompts``, ``runs``, ``focal_mention_rate``,
    ``share_of_voice``, ``recommendation_rate``, ``citation_rate``, ``top_competitor``,
    ``competitor_mention_rate``, ``gap`` (competitor − focal mention rate).
    """
    cols = ["cluster", "prompts", "runs", "focal_mention_rate", "share_of_voice",
            "recommendation_rate", "citation_rate", "top_competitor",
            "competitor_mention_rate", "gap"]
    prompts = prepare_prompts(data.prompts)
    if prompts.empty or data.response_runs.empty or dimension not in prompts.columns:
        return pd.DataFrame(columns=cols)

    rows = []
    for cluster_value, grp in prompts.groupby(dimension):
        pids = set(grp["prompt_id"])
        runs = _runs_for_prompts(data.response_runs, pids)
        n_runs = len(runs)
        if n_runs == 0:
            continue
        run_ids = set(runs["run_id"])
        mentions = data.brand_mentions[data.brand_mentions["run_id"].isin(run_ids)] if not data.brand_mentions.empty else data.brand_mentions
        citations = data.citations[data.citations["run_id"].isin(run_ids)] if not data.citations.empty else data.citations

        focal_runs = mentions[mentions["brand_name"] == focal_brand]["run_id"].nunique() if not mentions.empty else 0
        sov_df = M.share_of_voice(mentions)
        focal_sov = float(sov_df[sov_df["brand_name"] == focal_brand]["share_of_voice"].iloc[0]) if (
            not sov_df.empty and focal_brand in set(sov_df["brand_name"])) else 0.0
        rec = mentions[(mentions["brand_name"] == focal_brand) & (mentions["is_recommended"].astype(bool))]["run_id"].nunique() if not mentions.empty else 0
        cite_runs = citations["run_id"].nunique() if not citations.empty else 0

        # Strongest competitor in this cluster.
        comp_name, comp_rate = None, 0.0
        if not mentions.empty:
            comp = mentions[mentions["brand_name"] != focal_brand]
            if not comp.empty:
                by_comp = comp.groupby("brand_name")["run_id"].nunique() / n_runs
                comp_name = by_comp.idxmax()
                comp_rate = float(by_comp.max())

        focal_rate = focal_runs / n_runs
        rows.append({
            "cluster": cluster_value,
            "prompts": int(len(grp)),
            "runs": int(n_runs),
            "focal_mention_rate": focal_rate,
            "share_of_voice": focal_sov,
            "recommendation_rate": rec / n_runs,
            "citation_rate": cite_runs / n_runs,
            "top_competitor": comp_name,
            "competitor_mention_rate": comp_rate,
            "gap": comp_rate - focal_rate,
        })
    return pd.DataFrame(rows, columns=cols).sort_values("gap", ascending=False).reset_index(drop=True)


def competitor_rates_in_cluster(
    data: AnalysisData, dimension: str, cluster_value: str
) -> pd.DataFrame:
    """Mention rate and share of voice for **every** tracked brand inside one cluster.

    Returns columns: ``brand_name``, ``mention_rate``, ``share_of_voice``, ``recommendation_rate``.
    """
    cols = ["brand_name", "mention_rate", "share_of_voice", "recommendation_rate"]
    prompts = prepare_prompts(data.prompts)
    if prompts.empty or dimension not in prompts.columns:
        return pd.DataFrame(columns=cols)
    pids = set(prompts[prompts[dimension] == cluster_value]["prompt_id"])
    runs = _runs_for_prompts(data.response_runs, pids)
    if runs.empty:
        return pd.DataFrame(columns=cols)
    run_ids = set(runs["run_id"])
    mentions = data.brand_mentions[data.brand_mentions["run_id"].isin(run_ids)] if not data.brand_mentions.empty else data.brand_mentions
    if mentions.empty:
        return pd.DataFrame(columns=cols)
    mr = M.brand_mention_rate(mentions, runs)
    sov = M.share_of_voice(mentions)
    rec = M.recommendation_rate(mentions, runs)
    out = (
        mr[["brand_name", "mention_rate"]]
        .merge(sov[["brand_name", "share_of_voice"]], on="brand_name", how="outer")
        .merge(rec[["brand_name", "recommendation_rate"]], on="brand_name", how="outer")
        .fillna(0.0)
    )
    return out.sort_values("share_of_voice", ascending=False).reset_index(drop=True)


def question_outcomes(
    data: AnalysisData, focal_brand: str, dimension: str, cluster_value: str
) -> pd.DataFrame:
    """Per-question outcome inside a cluster: who wins each question.

    A question is classified by comparing, across the question's responses, the focal
    brand's mention rate with the strongest competitor's:

    * ``Focal brand wins``          — focal rate > best competitor rate
    * ``Competitor wins``           — best competitor rate > focal rate
    * ``No tracked brand appears``  — no tracked brand was mentioned at all

    Ties (both mentioned equally) are reported as ``Focal brand wins`` only when the
    focal brand is also mentioned first more often; otherwise ``Competitor wins``.
    Returns columns: ``prompt_id``, ``prompt_text``, ``runs``, ``focal_rate``,
    ``top_competitor``, ``competitor_rate``, ``outcome``.
    """
    cols = ["prompt_id", "prompt_text", "runs", "focal_rate", "top_competitor", "competitor_rate", "outcome"]
    prompts = prepare_prompts(data.prompts)
    if prompts.empty or dimension not in prompts.columns:
        return pd.DataFrame(columns=cols)
    cluster_prompts = prompts[prompts[dimension] == cluster_value]
    if cluster_prompts.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, p in cluster_prompts.iterrows():
        pid = p["prompt_id"]
        runs = _runs_for_prompts(data.response_runs, {pid})
        n = len(runs)
        if n == 0:
            continue
        run_ids = set(runs["run_id"])
        mentions = data.brand_mentions[data.brand_mentions["run_id"].isin(run_ids)] if not data.brand_mentions.empty else data.brand_mentions
        focal_rate = (mentions[mentions["brand_name"] == focal_brand]["run_id"].nunique() / n) if not mentions.empty else 0.0
        comp_name, comp_rate = None, 0.0
        if not mentions.empty:
            comp = mentions[mentions["brand_name"] != focal_brand]
            if not comp.empty:
                by_comp = comp.groupby("brand_name")["run_id"].nunique() / n
                comp_name = by_comp.idxmax()
                comp_rate = float(by_comp.max())

        if mentions.empty or (focal_rate == 0.0 and comp_rate == 0.0):
            outcome = OUTCOME_NO_BRAND
        elif focal_rate > comp_rate:
            outcome = OUTCOME_FOCAL_WINS
        elif comp_rate > focal_rate:
            outcome = OUTCOME_COMPETITOR_WINS
        else:
            # Tie on mention rate -> break using first-mention share.
            fms = M.first_mention_share(mentions)
            focal_first = float(fms[fms["brand_name"] == focal_brand]["first_mention_share"].iloc[0]) if (
                not fms.empty and focal_brand in set(fms["brand_name"])) else 0.0
            comp_first = float(fms[fms["brand_name"] == comp_name]["first_mention_share"].iloc[0]) if (
                not fms.empty and comp_name in set(fms["brand_name"])) else 0.0
            outcome = OUTCOME_FOCAL_WINS if focal_first > comp_first else OUTCOME_COMPETITOR_WINS

        rows.append({
            "prompt_id": pid,
            "prompt_text": p.get("prompt_text", ""),
            "runs": int(n),
            "focal_rate": focal_rate,
            "top_competitor": comp_name,
            "competitor_rate": comp_rate,
            "outcome": outcome,
        })
    return pd.DataFrame(rows, columns=cols)


def cluster_coverage_gaps(
    data: AnalysisData, focal_brand: str, dimension: str, cluster_value: str
) -> pd.DataFrame:
    """Content coverage gaps within a cluster, by topic (competitor rate − focal rate)."""
    prompts = prepare_prompts(data.prompts)
    if prompts.empty or dimension not in prompts.columns:
        return pd.DataFrame(columns=["topic", "focal_rate", "top_competitor", "competitor_rate", "gap", "total_runs"])
    pids = set(prompts[prompts[dimension] == cluster_value]["prompt_id"])
    scoped_prompts = prompts[prompts["prompt_id"].isin(pids)]
    scoped_runs = _runs_for_prompts(data.response_runs, pids)
    if scoped_runs.empty:
        return pd.DataFrame(columns=["topic", "focal_rate", "top_competitor", "competitor_rate", "gap", "total_runs"])
    run_ids = set(scoped_runs["run_id"])
    mentions = data.brand_mentions[data.brand_mentions["run_id"].isin(run_ids)] if not data.brand_mentions.empty else data.brand_mentions
    enriched = M.enrich_mentions(mentions, scoped_runs, scoped_prompts)
    return M.content_coverage_gaps(enriched, scoped_runs, scoped_prompts, focal_brand, dimension="topic")


# ---------------------------------------------------------------------------
# Page consolidation recommendation (deterministic, grounded, explainable).
# ---------------------------------------------------------------------------

# Rules are exposed so the UI can display exactly why a recommendation was made.
CONSOLIDATION_RULES = (
    "One comprehensive page when the cluster's questions share a single search intent AND a "
    "single journey stage AND at most one persona, and there are ≤ 6 questions. "
    "One page with per-persona sections when intent is single, journey stages ≤ 2, and questions ≤ 8. "
    "Otherwise separate pages, split by whichever dimension varies most (intent, then journey stage, then persona)."
)


def page_consolidation_recommendation(
    data: AnalysisData, dimension: str, cluster_value: str
) -> dict:
    """Recommend one comprehensive page vs separate pages for a cluster.

    Grounded purely in the cluster's own metadata spread (intents, journey stages,
    personas, question count). Returns ``recommendation``, ``reason``, ``rules``, and
    the ``evidence`` counts that drove it.
    """
    prompts = prepare_prompts(data.prompts)
    empty = {
        "recommendation": "Not enough data",
        "reason": "This cluster has no prompts in the current view.",
        "rules": CONSOLIDATION_RULES,
        "evidence": {},
        "split_by": None,
    }
    if prompts.empty or dimension not in prompts.columns:
        return empty
    grp = prompts[prompts[dimension] == cluster_value]
    if grp.empty:
        return empty

    intents = sorted(grp["search_intent"].dropna().unique().tolist())
    stages = sorted(grp["journey_stage"].dropna().unique().tolist())
    personas = sorted(grp["persona"].dropna().unique().tolist())
    n_q = int(len(grp))
    evidence = {
        "questions": n_q,
        "search_intents": intents,
        "journey_stages": stages,
        "personas": personas,
    }

    if len(intents) <= 1 and len(stages) <= 1 and len(personas) <= 1 and n_q <= 6:
        return {
            "recommendation": "One comprehensive page",
            "reason": (
                f"All {n_q} question(s) share one search intent ({intents[0] if intents else 'n/a'}) and one "
                f"journey stage ({stages[0] if stages else 'n/a'}), targeting a single persona. A single page "
                "answering each question in its own section should serve the whole cluster."
            ),
            "rules": CONSOLIDATION_RULES,
            "evidence": evidence,
            "split_by": None,
        }

    if len(intents) <= 1 and len(stages) <= 2 and n_q <= 8:
        return {
            "recommendation": "One comprehensive page with per-persona sections",
            "reason": (
                f"The {n_q} questions share a single search intent ({intents[0] if intents else 'n/a'}) but span "
                f"{len(personas)} persona(s) and {len(stages)} journey stage(s). One page can cover them if each "
                "persona gets a clearly-headed section."
            ),
            "rules": CONSOLIDATION_RULES,
            "evidence": evidence,
            "split_by": None,
        }

    # Otherwise: separate pages, split by whichever dimension varies most.
    if len(intents) > 1:
        split_by, values = "search_intent", intents
    elif len(stages) > 1:
        split_by, values = "journey_stage", stages
    else:
        split_by, values = "persona", personas
    return {
        "recommendation": "Separate pages",
        "reason": (
            f"The {n_q} questions span {len(intents)} search intent(s), {len(stages)} journey stage(s) and "
            f"{len(personas)} persona(s). Mixing these on one page tends to serve none of them well — split by "
            f"{split_by.replace('_', ' ')}: {', '.join(map(str, values))}."
        ),
        "rules": CONSOLIDATION_RULES,
        "evidence": evidence,
        "split_by": split_by,
    }


def cluster_questions(data: AnalysisData, dimension: str, cluster_value: str) -> list[str]:
    """The prompt texts belonging to a cluster (used by extractability coverage)."""
    prompts = prepare_prompts(data.prompts)
    if prompts.empty or dimension not in prompts.columns:
        return []
    grp = prompts[prompts[dimension] == cluster_value]
    return [str(q) for q in grp["prompt_text"].dropna().tolist() if str(q).strip()]
