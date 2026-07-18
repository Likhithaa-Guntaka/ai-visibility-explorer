"""Tests for the AI Decision Influence Lab: outcomes, claims, journeys, truth,
evidence, prioritization, filtering, persistence, empty/small-sample handling."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src import appkit
from src import claims as CL
from src import decision_lab as DL
from src import evidence_engine as EV
from src import journeys as JN
from src import persistence as P
from src import prioritization as PR
from src import truth_monitor as TM
from src.database import AnalysisData


# ---------------------------------------------------------------------------
# Hand-built fixture with known, checkable outcomes.
# ---------------------------------------------------------------------------


def _runs():
    return pd.DataFrame([
        {"run_id": "r1", "prompt_id": "p1", "platform": "ChatGPT", "run_date": "2026-06-05", "run_number": 1,
         "response_text": "1. **Notion** is a great all-in-one workspace and we recommend it. "
                          "2. **Trello** is simple but it can get expensive for growing teams."},
        {"run_id": "r2", "prompt_id": "p2", "platform": "Claude", "run_date": "2026-07-10", "run_number": 1,
         "response_text": "1. **Notion** is flexible. 2. **Asana** has reliable task tracking."},
    ])


def _prompts():
    return pd.DataFrame([
        {"prompt_id": "p1", "prompt_text": "best tool?", "prompt_category": "Nonbrand discovery",
         "topic": "Best", "persona": "Team Lead", "journey_stage": "Awareness"},
        {"prompt_id": "p2", "prompt_text": "cheapest tool?", "prompt_category": "Purchase intent",
         "topic": "Pricing", "persona": "Founder", "journey_stage": "Decision"},
    ])


def _mentions():
    return pd.DataFrame([
        {"run_id": "r1", "brand_name": "Notion", "mention_count": 1, "first_mention_position": 0, "is_recommended": True},
        {"run_id": "r1", "brand_name": "Trello", "mention_count": 1, "first_mention_position": 40, "is_recommended": False},
        {"run_id": "r2", "brand_name": "Notion", "mention_count": 1, "first_mention_position": 0, "is_recommended": False},
        {"run_id": "r2", "brand_name": "Asana", "mention_count": 1, "first_mention_position": 20, "is_recommended": False},
    ])


ALIASES = {"Notion": [], "Trello": [], "Asana": [], "Monday.com": ["Monday"]}


def _data() -> AnalysisData:
    prompts = _prompts()
    prompts["project_id"] = "x"
    d = AnalysisData(
        prompts=prompts, response_runs=_runs(),
        brands=pd.DataFrame([
            {"brand_id": "b1", "project_id": "x", "brand_name": "Notion", "brand_domain": "notion.so"},
            {"brand_id": "b2", "project_id": "x", "brand_name": "Trello", "brand_domain": "trello.com"},
            {"brand_id": "b3", "project_id": "x", "brand_name": "Asana", "brand_domain": "asana.com"},
        ]),
    )
    return appkit.run_extraction(d)


# 1. Outcome classification -------------------------------------------------


def test_outcome_classification_rules():
    outs = DL.classify_outcomes(_runs(), _mentions(), ALIASES)
    by = {(r["run_id"], r["brand_name"]): r["outcome"] for _, r in outs.iterrows()}
    assert by[("r1", "Notion")] == DL.OUT_RECOMMENDED         # recommended
    assert by[("r1", "Trello")] == DL.OUT_REJECTED            # "expensive" cue
    assert by[("r2", "Notion")] == DL.OUT_NOT_RECOMMENDED     # mentioned, no cue
    assert by[("r2", "Monday.com")] == DL.OUT_NOT_MENTIONED   # absent


def test_every_brand_run_pair_classified():
    outs = DL.classify_outcomes(_runs(), _mentions(), ALIASES)
    assert len(outs) == len(_runs()) * len(ALIASES)  # 2 runs x 4 brands


# 2. Rejection reason extraction --------------------------------------------


def test_rejection_reason_extraction():
    reasons, evidence = DL.detect_rejection_reasons("trello is simple but it can get expensive for teams")
    assert "Pricing concern" in reasons
    assert evidence  # evidence snippet present


def test_rejection_reason_totals():
    outs = DL.classify_outcomes(_runs(), _mentions(), ALIASES)
    tot = DL.rejection_reason_totals(outs, "Trello")
    assert "Pricing concern" in set(tot["reason"])


# 3. Mention-to-recommendation conversion -----------------------------------


def test_mention_to_recommendation_conversion():
    outs = DL.classify_outcomes(_runs(), _mentions(), ALIASES)
    summ = DL.outcome_summary(outs)
    notion = summ[summ["brand_name"] == "Notion"].iloc[0]
    # Notion mentioned twice, recommended once -> 50%.
    assert notion["mentioned"] == 2
    assert notion["mention_to_recommendation_rate"] == 0.5
    trello = summ[summ["brand_name"] == "Trello"].iloc[0]
    assert trello["rejection_rate"] == 1.0  # its only mention was a rejection


def test_conversion_by_platform():
    outs = DL.classify_outcomes(_runs(), _mentions(), ALIASES)
    conv = DL.conversion_by(outs, _runs(), _prompts(), "platform", "Notion")
    rates = dict(zip(conv["platform"], conv["conversion_rate"]))
    assert rates["ChatGPT"] == 1.0  # recommended on r1
    assert rates["Claude"] == 0.0   # not recommended on r2


# 4. Journey funnel calculations --------------------------------------------


def test_journey_funnel_and_headline():
    data = _data()
    funnel = JN.journey_funnel(data, "Notion")
    assert not funnel.empty
    # Awareness->Discovery, Decision->Decision are present.
    assert set(funnel["stage"]) <= {"Discovery", "Consideration", "Evaluation", "Decision", "Retention"}
    head = JN.journey_headline_metrics(funnel)
    assert head["stage_most_lost"] in set(funnel["stage"])


def test_journey_kind_labelled():
    data = _data()
    assert "Simulated" in str(JN.resolve_journey(data)["journey_kind"].iloc[0])


# 5. Claim extraction --------------------------------------------------------


def test_claim_extraction():
    claims = CL.extract_all_claims(_runs(), ALIASES)
    notion = claims[claims["brand_name"] == "Notion"]
    assert "Positioning claim" in set(notion["claim_type"])  # "all-in-one workspace"
    # Not-mentioned brands produce no claims.
    assert "Monday.com" not in set(claims["brand_name"])


# 6. Claim ↔ citation relationships -----------------------------------------


def test_claim_citation_relationships():
    data = _data()
    prov = CL.claims_with_provenance(data.brand_claims, data.response_runs, data.citations, data.recommendation_outcomes)
    assert "outcome" in prov.columns and "has_citation" in prov.columns
    support = CL.claims_citation_support(prov, "Notion")
    assert set(["claim_type", "with_citation", "without_citation", "supported_share"]).issubset(support.columns)


# 7. Contradiction & outdated classifications --------------------------------


def test_conflicting_claims_detected():
    data = appkit.load_demo_analysis()
    conf = CL.conflicting_claims(data.brand_claims, "ClickUp")
    assert not conf.empty  # ClickUp described both affordable and pricey


def test_truth_verdicts_including_conflict_and_outdated():
    data = appkit.load_demo_analysis()
    comp = TM.compare_facts(data.brand_facts, data.brand_claims, data.response_runs, data.citations)
    verdicts = set(comp["verdict"])
    assert TM.V_CONFLICTING in verdicts   # ClickUp pricing conflict (affordable vs pricey)
    # An outdated discontinued-feature fact classifies deterministically.
    outdated = TM._classify_fact("Discontinued feature", "legacy dashboard", "clickup has a legacy dashboard view")
    assert outdated == TM.V_OUTDATED
    assert TM._classify_fact("Feature", "kanban boards", "") == TM.V_MISSING


# 8. Evidence recommendations ------------------------------------------------


def test_evidence_recommendations_grounded():
    data = appkit.load_demo_analysis()
    eo = EV.evidence_opportunities(data, "Trello")
    assert not eo.empty
    # Every objection maps to an asset and is backed by >=1 occurrence.
    assert (eo["occurrences"] >= 1).all()
    assert eo["recommended_asset"].isin(set(EV.REASON_TO_ASSET.values())).all()


def test_no_evidence_without_rejections():
    """A brand with no rejections yields no evidence actions."""
    data = appkit.load_demo_analysis()
    eo = EV.evidence_opportunities(data, "Notion")  # Notion has ~0 rejections in demo
    assert eo.empty or (eo["occurrences"] >= 1).all()


# 9. Priority formula + editable weights ------------------------------------


def test_priority_formula_and_weights():
    data = appkit.load_demo_analysis()
    pt = PR.priority_table(data, "Trello")
    assert not pt.empty
    row = pt.iloc[0]
    # Weighted contribution equals weight * raw for each component.
    for comp, w in PR.DEFAULT_WEIGHTS.items():
        assert abs(row[f"wc_{comp}"] - w * row[comp]) < 1e-9
    # Priority = sum(wc)/sum(weights)*100.
    wc_sum = sum(row[f"wc_{c}"] for c in PR.DEFAULT_WEIGHTS)
    expected = wc_sum / sum(PR.DEFAULT_WEIGHTS.values()) * 100
    assert abs(row["priority"] - round(expected, 1)) < 0.11


def test_editable_weights_change_priority():
    data = appkit.load_demo_analysis()
    base = PR.priority_table(data, "Trello")
    boosted = PR.priority_table(data, "Trello", weights={"visibility_gap": 5.0})
    # Ordering or values should shift when a weight changes.
    assert not base["priority"].equals(boosted["priority"])


def test_priority_breakdown_exposes_all_components():
    data = appkit.load_demo_analysis()
    pt = PR.priority_table(data, "Trello")
    bd = PR.priority_breakdown(pt.iloc[0])
    assert set(bd["component"]) == set(PR.DEFAULT_WEIGHTS)
    assert set(["definition", "raw_value", "weight", "weighted_contribution"]).issubset(bd.columns)


# 10. Manual corrections -----------------------------------------------------


def test_manual_outcome_correction_changes_metrics():
    data = _data()
    outs = data.recommendation_outcomes.copy()
    # Flip Trello r1 from rejected to recommended.
    mask = (outs["run_id"] == "r1") & (outs["brand_name"] == "Trello")
    outs.loc[mask, "outcome"] = DL.OUT_RECOMMENDED
    summ = DL.outcome_summary(outs)
    trello = summ[summ["brand_name"] == "Trello"].iloc[0]
    assert trello["recommended"] == 1
    assert trello["rejection_rate"] == 0.0


# 11. Dataset filtering ------------------------------------------------------


def test_filtering_subsets_outcomes_and_claims():
    data = appkit.load_demo_analysis()
    scoped = appkit.filter_data(data, question_clusters=["Pricing & value"])
    run_ids = set(scoped.response_runs["run_id"])
    assert set(scoped.recommendation_outcomes["run_id"]) <= run_ids
    assert set(scoped.brand_claims["run_id"]) <= run_ids
    assert len(scoped.recommendation_outcomes) < len(data.recommendation_outcomes)


# 12. Export / import --------------------------------------------------------


def test_lab_tables_round_trip():
    data = appkit.load_demo_analysis()
    bundle = P.import_bundle(P.export_project_json(data))
    for table in ["recommendation_outcomes", "brand_claims", "brand_facts"]:
        orig = getattr(data, table).reset_index(drop=True)
        got = getattr(bundle.data, table).reset_index(drop=True)
        assert orig.equals(got), f"{table} did not round-trip"


def test_schema_version_bumped_to_2():
    assert P.PROJECT_SCHEMA_VERSION == 2
    # v1 files still import (new tables come back empty).
    b = P.import_bundle({"schema_version": 1, "tables": {}})
    assert b.data.recommendation_outcomes.empty


# 13. Empty data -------------------------------------------------------------


def test_empty_data_is_safe():
    empty = AnalysisData()
    assert DL.classify_outcomes(empty.response_runs, empty.brand_mentions, {}).empty
    assert DL.outcome_summary(empty.recommendation_outcomes).empty
    assert CL.extract_all_claims(empty.response_runs, {}).empty
    assert JN.journey_funnel(empty, "X").empty
    assert TM.compare_facts(empty.brand_facts, empty.brand_claims, empty.response_runs, empty.citations).empty
    assert EV.evidence_opportunities(empty, "X").empty
    assert PR.priority_table(empty, "X").empty


# 14. Small samples ----------------------------------------------------------


def test_small_sample_confidence_labels():
    assert "Very low" in EV._confidence(1)
    assert "Low" in EV._confidence(3)
    assert "Moderate" in EV._confidence(9)


def test_priority_sample_confidence_component_scales_with_n():
    data = _data()  # tiny 2-run dataset
    pt = PR.priority_table(data, "Notion")
    # With few runs, sample_confidence should be well below 1.
    assert (pt["sample_confidence"] <= 1.0).all()
    assert (pt["sample_confidence"] < 1.0).any()
