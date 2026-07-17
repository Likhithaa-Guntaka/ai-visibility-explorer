"""Tests for manual-input and brand validation (mirrors the CSV validation rules)."""

from __future__ import annotations

import pandas as pd

from src import validation as V


# -- manual response validation -----------------------------------------------------


def _valid_row(**overrides):
    row = {
        "run_id": "m001", "prompt_id": "p1", "platform": "ChatGPT (manual)",
        "run_number": 1, "run_date": "2026-07-10", "dataset_kind": "Real",
        "response_text": "1. Notion is flexible.",
    }
    row.update(overrides)
    return row


def test_valid_manual_response_passes():
    res = V.validate_manual_response(_valid_row(), known_prompt_ids={"p1"}, existing_run_ids=set())
    assert res.ok
    assert res.errors == []


def test_missing_response_text_is_error():
    res = V.validate_manual_response(_valid_row(response_text="  "), known_prompt_ids={"p1"})
    assert not res.ok
    assert any("empty" in e.lower() for e in res.errors)


def test_missing_run_id_is_error():
    res = V.validate_manual_response(_valid_row(run_id=""), known_prompt_ids={"p1"})
    assert not res.ok
    assert any("run id" in e.lower() for e in res.errors)


def test_duplicate_run_id_is_error():
    res = V.validate_manual_response(_valid_row(run_id="m001"), known_prompt_ids={"p1"}, existing_run_ids={"m001"})
    assert not res.ok
    assert any("already exists" in e.lower() for e in res.errors)


def test_unknown_prompt_is_error():
    res = V.validate_manual_response(_valid_row(prompt_id="ghost"), known_prompt_ids={"p1"})
    assert not res.ok
    assert any("does not exist" in e.lower() for e in res.errors)


def test_invalid_run_number_is_error():
    assert not V.validate_manual_response(_valid_row(run_number="abc"), known_prompt_ids={"p1"}).ok
    assert not V.validate_manual_response(_valid_row(run_number=0), known_prompt_ids={"p1"}).ok


def test_invalid_dataset_kind_is_error():
    res = V.validate_manual_response(_valid_row(dataset_kind="Made up"), known_prompt_ids={"p1"})
    assert not res.ok
    assert any("dataset type" in e.lower() for e in res.errors)


def test_bad_date_is_warning_not_error():
    res = V.validate_manual_response(_valid_row(run_date="July 10"), known_prompt_ids={"p1"})
    assert res.ok  # still allowed
    assert any("date" in w.lower() for w in res.warnings)


# -- domain normalization -----------------------------------------------------------


def test_normalize_domain_input():
    assert V.normalize_domain_input("https://www.Notion.so/product") == "notion.so"
    assert V.normalize_domain_input("Monday.com") == "monday.com"
    assert V.normalize_domain_input("http://sub.Example.CO.uk:443/x") == "sub.example.co.uk"
    assert V.normalize_domain_input("") == ""


def test_is_valid_domain():
    assert V.is_valid_domain("notion.so")
    assert V.is_valid_domain("sub.example.co.uk")
    assert not V.is_valid_domain("not a domain")
    assert not V.is_valid_domain("localhost")


# -- brand validation ---------------------------------------------------------------


def _brands(rows):
    return pd.DataFrame(rows)


def test_valid_brands_pass():
    df = _brands([
        {"brand_name": "Notion", "brand_domain": "notion.so"},
        {"brand_name": "Trello", "brand_domain": "https://trello.com"},
    ])
    assert V.validate_brands(df).ok


def test_empty_brand_name_is_error():
    df = _brands([{"brand_name": "  ", "brand_domain": "x.com"}])
    res = V.validate_brands(df)
    assert not res.ok
    assert any("non-empty name" in e.lower() for e in res.errors)


def test_duplicate_brand_names_is_error():
    df = _brands([
        {"brand_name": "Notion", "brand_domain": "notion.so"},
        {"brand_name": "notion", "brand_domain": "notion.io"},
    ])
    res = V.validate_brands(df)
    assert not res.ok
    assert any("unique" in e.lower() for e in res.errors)


def test_invalid_domain_is_warning():
    df = _brands([{"brand_name": "Notion", "brand_domain": "not a domain!!"}])
    res = V.validate_brands(df)
    assert res.ok  # invalid domain is a warning, not a blocker
    assert any("valid host" in w.lower() for w in res.warnings)


def test_empty_brands_table_is_error():
    assert not V.validate_brands(pd.DataFrame(columns=["brand_name", "brand_domain"])).ok
