"""CSV / DataFrame validation with human-readable messages.

Uploaded data is noisy by nature, so validation is *lenient but honest*: it reports
missing required columns as errors (which block loading) and questionable values as
warnings (which are surfaced to the user but do not block). The goal is to help a
non-technical user fix their file, not to reject it on a technicality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from .database import (
    DATASET_KINDS,
    PROMPT_CATEGORIES,
    PROMPTS_COLUMNS,
    RESPONSE_RUNS_COLUMNS,
)

# Columns a user *must* provide. Others can be derived or defaulted.
REQUIRED_PROMPT_COLUMNS: list[str] = ["prompt_id", "prompt_text", "prompt_category"]
REQUIRED_RESPONSE_COLUMNS: list[str] = ["run_id", "prompt_id", "platform", "response_text"]

# A permissive-but-real domain check (label.label, each label alnum/hyphen).
_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9](-?[a-z0-9])*\.)+[a-z]{2,63}$")


@dataclass
class ValidationResult:
    """Outcome of validating a DataFrame."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_prompts(df: pd.DataFrame) -> ValidationResult:
    """Validate a prompts DataFrame/CSV before loading it into an analysis."""
    result = ValidationResult()
    _check_required_columns(df, REQUIRED_PROMPT_COLUMNS, "prompts", result)
    if not result.ok:
        return result

    if df["prompt_id"].duplicated().any():
        dupes = df.loc[df["prompt_id"].duplicated(), "prompt_id"].unique().tolist()
        result.add_error(f"Duplicate prompt_id values found: {', '.join(map(str, dupes[:5]))}")

    if df["prompt_text"].astype(str).str.strip().eq("").any():
        result.add_warning("Some prompts have empty prompt_text; those rows will add no signal.")

    if "prompt_category" in df.columns:
        unknown = sorted(set(df["prompt_category"].dropna().astype(str)) - set(PROMPT_CATEGORIES))
        if unknown:
            result.add_warning(
                "Unrecognized prompt_category values (allowed for flexibility, but they "
                f"won't map to the standard categories): {', '.join(unknown[:6])}"
            )

    _warn_missing_optional(df, PROMPTS_COLUMNS, "prompts", result)
    return result


def validate_responses(df: pd.DataFrame, known_prompt_ids: set[str] | None = None) -> ValidationResult:
    """Validate a responses DataFrame/CSV; optionally check prompt_id references."""
    result = ValidationResult()
    _check_required_columns(df, REQUIRED_RESPONSE_COLUMNS, "responses", result)
    if not result.ok:
        return result

    if df["run_id"].duplicated().any():
        dupes = df.loc[df["run_id"].duplicated(), "run_id"].unique().tolist()
        result.add_error(f"Duplicate run_id values found: {', '.join(map(str, dupes[:5]))}")

    if df["response_text"].astype(str).str.strip().eq("").any():
        n = int(df["response_text"].astype(str).str.strip().eq("").sum())
        result.add_warning(f"{n} response(s) have empty response_text and will yield no mentions.")

    if known_prompt_ids is not None:
        missing = sorted(set(df["prompt_id"].astype(str)) - {str(p) for p in known_prompt_ids})
        if missing:
            result.add_warning(
                "Some responses reference prompt_id values not found in the prompts "
                f"table: {', '.join(missing[:6])}. They will be kept but won't get prompt context."
            )

    _warn_missing_optional(df, RESPONSE_RUNS_COLUMNS, "responses", result)
    return result


def _check_required_columns(
    df: pd.DataFrame, required: list[str], label: str, result: ValidationResult
) -> None:
    if df is None or len(df.columns) == 0:
        result.add_error(f"The {label} file appears to be empty or has no header row.")
        return
    missing = [c for c in required if c not in df.columns]
    if missing:
        result.add_error(
            f"The {label} file is missing required column(s): {', '.join(missing)}. "
            f"Required columns are: {', '.join(required)}."
        )
    if len(df) == 0:
        result.add_warning(f"The {label} file has a header but no data rows.")


def _warn_missing_optional(
    df: pd.DataFrame, full_schema: list[str], label: str, result: ValidationResult
) -> None:
    optional_missing = [c for c in full_schema if c not in df.columns]
    if optional_missing:
        result.add_warning(
            f"Optional {label} column(s) not provided (defaults will be used): "
            f"{', '.join(optional_missing)}."
        )


# ---------------------------------------------------------------------------
# Manual (pasted) response validation — the same core rules as the CSV path.
# ---------------------------------------------------------------------------


def normalize_domain_input(raw: str) -> str:
    """Normalize a user-entered brand domain to a bare host.

    Accepts bare domains, or full URLs with scheme/path, and returns
    ``lowercase host without a leading www.``. Returns "" for blanks.

    >>> normalize_domain_input("https://www.Notion.so/product")
    'notion.so'
    """
    raw = (raw or "").strip().lower()
    if not raw:
        return ""
    candidate = raw if "//" in raw else "//" + raw  # let urlparse find the host
    host = urlparse(candidate).netloc or urlparse("//" + raw).netloc
    host = host.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def is_valid_domain(domain: str) -> bool:
    """True if ``domain`` looks like a real hostname (after normalization)."""
    return bool(_DOMAIN_RE.match(domain))


def validate_manual_response(
    row: dict,
    known_prompt_ids: Optional[set[str]] = None,
    existing_run_ids: Optional[set[str]] = None,
) -> ValidationResult:
    """Validate a single pasted response, applying the same rules as CSV uploads.

    Checks: required values, a valid prompt reference, a unique run id, a valid run
    number, a valid date, a valid dataset type, and non-empty response text.
    """
    result = ValidationResult()
    existing_run_ids = existing_run_ids or set()

    run_id = str(row.get("run_id", "") or "").strip()
    if not run_id:
        result.add_error("Run ID is required.")
    elif run_id in existing_run_ids:
        result.add_error(f"Run ID '{run_id}' already exists — run IDs must be unique.")

    prompt_id = str(row.get("prompt_id", "") or "").strip()
    if not prompt_id:
        result.add_error("A prompt must be selected (prompt_id is required).")
    elif known_prompt_ids is not None and prompt_id not in {str(p) for p in known_prompt_ids}:
        result.add_error(f"Prompt '{prompt_id}' does not exist. Add the prompt first, or pick an existing one.")

    if not str(row.get("response_text", "") or "").strip():
        result.add_error("Response text cannot be empty.")

    run_number = row.get("run_number", 1)
    try:
        if int(run_number) < 1:
            result.add_error("Run number must be 1 or greater.")
    except (TypeError, ValueError):
        result.add_error(f"Run number '{run_number}' is not a valid whole number.")

    dataset_kind = str(row.get("dataset_kind", "") or "").strip()
    if dataset_kind and dataset_kind not in DATASET_KINDS:
        result.add_error(f"Dataset type '{dataset_kind}' is invalid. Choose one of: {', '.join(DATASET_KINDS)}.")

    run_date = str(row.get("run_date", "") or "").strip()
    if run_date and not _looks_like_date(run_date):
        result.add_warning(f"Collection date '{run_date}' is not in YYYY-MM-DD format; it will be stored as text.")

    if not str(row.get("platform", "") or "").strip():
        result.add_warning("No platform label was given; the response will still be added.")

    return result


def _looks_like_date(value: str) -> bool:
    """True if ``value`` parses as an ISO date (YYYY-MM-DD)."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Brand validation (names + domains).
# ---------------------------------------------------------------------------


def validate_brands(df: pd.DataFrame) -> ValidationResult:
    """Validate a brands table: non-empty unique names and normalizable domains.

    Domains are validated after normalization; malformed domains are a warning (they
    simply won't classify source ownership), not a hard error.
    """
    result = ValidationResult()
    if df is None or df.empty or "brand_name" not in df.columns:
        result.add_error("Add at least one brand with a name.")
        return result

    names = df["brand_name"].fillna("").astype(str).str.strip()
    if (names == "").any():
        result.add_error("Every brand must have a non-empty name.")

    non_empty = names[names != ""]
    dupes = non_empty[non_empty.str.lower().duplicated()].unique().tolist()
    if dupes:
        result.add_error(f"Duplicate brand name(s): {', '.join(dupes)}. Brand names must be unique.")

    if "brand_domain" in df.columns:
        for _, raw in df["brand_domain"].fillna("").astype(str).items():
            raw = raw.strip()
            if not raw:
                continue
            normalized = normalize_domain_input(raw)
            if not is_valid_domain(normalized):
                result.add_warning(
                    f"Domain '{raw}' does not look like a valid host; source ownership won't be detected for it."
                )
    return result
