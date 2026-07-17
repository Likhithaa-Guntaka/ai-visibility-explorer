"""CSV / DataFrame validation with human-readable messages.

Uploaded data is noisy by nature, so validation is *lenient but honest*: it reports
missing required columns as errors (which block loading) and questionable values as
warnings (which are surfaced to the user but do not block). The goal is to help a
non-technical user fix their file, not to reject it on a technicality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .database import (
    PROMPT_CATEGORIES,
    PROMPTS_COLUMNS,
    RESPONSE_RUNS_COLUMNS,
)

# Columns a user *must* provide. Others can be derived or defaulted.
REQUIRED_PROMPT_COLUMNS: list[str] = ["prompt_id", "prompt_text", "prompt_category"]
REQUIRED_RESPONSE_COLUMNS: list[str] = ["run_id", "prompt_id", "platform", "response_text"]


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
