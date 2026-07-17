"""File-based project export / import (no hosted database required).

Streamlit keeps the active analysis only in session state, so a refresh loses it —
including manual extraction corrections. This module serialises an entire project to a
single JSON document and restores it later, so work can be saved and shared as a file.

What is saved
-------------
projects, benchmarks, brands, prompts, response_runs, brand_mentions, citations,
brand_entities, page_audits (when present), plus the UI extras: brand aliases, the
focal brand, and experiment definitions. A ``schema_version`` is embedded for forward
compatibility.

What is NOT saved
-----------------
No secrets, API keys, environment variables, or `.env` contents — only the analysis
data listed above.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

from .database import (
    BENCHMARKS_COLUMNS,
    BRAND_ENTITIES_COLUMNS,
    BRAND_MENTIONS_COLUMNS,
    BRANDS_COLUMNS,
    CITATIONS_COLUMNS,
    PAGE_AUDITS_COLUMNS,
    PROJECTS_COLUMNS,
    PROMPTS_COLUMNS,
    RESPONSE_RUNS_COLUMNS,
    AnalysisData,
)

#: Bump when the on-disk format changes incompatibly.
PROJECT_SCHEMA_VERSION = 1

#: table name -> canonical column list (drives (de)serialisation + reindexing).
_TABLE_COLUMNS: dict[str, list[str]] = {
    "projects": PROJECTS_COLUMNS,
    "benchmarks": BENCHMARKS_COLUMNS,
    "brands": BRANDS_COLUMNS,
    "prompts": PROMPTS_COLUMNS,
    "response_runs": RESPONSE_RUNS_COLUMNS,
    "brand_mentions": BRAND_MENTIONS_COLUMNS,
    "citations": CITATIONS_COLUMNS,
    "brand_entities": BRAND_ENTITIES_COLUMNS,
    "page_audits": PAGE_AUDITS_COLUMNS,
}


class ProjectImportError(Exception):
    """Raised when an imported project file is invalid or incompatible."""


@dataclass
class ProjectBundle:
    """The full restorable state of a project."""

    data: AnalysisData
    alias_overrides: dict[str, list[str]] = field(default_factory=dict)
    focal_brand: Optional[str] = None
    experiments: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DataFrame (de)serialisation.
# ---------------------------------------------------------------------------


def _df_to_json(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Serialise a DataFrame to ``{"columns": [...], "records": [...]}`` (JSON-safe)."""
    if df is None or df.empty:
        return {"columns": columns, "records": []}
    safe = df.reindex(columns=columns).where(pd.notna(df.reindex(columns=columns)), None)
    return {"columns": columns, "records": safe.to_dict(orient="records")}


def _df_from_json(blob: Any, columns: list[str]) -> pd.DataFrame:
    """Rebuild a DataFrame from a serialised table, reindexed to canonical columns."""
    if not isinstance(blob, dict) or "records" not in blob:
        raise ProjectImportError("A table entry is malformed (missing 'records').")
    records = blob.get("records") or []
    if not isinstance(records, list):
        raise ProjectImportError("A table's 'records' must be a list.")
    df = pd.DataFrame(records)
    return df.reindex(columns=columns) if not df.empty else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Export.
# ---------------------------------------------------------------------------


def export_bundle(
    data: AnalysisData,
    alias_overrides: Optional[dict[str, list[str]]] = None,
    focal_brand: Optional[str] = None,
    experiments: Optional[list] = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable project dict. ``experiments`` may be Experiment
    dataclasses or plain dicts."""
    exp_dicts: list[dict] = []
    for e in experiments or []:
        exp_dicts.append(e if isinstance(e, dict) else asdict(e))
    tables = {name: _df_to_json(getattr(data, name), cols) for name, cols in _TABLE_COLUMNS.items()}
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app": "ai-visibility-explorer",
        "tables": tables,
        "alias_overrides": alias_overrides or {},
        "focal_brand": focal_brand,
        "experiments": exp_dicts,
    }


def bundle_to_json(payload: dict[str, Any], indent: int = 2) -> str:
    """Serialise an exported bundle to a JSON string."""
    return json.dumps(payload, indent=indent, default=str)


def export_project_json(
    data: AnalysisData,
    alias_overrides: Optional[dict[str, list[str]]] = None,
    focal_brand: Optional[str] = None,
    experiments: Optional[list] = None,
) -> str:
    """Convenience: export straight to a JSON string."""
    return bundle_to_json(export_bundle(data, alias_overrides, focal_brand, experiments))


# ---------------------------------------------------------------------------
# Import + validation.
# ---------------------------------------------------------------------------


def import_bundle(payload: Any) -> ProjectBundle:
    """Validate and restore a project from a dict or JSON string.

    Raises :class:`ProjectImportError` with a human-readable message when the file is
    not valid JSON, is missing required structure, or has an unsupported schema version.
    Missing optional tables/fields are tolerated and restored as empty.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProjectImportError(f"File is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ProjectImportError("Project file must be a JSON object.")

    version = payload.get("schema_version")
    if version is None:
        raise ProjectImportError("Missing 'schema_version'. This does not look like a project export.")
    if not isinstance(version, int) or version > PROJECT_SCHEMA_VERSION:
        raise ProjectImportError(
            f"Unsupported schema_version {version!r}. This app supports up to "
            f"version {PROJECT_SCHEMA_VERSION}. Please update the app."
        )

    tables = payload.get("tables")
    if not isinstance(tables, dict):
        raise ProjectImportError("Missing or invalid 'tables' section.")

    kwargs = {}
    for name, cols in _TABLE_COLUMNS.items():
        blob = tables.get(name)
        kwargs[name] = _df_from_json(blob, cols) if blob is not None else pd.DataFrame(columns=cols)

    data = AnalysisData(**kwargs)

    aliases = payload.get("alias_overrides") or {}
    if not isinstance(aliases, dict):
        raise ProjectImportError("'alias_overrides' must be an object mapping brand -> list of aliases.")

    experiments = payload.get("experiments") or []
    if not isinstance(experiments, list):
        raise ProjectImportError("'experiments' must be a list.")

    return ProjectBundle(
        data=data,
        alias_overrides={k: list(v) for k, v in aliases.items()},
        focal_brand=payload.get("focal_brand"),
        experiments=experiments,
    )
