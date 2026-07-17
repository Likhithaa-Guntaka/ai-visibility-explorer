"""DuckDB data layer for AI Visibility Explorer.

Design philosophy
-----------------
CSV files (or user-entered rows held in the Streamlit session) are the **source of
truth**. DuckDB is used purely as a fast, SQL-native *analytics engine* over those
rows. We therefore keep an in-memory DuckDB connection, register the canonical
pandas DataFrames as tables, and run SQL for metrics. Nothing important lives only
inside the ``.duckdb`` file, so it is safe to delete and rebuild at any time.

The schema below mirrors the project's documented data model one-to-one so the SQL
and the README never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column definitions (single source of truth, reused by validation).
# ---------------------------------------------------------------------------

PROJECTS_COLUMNS: list[str] = ["project_id", "project_name", "industry", "created_at"]

BRANDS_COLUMNS: list[str] = ["brand_id", "project_id", "brand_name", "brand_domain"]

PROMPTS_COLUMNS: list[str] = [
    "prompt_id",
    "project_id",
    "prompt_text",
    "prompt_category",
    "topic",
    "persona",
    "journey_stage",
    "is_brand_prompt",
]

RESPONSE_RUNS_COLUMNS: list[str] = [
    "run_id",
    "prompt_id",
    "platform",
    "model_name",
    "run_date",
    "run_number",
    "response_text",
    "has_citations",
]

BRAND_MENTIONS_COLUMNS: list[str] = [
    "run_id",
    "brand_name",
    "mention_count",
    "first_mention_position",
    "is_recommended",
]

CITATIONS_COLUMNS: list[str] = [
    "run_id",
    "citation_url",
    "citation_domain",
    "citation_position",
]

PAGE_AUDITS_COLUMNS: list[str] = [
    "citation_url",
    "robots_accessible",
    "sitemap_found",
    "canonical_url",
    "page_title",
    "h1_count",
    "h2_count",
    "schema_types",
    "published_date",
    "modified_date",
    "word_count",
    "audit_status",
]

# Allowed controlled-vocabulary values (used by validation + UI dropdowns).
PROMPT_CATEGORIES: list[str] = [
    "Informational",
    "Product comparison",
    "Purchase intent",
    "Problem based",
    "Customer persona",
    "Brand specific",
    "Nonbrand discovery",
]

JOURNEY_STAGES: list[str] = ["Awareness", "Consideration", "Decision", "Retention"]

# DDL for the persistent/in-memory schema. Kept close to the DuckDB type system.
_SCHEMA_DDL: str = """
CREATE TABLE IF NOT EXISTS projects (
    project_id     VARCHAR PRIMARY KEY,
    project_name   VARCHAR,
    industry       VARCHAR,
    created_at     VARCHAR
);

CREATE TABLE IF NOT EXISTS brands (
    brand_id     VARCHAR PRIMARY KEY,
    project_id   VARCHAR,
    brand_name   VARCHAR,
    brand_domain VARCHAR
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id       VARCHAR PRIMARY KEY,
    project_id      VARCHAR,
    prompt_text     VARCHAR,
    prompt_category VARCHAR,
    topic           VARCHAR,
    persona         VARCHAR,
    journey_stage   VARCHAR,
    is_brand_prompt BOOLEAN
);

CREATE TABLE IF NOT EXISTS response_runs (
    run_id        VARCHAR PRIMARY KEY,
    prompt_id     VARCHAR,
    platform      VARCHAR,
    model_name    VARCHAR,
    run_date      VARCHAR,
    run_number    INTEGER,
    response_text VARCHAR,
    has_citations BOOLEAN
);

CREATE TABLE IF NOT EXISTS brand_mentions (
    run_id                 VARCHAR,
    brand_name             VARCHAR,
    mention_count          INTEGER,
    first_mention_position INTEGER,
    is_recommended         BOOLEAN
);

CREATE TABLE IF NOT EXISTS citations (
    run_id            VARCHAR,
    citation_url      VARCHAR,
    citation_domain   VARCHAR,
    citation_position INTEGER
);

CREATE TABLE IF NOT EXISTS page_audits (
    citation_url    VARCHAR,
    robots_accessible BOOLEAN,
    sitemap_found     BOOLEAN,
    canonical_url     VARCHAR,
    page_title        VARCHAR,
    h1_count          INTEGER,
    h2_count          INTEGER,
    schema_types      VARCHAR,
    published_date    VARCHAR,
    modified_date     VARCHAR,
    word_count        INTEGER,
    audit_status      VARCHAR
);
"""


@dataclass
class AnalysisData:
    """Container for the canonical DataFrames that make up one analysis project.

    Every field is a plain pandas DataFrame so the data is easy to inspect, edit
    in the Streamlit UI, and round-trip to CSV. Derived tables (``brand_mentions``,
    ``citations``, ``page_audits``) may start empty and be populated by extraction
    and the page-audit feature.
    """

    projects: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=PROJECTS_COLUMNS))
    brands: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BRANDS_COLUMNS))
    prompts: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=PROMPTS_COLUMNS))
    response_runs: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=RESPONSE_RUNS_COLUMNS)
    )
    brand_mentions: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=BRAND_MENTIONS_COLUMNS)
    )
    citations: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CITATIONS_COLUMNS))
    page_audits: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=PAGE_AUDITS_COLUMNS)
    )

    def table(self, name: str) -> pd.DataFrame:
        """Return a canonical DataFrame by table name (raises KeyError if unknown)."""
        return getattr(self, name)

    def brand_names(self) -> list[str]:
        """Convenience: the list of brand names configured for this project."""
        if self.brands.empty:
            return []
        return sorted(self.brands["brand_name"].dropna().unique().tolist())


class Database:
    """Thin wrapper around a DuckDB connection scoped to one :class:`AnalysisData`.

    Use as a context manager so the connection is always closed::

        with Database.from_analysis_data(data) as db:
            df = db.query("SELECT COUNT(*) FROM prompts")
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Open a DuckDB connection. ``path=':memory:'`` (default) keeps it in RAM."""
        self.con: duckdb.DuckDBPyConnection = duckdb.connect(path)
        self.con.execute(_SCHEMA_DDL)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_analysis_data(cls, data: AnalysisData, path: str = ":memory:") -> "Database":
        """Create a database and register every canonical DataFrame as a table."""
        db = cls(path)
        db.load(data)
        return db

    def load(self, data: AnalysisData) -> None:
        """Replace all tables with the contents of ``data``.

        We register each DataFrame as a temporary DuckDB relation and copy it into
        the typed table so downstream SQL sees correct types (e.g. BOOLEAN).
        """
        mapping: dict[str, pd.DataFrame] = {
            "projects": data.projects,
            "brands": data.brands,
            "prompts": data.prompts,
            "response_runs": data.response_runs,
            "brand_mentions": data.brand_mentions,
            "citations": data.citations,
            "page_audits": data.page_audits,
        }
        for name, df in mapping.items():
            self.con.execute(f"DELETE FROM {name}")
            if df is None or df.empty:
                continue
            # Register the DataFrame under a temp name, then insert matching columns.
            self.con.register(f"_stage_{name}", df)
            cols = ", ".join(df.columns)
            self.con.execute(f"INSERT INTO {name} ({cols}) SELECT {cols} FROM _stage_{name}")
            self.con.unregister(f"_stage_{name}")

    # -- querying ------------------------------------------------------------

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Run SQL and return the result as a pandas DataFrame."""
        if params:
            return self.con.execute(sql, params).fetch_df()
        return self.con.execute(sql).fetch_df()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection."""
        self.con.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_analysis_from_csvs(
    prompts_csv: str,
    responses_csv: str,
    project_name: str = "Demo Project",
    industry: str = "Productivity Software",
    project_id: str = "demo",
) -> AnalysisData:
    """Build an :class:`AnalysisData` from a prompts CSV and a responses CSV.

    Brands are *not* read from these files; callers attach brands separately (the
    demo loader in the app does this). Derived tables start empty and are filled by
    extraction. This function only assembles the raw, user-provided tables.
    """
    prompts = pd.read_csv(prompts_csv, dtype=str).fillna("")
    responses = pd.read_csv(responses_csv, dtype=str).fillna("")

    # Coerce the few typed columns the schema expects.
    if "is_brand_prompt" in prompts.columns:
        prompts["is_brand_prompt"] = _to_bool(prompts["is_brand_prompt"])
    if "run_number" in responses.columns:
        responses["run_number"] = pd.to_numeric(responses["run_number"], errors="coerce").fillna(1).astype(int)
    if "has_citations" in responses.columns:
        responses["has_citations"] = _to_bool(responses["has_citations"])

    projects = pd.DataFrame(
        [{"project_id": project_id, "project_name": project_name, "industry": industry, "created_at": ""}],
        columns=PROJECTS_COLUMNS,
    )
    if "project_id" not in prompts.columns:
        prompts["project_id"] = project_id

    return AnalysisData(
        projects=projects,
        prompts=prompts.reindex(columns=PROMPTS_COLUMNS),
        response_runs=responses.reindex(columns=RESPONSE_RUNS_COLUMNS),
    )


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a string/mixed column of truthy values into real booleans."""
    truthy = {"true", "t", "yes", "y", "1"}
    return series.astype(str).str.strip().str.lower().isin(truthy)
