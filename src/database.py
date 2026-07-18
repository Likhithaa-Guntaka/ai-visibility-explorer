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
    # -- AEO question-cluster fields (added in the AEO upgrade) -------------
    "search_intent",     # one of SEARCH_INTENTS; derived from prompt_category when absent
    "question_cluster",  # user-defined cluster label; defaults to the prompt's topic
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
    # -- Real Benchmark Mode fields (added in the upgrade) -------------------
    "dataset_kind",     # one of DATASET_KINDS — keeps real and synthetic separate
    "benchmark_name",   # optional named benchmark this run belongs to
    "collection_date",  # when the user actually collected the response
    "collection_notes", # free-text provenance notes
]

# A benchmark is a named collection of runs. It records the honesty label for the
# whole collection so real and synthetic data never get silently mixed.
BENCHMARKS_COLUMNS: list[str] = [
    "benchmark_name",
    "dataset_kind",     # Synthetic | Real | User Collected
    "created_at",
    "notes",
]

# Deterministically-extracted, user-editable narrative entities per (run, brand).
# Multi-valued fields are stored as "; "-joined strings so they round-trip to CSV
# and are easy to correct in a Streamlit data editor.
BRAND_ENTITIES_COLUMNS: list[str] = [
    "run_id",
    "brand_name",
    "brand_category",
    "products",
    "features",
    "personas",
    "strengths",
    "weaknesses",
    "pricing_positioning",
    "competitors_alongside",
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
    # -- AI Answer Readiness Audit fields (added in the upgrade) -------------
    "h3_count",
    "answer_upfront",         # a substantive paragraph appears before the first H2
    "question_heading_count", # headings phrased as questions (AI answers favour these)
    "has_author",             # author info present (meta/schema/rel=author)
    "external_link_count",    # outbound source links (factual evidence signal)
    # -- Answer Extractability fields (added in the AEO upgrade) ------------
    "list_count",             # <ul>/<ol> elements (scannable, extractable structure)
    "table_count",            # all <table> elements
    "comparison_table_count", # tables with a header row and 3+ columns
    "short_answer_count",     # question headings followed by a short (<=60 word) answer
    "headings_text",          # concatenated H1-H3 text, used for cluster-question coverage
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

# Search intent for AEO clustering. Derived from prompt_category when not supplied,
# so clustering uses existing structured metadata rather than keyword guessing.
SEARCH_INTENTS: list[str] = [
    "Informational",
    "Commercial investigation",
    "Transactional",
    "Problem solving",
    "Navigational / brand",
]

CATEGORY_TO_INTENT: dict[str, str] = {
    "Informational": "Informational",
    "Product comparison": "Commercial investigation",
    "Purchase intent": "Transactional",
    "Problem based": "Problem solving",
    "Customer persona": "Commercial investigation",
    "Brand specific": "Navigational / brand",
    "Nonbrand discovery": "Commercial investigation",
}

# Honesty labels for a dataset. "Synthetic" must never be presented as a real
# platform output; the UI and exports keep these strictly separated.
DATASET_KINDS: list[str] = ["Synthetic", "Real", "User Collected"]

# ---------------------------------------------------------------------------
# AI Decision Influence Lab — vocabularies and table columns (research layer).
# ---------------------------------------------------------------------------

# Recommendation outcome per (response, brand). Deterministic, editable.
OUTCOMES: list[str] = [
    "Mentioned and recommended",
    "Mentioned but not recommended",
    "Compared but rejected",
    "Not mentioned",
]

# Transparent reason categories for a "Compared but rejected" outcome.
REJECTION_REASONS: list[str] = [
    "Pricing concern",
    "Missing capability",
    "Complexity",
    "Ease of use concern",
    "Integration concern",
    "Scalability concern",
    "Persona mismatch",
    "Trust or evidence concern",
    "Competitor advantage",
    "Other or unknown",
]

# Claim types extracted per tracked brand.
CLAIM_TYPES: list[str] = [
    "Product capability",
    "Pricing claim",
    "Positioning claim",
    "Performance claim",
    "Ease of use claim",
    "Customer suitability claim",
    "Limitation",
    "Comparative claim",
]

# Ordered customer decision journey stages (distinct from the AEO JOURNEY_STAGES).
DECISION_JOURNEY_STAGES: list[str] = [
    "Discovery",
    "Consideration",
    "Evaluation",
    "Decision",
    "Retention",
]

# Map the existing prompt journey_stage vocabulary onto the ordered decision journey,
# so journeys can be derived from existing structured metadata by default.
JOURNEY_STAGE_MAP: dict[str, str] = {
    "Awareness": "Discovery",
    "Consideration": "Consideration",
    "Evaluation": "Evaluation",
    "Decision": "Decision",
    "Retention": "Retention",
    # Common synonyms a user might type.
    "Discovery": "Discovery",
    "Post purchase": "Retention",
    "Post-purchase": "Retention",
}

# Whether a journey is a set of independent prompts or a real linked conversation.
JOURNEY_KINDS: list[str] = ["Simulated (independent prompts)", "Linked conversation"]

# Authoritative brand-fact types the Truth & Freshness monitor accepts.
FACT_TYPES: list[str] = [
    "Product name",
    "Pricing",
    "Feature",
    "Integration",
    "Customer segment",
    "Company description",
    "Supported location",
    "Launch date",
    "Discontinued feature",
    "Official source URL",
]

# Comparison verdicts (authoritative-source comparison — NOT absolute truth).
TRUTH_VERDICTS: list[str] = [
    "Supported",
    "Partially supported",
    "Conflicting",
    "Outdated",
    "Unverifiable",
    "Missing from AI responses",
]

RECOMMENDATION_OUTCOMES_COLUMNS: list[str] = [
    "run_id",
    "brand_name",
    "outcome",
    "reason_categories",   # "; "-joined REJECTION_REASONS (only for rejected)
    "evidence_text",       # the exact text snippet the rule keyed on
]

BRAND_CLAIMS_COLUMNS: list[str] = [
    "claim_id",
    "run_id",
    "brand_name",
    "claim_type",
    "claim_text",          # the observed statement
    "evidence_text",       # the surrounding block used as evidence
]

BRAND_FACTS_COLUMNS: list[str] = [
    "brand_name",
    "fact_type",
    "fact_value",
    "source_url",
    "as_of_date",
]

JOURNEYS_COLUMNS: list[str] = [
    "journey_id",
    "journey_name",
    "journey_kind",
    "stage",               # one of DECISION_JOURNEY_STAGES
    "stage_order",         # integer 1..5
    "prompt_id",
]

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
    is_brand_prompt BOOLEAN,
    search_intent    VARCHAR,
    question_cluster VARCHAR
);

CREATE TABLE IF NOT EXISTS response_runs (
    run_id        VARCHAR PRIMARY KEY,
    prompt_id     VARCHAR,
    platform      VARCHAR,
    model_name    VARCHAR,
    run_date      VARCHAR,
    run_number    INTEGER,
    response_text VARCHAR,
    has_citations BOOLEAN,
    dataset_kind     VARCHAR,
    benchmark_name   VARCHAR,
    collection_date  VARCHAR,
    collection_notes VARCHAR
);

CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_name VARCHAR,
    dataset_kind   VARCHAR,
    created_at     VARCHAR,
    notes          VARCHAR
);

CREATE TABLE IF NOT EXISTS brand_entities (
    run_id                VARCHAR,
    brand_name            VARCHAR,
    brand_category        VARCHAR,
    products              VARCHAR,
    features              VARCHAR,
    personas              VARCHAR,
    strengths             VARCHAR,
    weaknesses            VARCHAR,
    pricing_positioning   VARCHAR,
    competitors_alongside VARCHAR
);

CREATE TABLE IF NOT EXISTS recommendation_outcomes (
    run_id            VARCHAR,
    brand_name        VARCHAR,
    outcome           VARCHAR,
    reason_categories VARCHAR,
    evidence_text     VARCHAR
);

CREATE TABLE IF NOT EXISTS brand_claims (
    claim_id      VARCHAR,
    run_id        VARCHAR,
    brand_name    VARCHAR,
    claim_type    VARCHAR,
    claim_text    VARCHAR,
    evidence_text VARCHAR
);

CREATE TABLE IF NOT EXISTS brand_facts (
    brand_name VARCHAR,
    fact_type  VARCHAR,
    fact_value VARCHAR,
    source_url VARCHAR,
    as_of_date VARCHAR
);

CREATE TABLE IF NOT EXISTS journeys (
    journey_id   VARCHAR,
    journey_name VARCHAR,
    journey_kind VARCHAR,
    stage        VARCHAR,
    stage_order  INTEGER,
    prompt_id    VARCHAR
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
    audit_status      VARCHAR,
    h3_count          INTEGER,
    answer_upfront    BOOLEAN,
    question_heading_count INTEGER,
    has_author        BOOLEAN,
    external_link_count    INTEGER,
    list_count             INTEGER,
    table_count            INTEGER,
    comparison_table_count INTEGER,
    short_answer_count     INTEGER,
    headings_text          VARCHAR
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
    benchmarks: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=BENCHMARKS_COLUMNS)
    )
    brand_entities: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=BRAND_ENTITIES_COLUMNS)
    )
    # -- AI Decision Influence Lab tables (research layer) ------------------
    recommendation_outcomes: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=RECOMMENDATION_OUTCOMES_COLUMNS)
    )
    brand_claims: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=BRAND_CLAIMS_COLUMNS)
    )
    brand_facts: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=BRAND_FACTS_COLUMNS)
    )
    journeys: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=JOURNEYS_COLUMNS)
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
            "benchmarks": data.benchmarks,
            "brand_entities": data.brand_entities,
            "recommendation_outcomes": data.recommendation_outcomes,
            "brand_claims": data.brand_claims,
            "brand_facts": data.brand_facts,
            "journeys": data.journeys,
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

    # Default the Real-Benchmark fields when a CSV does not provide them. A dataset
    # with no explicit label is treated as "Synthetic" so it is never mistaken for
    # real platform output.
    if "dataset_kind" not in responses.columns or responses["dataset_kind"].eq("").all():
        responses["dataset_kind"] = "Synthetic"
    responses["dataset_kind"] = responses["dataset_kind"].replace("", "Synthetic")
    for col, default in [("benchmark_name", ""), ("collection_date", ""), ("collection_notes", "")]:
        if col not in responses.columns:
            responses[col] = default

    projects = pd.DataFrame(
        [{"project_id": project_id, "project_name": project_name, "industry": industry, "created_at": ""}],
        columns=PROJECTS_COLUMNS,
    )
    if "project_id" not in prompts.columns:
        prompts["project_id"] = project_id

    prompts = ensure_prompt_cluster_columns(prompts)
    return AnalysisData(
        projects=projects,
        prompts=prompts.reindex(columns=PROMPTS_COLUMNS),
        response_runs=responses.reindex(columns=RESPONSE_RUNS_COLUMNS),
    )


def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a string/mixed column of truthy values into real booleans."""
    truthy = {"true", "t", "yes", "y", "1"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


def ensure_prompt_cluster_columns(prompts: pd.DataFrame) -> pd.DataFrame:
    """Guarantee ``search_intent`` and ``question_cluster`` exist and are populated.

    Uses existing structured metadata rather than keyword guessing:
    * ``search_intent`` defaults from ``prompt_category`` via :data:`CATEGORY_TO_INTENT`.
    * ``question_cluster`` defaults to the prompt's ``topic``.
    Existing user-supplied values are always preserved.
    """
    if prompts is None or prompts.empty:
        out = prompts.copy() if prompts is not None else pd.DataFrame(columns=PROMPTS_COLUMNS)
        for col in ("search_intent", "question_cluster"):
            if col not in out.columns:
                out[col] = pd.Series(dtype=str)
        return out

    out = prompts.copy()
    if "search_intent" not in out.columns:
        out["search_intent"] = ""
    if "question_cluster" not in out.columns:
        out["question_cluster"] = ""

    out["search_intent"] = out["search_intent"].fillna("").astype(str).str.strip()
    out["question_cluster"] = out["question_cluster"].fillna("").astype(str).str.strip()

    category = out["prompt_category"] if "prompt_category" in out.columns else pd.Series([""] * len(out), index=out.index)
    derived_intent = category.map(lambda c: CATEGORY_TO_INTENT.get(str(c), "Informational"))
    out["search_intent"] = out["search_intent"].where(out["search_intent"] != "", derived_intent)

    topic = out["topic"] if "topic" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["question_cluster"] = out["question_cluster"].where(
        out["question_cluster"] != "", topic.fillna("").astype(str)
    )
    return out
