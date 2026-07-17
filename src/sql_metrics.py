"""DuckDB SQL implementations of the core visibility metrics.

This module makes DuckDB a *genuine part of the runtime*: the Streamlit dashboard and
citation page compute their headline metrics with the SQL in ``sql/*.sql``, executed
against an in-memory DuckDB database built from the active :class:`AnalysisData`.

Which metrics run in SQL (this module)
--------------------------------------
brand mention rate · share of voice · first mention share · recommendation rate ·
citation rate · source domain share · platform comparison · competitor visibility ·
prompt category performance · persona performance (the last two via
``visibility_by_attribute``).

Which metrics stay in pandas (``src/metrics.py``) and why
--------------------------------------------------------
Content coverage gaps, response/narrative consistency, citation diversity /
concentration / opportunities, entity extraction, clusters, experiments and briefs
stay in pandas: they rely on pairwise set operations (Jaccard), per-run Python objects,
lexicon matching, and cross-table reshaping that are clearer and safer in pandas than
in SQL. Pandas also remains the layer for transformation, Streamlit display and charts.

The SQL definitions and denominators are identical to the pandas reference functions in
:mod:`src.metrics`; ``tests/test_sql_metrics.py`` asserts equivalence value-for-value.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

import pandas as pd

from .database import AnalysisData, Database

SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql")

# Prompt attribute columns that may be substituted into visibility_by_attribute.
# Strictly allow-listed so the {attr} substitution can never be SQL injection.
ALLOWED_ATTRIBUTES: set[str] = {
    "prompt_category", "persona", "topic", "journey_stage", "search_intent", "question_cluster",
}


@lru_cache(maxsize=None)
def _load_queries(filename: str) -> dict[str, str]:
    """Parse a ``sql/<filename>`` into a ``{name: query}`` dict using ``-- name:`` markers."""
    path = os.path.join(SQL_DIR, filename)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    queries: dict[str, str] = {}
    current: Optional[str] = None
    buffer: list[str] = []
    for line in text.splitlines():
        marker = re.match(r"^\s*--\s*name:\s*(\w+)\s*$", line)
        if marker:
            if current is not None:
                queries[current] = "\n".join(buffer).strip()
            current = marker.group(1)
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        queries[current] = "\n".join(buffer).strip()
    return queries


def load_query(filename: str, name: str) -> str:
    """Return one named query from a SQL file (raises KeyError if missing)."""
    return _load_queries(filename)[name]


class SqlMetrics:
    """Compute visibility metrics with DuckDB SQL over one :class:`AnalysisData`.

    Open once per analysis view and reuse the connection for every metric::

        with SqlMetrics(data) as m:
            sov = m.share_of_voice()
    """

    def __init__(self, data: AnalysisData) -> None:
        self.data = data
        self.db = Database.from_analysis_data(data)

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "SqlMetrics":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- helpers -------------------------------------------------------------
    def _has_runs(self) -> bool:
        return not self.data.response_runs.empty

    def _has_mentions(self) -> bool:
        return not self.data.brand_mentions.empty

    # -- brand visibility ----------------------------------------------------
    def brand_mention_rate(self) -> pd.DataFrame:
        cols = ["brand_name", "mentioned_runs", "total_runs", "mention_rate"]
        if not self._has_mentions() or not self._has_runs():
            return pd.DataFrame(columns=cols)
        return self.db.query(load_query("brand_visibility.sql", "brand_mention_rate"))

    def share_of_voice(self) -> pd.DataFrame:
        cols = ["brand_name", "mentions", "share_of_voice"]
        if not self._has_mentions():
            return pd.DataFrame(columns=cols)
        return self.db.query(load_query("brand_visibility.sql", "share_of_voice"))

    def recommendation_rate(self) -> pd.DataFrame:
        cols = ["brand_name", "recommended_runs", "total_runs", "recommendation_rate"]
        if not self._has_mentions() or not self._has_runs():
            return pd.DataFrame(columns=cols)
        return self.db.query(load_query("brand_visibility.sql", "recommendation_rate"))

    def first_mention_share(self) -> pd.DataFrame:
        cols = ["brand_name", "first_mentions", "mentioned_runs", "first_mention_share"]
        if not self._has_mentions():
            return pd.DataFrame(columns=cols)
        out = self.db.query(load_query("brand_visibility.sql", "first_mention_share"))
        return out if not out.empty else pd.DataFrame(columns=cols)

    def competitor_visibility(self) -> pd.DataFrame:
        cols = ["brand_name", "mention_rate", "share_of_voice", "recommendation_rate"]
        if not self._has_mentions() or not self._has_runs():
            return pd.DataFrame(columns=cols)
        return self.db.query(load_query("brand_visibility.sql", "competitor_visibility"))

    # -- citations -----------------------------------------------------------
    def citation_rate(self) -> dict[str, float]:
        if not self._has_runs():
            return {"runs_with_citations": 0, "total_runs": 0, "citation_rate": 0.0}
        row = self.db.query(load_query("citation_metrics.sql", "citation_rate")).iloc[0]
        n = int(row["total_runs"])
        runs_with = int(row["runs_with_citations"])
        return {
            "runs_with_citations": runs_with,
            "total_runs": n,
            "citation_rate": (runs_with / n) if n else 0.0,
        }

    def source_domain_share(self, top_n: Optional[int] = None) -> pd.DataFrame:
        cols = ["citation_domain", "citations", "runs", "domain_share"]
        if self.data.citations.empty:
            return pd.DataFrame(columns=cols)
        out = self.db.query(load_query("citation_metrics.sql", "source_domain_share"))
        return out.head(top_n) if top_n else out

    # -- segment performance -------------------------------------------------
    def visibility_by_attribute(self, attribute: str, brand_name: str) -> pd.DataFrame:
        out_cols = [attribute, "mentioned_runs", "total_runs", "mention_rate"]
        if attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(
                f"Attribute {attribute!r} is not allowed. Choose one of {sorted(ALLOWED_ATTRIBUTES)}."
            )
        if self.data.response_runs.empty or self.data.prompts.empty:
            return pd.DataFrame(columns=out_cols)
        sql = load_query("segment_performance.sql", "visibility_by_attribute").replace("{attr}", attribute)
        return self.db.query(sql, params=[brand_name])

    def platform_comparison(self, brand_name: str) -> pd.DataFrame:
        out_cols = ["platform", "mentioned_runs", "total_runs", "mention_rate"]
        if self.data.response_runs.empty:
            return pd.DataFrame(columns=out_cols)
        return self.db.query(
            load_query("segment_performance.sql", "platform_comparison"), params=[brand_name]
        )


# ---------------------------------------------------------------------------
# Drop-in functions mirroring src.metrics signatures (used by equivalence tests).
# ---------------------------------------------------------------------------


def brand_mention_rate(brand_mentions: pd.DataFrame, response_runs: pd.DataFrame) -> pd.DataFrame:
    with SqlMetrics(AnalysisData(brand_mentions=brand_mentions, response_runs=response_runs)) as m:
        return m.brand_mention_rate()


def share_of_voice(brand_mentions: pd.DataFrame) -> pd.DataFrame:
    with SqlMetrics(AnalysisData(brand_mentions=brand_mentions)) as m:
        return m.share_of_voice()


def recommendation_rate(brand_mentions: pd.DataFrame, response_runs: pd.DataFrame) -> pd.DataFrame:
    with SqlMetrics(AnalysisData(brand_mentions=brand_mentions, response_runs=response_runs)) as m:
        return m.recommendation_rate()


def first_mention_share(brand_mentions: pd.DataFrame) -> pd.DataFrame:
    with SqlMetrics(AnalysisData(brand_mentions=brand_mentions)) as m:
        return m.first_mention_share()


def citation_rate(citations: pd.DataFrame, response_runs: pd.DataFrame) -> dict[str, float]:
    with SqlMetrics(AnalysisData(citations=citations, response_runs=response_runs)) as m:
        return m.citation_rate()


def source_domain_share(citations: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    with SqlMetrics(AnalysisData(citations=citations)) as m:
        return m.source_domain_share(top_n=top_n)


def competitor_visibility(brand_mentions: pd.DataFrame, response_runs: pd.DataFrame) -> pd.DataFrame:
    with SqlMetrics(AnalysisData(brand_mentions=brand_mentions, response_runs=response_runs)) as m:
        return m.competitor_visibility()


def visibility_by_attribute(
    brand_mentions: pd.DataFrame,
    response_runs: pd.DataFrame,
    prompts: pd.DataFrame,
    attribute: str,
    brand_name: str,
) -> pd.DataFrame:
    data = AnalysisData(brand_mentions=brand_mentions, response_runs=response_runs, prompts=prompts)
    with SqlMetrics(data) as m:
        return m.visibility_by_attribute(attribute, brand_name)


def platform_comparison(
    brand_mentions: pd.DataFrame, response_runs: pd.DataFrame, brand_name: str
) -> pd.DataFrame:
    with SqlMetrics(AnalysisData(brand_mentions=brand_mentions, response_runs=response_runs)) as m:
        return m.platform_comparison(brand_name)
