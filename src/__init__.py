"""AI Visibility Explorer — source package.

Modules
-------
database        DuckDB connection manager + schema (source of truth = CSVs / session DataFrames).
extraction      Deterministic brand-mention and citation extraction (LLM-ready interface).
metrics         Pure functions computing the 12 visibility metrics, each with a definition.
validation      CSV / DataFrame schema validation with human-readable error messages.
recommendations Deterministic, metric-grounded customer-facing readout templates.
page_audit      Optional, polite public-web page inspection for cited URLs.
"""

__version__ = "0.1.0"
