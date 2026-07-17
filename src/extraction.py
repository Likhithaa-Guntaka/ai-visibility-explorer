"""Deterministic brand-mention and citation extraction.

The MVP uses transparent, rule-based extraction so that every number in the
dashboard can be traced back to a concrete string match — no black boxes. The
:class:`Extractor` interface is deliberately small so an LLM-based extractor can be
dropped in later (Phase 2) without touching the metrics or the UI.

Key definitions
---------------
first_mention_position
    The **character offset** of a brand's earliest match in the response text.
    The brand with the smallest offset in a run is that run's "first mention".
is_recommended
    A transparent heuristic: the brand appears close to a recommendation cue
    (e.g. "recommend", "best", "top pick", "great starting point"), or it is the
    single top-ranked item in a numbered list. This is a signal, not a guarantee;
    the UI lets users correct it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from .database import BRAND_MENTIONS_COLUMNS, CITATIONS_COLUMNS

# A URL matcher covering http/https links, including trailing paths and queries.
_URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)

# Cue words that suggest a brand is being recommended, checked in a text window.
_RECOMMENDATION_CUES = [
    "recommend",
    "best choice",
    "best option",
    "best starting point",
    "great starting point",
    "top pick",
    "our pick",
    "i'd suggest",
    "i would suggest",
    "go with",
    "start with",
    "is a great",
    "is the best",
    "would recommend",
]

# How many characters on each side of a brand mention we scan for a cue word.
_RECOMMENDATION_WINDOW = 90


@dataclass
class ExtractionResult:
    """Extraction output for a single response run."""

    mentions: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)


class Extractor(ABC):
    """Interface for anything that turns response text into mentions + citations.

    Implement :meth:`extract_run` to add a new strategy (e.g. an LLM extractor).
    The rest of the app depends only on this interface.
    """

    @abstractmethod
    def extract_run(
        self,
        run_id: str,
        response_text: str,
        brand_aliases: dict[str, list[str]],
    ) -> ExtractionResult:
        """Extract mentions and citations for one response.

        Parameters
        ----------
        run_id
            Identifier of the response run (foreign key into ``response_runs``).
        response_text
            The raw AI response text.
        brand_aliases
            Mapping of canonical brand name -> list of aliases (may be empty). The
            canonical name is always treated as an alias of itself.
        """
        raise NotImplementedError


class DeterministicExtractor(Extractor):
    """Rule-based extractor: case-insensitive matching + URL parsing. The default."""

    def extract_run(
        self,
        run_id: str,
        response_text: str,
        brand_aliases: dict[str, list[str]],
    ) -> ExtractionResult:
        text = response_text or ""
        mentions = self._extract_mentions(run_id, text, brand_aliases)
        citations = self._extract_citations(run_id, text)
        return ExtractionResult(mentions=mentions, citations=citations)

    # -- brand mentions ------------------------------------------------------

    def _extract_mentions(
        self, run_id: str, text: str, brand_aliases: dict[str, list[str]]
    ) -> list[dict]:
        lowered = text.lower()
        rows: list[dict] = []
        for brand, aliases in brand_aliases.items():
            terms = _unique_terms([brand, *aliases])
            count = 0
            first_pos: Optional[int] = None
            for term in terms:
                for match in _iter_term_matches(lowered, term.lower()):
                    count += 1
                    if first_pos is None or match < first_pos:
                        first_pos = match
            if count == 0:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "brand_name": brand,
                    "mention_count": count,
                    "first_mention_position": int(first_pos) if first_pos is not None else -1,
                    "is_recommended": self._is_recommended(text, lowered, terms, first_pos),
                }
            )
        return rows

    def _is_recommended(
        self,
        text: str,
        lowered: str,
        terms: list[str],
        first_pos: Optional[int],
    ) -> bool:
        """Heuristic recommendation flag — see module docstring for the definition."""
        if first_pos is None:
            return False
        # 1) A cue word appears within a window around any mention of the brand.
        for term in terms:
            for pos in _iter_term_matches(lowered, term.lower()):
                start = max(0, pos - _RECOMMENDATION_WINDOW)
                end = min(len(lowered), pos + len(term) + _RECOMMENDATION_WINDOW)
                window = lowered[start:end]
                if any(cue in window for cue in _RECOMMENDATION_CUES):
                    return True
        # 2) The brand heads a numbered list ("1. **Brand** ...").
        for term in terms:
            if re.search(rf"(^|\n)\s*1[.)]\s*\**{re.escape(term)}", text, re.IGNORECASE):
                return True
        return False

    # -- citations -----------------------------------------------------------

    def _extract_citations(self, run_id: str, text: str) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()
        position = 0
        for match in _URL_RE.finditer(text):
            url = _clean_url(match.group(0))
            if not url or url in seen:
                continue
            seen.add(url)
            position += 1
            rows.append(
                {
                    "run_id": run_id,
                    "citation_url": url,
                    "citation_domain": normalize_domain(url),
                    "citation_position": position,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Batch helpers used by the app / tests.
# ---------------------------------------------------------------------------


def build_alias_map(brands_df: pd.DataFrame, alias_overrides: Optional[dict[str, list[str]]] = None) -> dict[str, list[str]]:
    """Build a ``{brand_name: [aliases]}`` map from a brands DataFrame.

    ``alias_overrides`` lets the UI attach user-entered aliases (e.g. "Monday" for
    "Monday.com"). A sensible built-in alias is added for Monday.com automatically.
    """
    alias_overrides = alias_overrides or {}
    default_aliases = {"Monday.com": ["Monday", "monday.com"]}
    result: dict[str, list[str]] = {}
    for brand in brands_df["brand_name"].dropna().unique():
        aliases = list(default_aliases.get(brand, []))
        aliases.extend(alias_overrides.get(brand, []))
        result[brand] = _unique_terms(aliases)
    return result


def extract_all(
    response_runs: pd.DataFrame,
    brand_aliases: dict[str, list[str]],
    extractor: Optional[Extractor] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run extraction over every response and return (brand_mentions, citations).

    Returns two DataFrames matching the canonical schemas. Empty inputs yield empty,
    correctly-typed DataFrames.
    """
    extractor = extractor or DeterministicExtractor()
    mention_rows: list[dict] = []
    citation_rows: list[dict] = []
    for _, run in response_runs.iterrows():
        result = extractor.extract_run(
            run_id=str(run["run_id"]),
            response_text=str(run.get("response_text", "")),
            brand_aliases=brand_aliases,
        )
        mention_rows.extend(result.mentions)
        citation_rows.extend(result.citations)

    mentions_df = pd.DataFrame(mention_rows, columns=BRAND_MENTIONS_COLUMNS)
    citations_df = pd.DataFrame(citation_rows, columns=CITATIONS_COLUMNS)
    return mentions_df, citations_df


# ---------------------------------------------------------------------------
# Small pure helpers.
# ---------------------------------------------------------------------------


def normalize_domain(url: str) -> str:
    """Return a normalized domain for a URL: lowercase, no ``www.``, no port.

    >>> normalize_domain("https://www.G2.com:443/best")
    'g2.com'
    """
    try:
        netloc = urlparse(url).netloc.lower()
    except (ValueError, AttributeError):
        return ""
    netloc = netloc.split("@")[-1]  # strip any credentials
    netloc = netloc.split(":")[0]  # strip port
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _clean_url(raw: str) -> str:
    """Trim trailing punctuation that commonly clings to URLs in prose."""
    return raw.rstrip(".,;:!?’\"')")


def _unique_terms(terms: list[str]) -> list[str]:
    """De-duplicate terms case-insensitively while preserving order and dropping blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _iter_term_matches(lowered_text: str, lowered_term: str):
    """Yield character offsets of each match of ``lowered_term`` in ``lowered_text``.

    Uses a word-ish boundary so "Trello" does not match inside "Trellos" but brand
    names containing punctuation (e.g. "monday.com") still match. We escape the term
    and require the preceding/following char to not be a letter or digit.
    """
    if not lowered_term:
        return
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(lowered_term)}(?![a-z0-9])")
    for m in pattern.finditer(lowered_text):
        yield m.start()
