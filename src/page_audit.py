"""Optional public-web page audit for cited URLs.

For each cited page we *attempt* to observe simple, public SEO/content signals:
robots.txt accessibility, sitemap presence, canonical URL, title, heading structure,
JSON-LD / schema.org types, publish/modify dates, and an approximate word count.

Design principles
-----------------
* **Polite & honest.** A clear User-Agent, a short timeout, and no aggressive
  crawling. We fetch each page once.
* **Fails gracefully.** Any network error, timeout, block, or missing field is
  recorded in ``audit_status`` and the row is still returned with best-effort fields.
* **Association, not causation.** These signals are *descriptive*. Nothing here proves
  that a page trait caused an AI citation; the UI states this explicitly.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import pandas as pd

from .database import PAGE_AUDITS_COLUMNS
from .extraction import normalize_domain

try:  # requests / bs4 are required deps, but guard so importing never hard-fails.
    import requests
    from bs4 import BeautifulSoup

    _HAVE_WEB = True
except Exception:  # pragma: no cover - only in a broken env
    _HAVE_WEB = False


DEFAULT_USER_AGENT = os.environ.get(
    "AIVX_USER_AGENT",
    "AIVisibilityExplorer/0.1 (+portfolio-project; contact via GitHub)",
)
DEFAULT_TIMEOUT = float(os.environ.get("AIVX_REQUEST_TIMEOUT", "10"))


@dataclass
class PageAudit:
    """One page's audit result, matching the ``page_audits`` schema."""

    citation_url: str
    robots_accessible: Optional[bool] = None
    sitemap_found: Optional[bool] = None
    canonical_url: Optional[str] = None
    page_title: Optional[str] = None
    h1_count: Optional[int] = None
    h2_count: Optional[int] = None
    schema_types: Optional[str] = None
    published_date: Optional[str] = None
    modified_date: Optional[str] = None
    word_count: Optional[int] = None
    audit_status: str = "not_run"
    # -- AI Answer Readiness fields ----------------------------------------
    h3_count: Optional[int] = None
    answer_upfront: Optional[bool] = None
    question_heading_count: Optional[int] = None
    has_author: Optional[bool] = None
    external_link_count: Optional[int] = None
    # -- Answer Extractability fields --------------------------------------
    list_count: Optional[int] = None
    table_count: Optional[int] = None
    comparison_table_count: Optional[int] = None
    short_answer_count: Optional[int] = None
    headings_text: Optional[str] = None


def audit_url(url: str, session: "requests.Session | None" = None) -> PageAudit:
    """Audit a single URL, never raising — problems are captured in ``audit_status``.

    ``audit_status`` values: ``ok``, ``blocked``, ``timeout``, ``error``,
    ``unavailable`` (web libs missing), ``invalid_url``.
    """
    audit = PageAudit(citation_url=url)
    if not _HAVE_WEB:
        audit.audit_status = "unavailable"
        return audit
    if not _looks_like_url(url):
        audit.audit_status = "invalid_url"
        return audit

    own_session = session is None
    session = session or _make_session()
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        # robots.txt + sitemap discovery (best-effort; failures are non-fatal).
        audit.robots_accessible, sitemap_from_robots = _check_robots(session, base)
        audit.sitemap_found = _check_sitemap(session, base, sitemap_from_robots)

        resp = session.get(url, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        if resp.status_code in (401, 403, 429):
            audit.audit_status = "blocked"
            return audit
        if resp.status_code >= 400:
            audit.audit_status = f"http_{resp.status_code}"
            return audit

        soup = BeautifulSoup(resp.text, "html.parser")
        _extract_page_signals(soup, base, audit)
        audit.audit_status = "ok"
    except requests.exceptions.Timeout:
        audit.audit_status = "timeout"
    except requests.exceptions.RequestException:
        audit.audit_status = "error"
    except Exception:  # pragma: no cover - defensive catch-all
        audit.audit_status = "error"
    finally:
        if own_session:
            session.close()
    return audit


def audit_urls(urls: list[str]) -> pd.DataFrame:
    """Audit a list of URLs and return a DataFrame matching the ``page_audits`` schema."""
    unique = list(dict.fromkeys(u for u in urls if u))  # de-dupe, preserve order
    if not unique:
        return pd.DataFrame(columns=PAGE_AUDITS_COLUMNS)
    if not _HAVE_WEB:
        rows = [asdict(PageAudit(citation_url=u, audit_status="unavailable")) for u in unique]
        return pd.DataFrame(rows, columns=PAGE_AUDITS_COLUMNS)
    session = _make_session()
    try:
        rows = [asdict(audit_url(u, session=session)) for u in unique]
    finally:
        session.close()
    return pd.DataFrame(rows, columns=PAGE_AUDITS_COLUMNS)


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _make_session() -> "requests.Session":
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    return session


def _looks_like_url(url: str) -> bool:
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except (ValueError, AttributeError):
        return False


def _check_robots(session, base: str) -> tuple[Optional[bool], Optional[str]]:
    """Return (robots_accessible, sitemap_url_from_robots)."""
    try:
        resp = session.get(urljoin(base, "/robots.txt"), timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200 and "user-agent" in resp.text.lower():
            sitemap_match = re.search(r"(?im)^\s*sitemap:\s*(\S+)", resp.text)
            return True, (sitemap_match.group(1) if sitemap_match else None)
        return False, None
    except requests.exceptions.RequestException:
        return None, None


def _check_sitemap(session, base: str, sitemap_from_robots: Optional[str]) -> Optional[bool]:
    """Check whether an XML sitemap appears to exist."""
    candidates = [c for c in [sitemap_from_robots, urljoin(base, "/sitemap.xml")] if c]
    for candidate in candidates:
        try:
            resp = session.get(candidate, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200 and ("<urlset" in resp.text or "<sitemapindex" in resp.text):
                return True
        except requests.exceptions.RequestException:
            continue
    return False


def _extract_page_signals(soup: "BeautifulSoup", base: str, audit: PageAudit) -> None:
    """Populate content/SEO fields on ``audit`` from parsed HTML."""
    # Canonical URL.
    canonical = soup.find("link", rel=lambda v: v and "canonical" in v)
    if canonical and canonical.get("href"):
        audit.canonical_url = urljoin(base, canonical["href"])

    # Title.
    if soup.title and soup.title.string:
        audit.page_title = soup.title.string.strip()

    # Heading structure.
    h1s, h2s, h3s = soup.find_all("h1"), soup.find_all("h2"), soup.find_all("h3")
    audit.h1_count = len(h1s)
    audit.h2_count = len(h2s)
    audit.h3_count = len(h3s)

    # Question-based headings (AI answers favour pages that answer explicit questions).
    audit.question_heading_count = sum(1 for h in (h1s + h2s + h3s) if _is_question(h.get_text(" ", strip=True)))

    # JSON-LD / schema.org types.
    audit.schema_types = _collect_schema_types(soup)

    # Published / modified dates from common meta tags.
    audit.published_date = _first_meta(
        soup,
        [("property", "article:published_time"), ("name", "publishdate"), ("itemprop", "datePublished")],
    )
    audit.modified_date = _first_meta(
        soup,
        [("property", "article:modified_time"), ("name", "lastmod"), ("itemprop", "dateModified")],
    )

    # Author information (meta author, rel=author, or schema author).
    author_meta = _first_meta(soup, [("name", "author"), ("property", "article:author")])
    rel_author = soup.find(attrs={"rel": lambda v: v and "author" in v})
    audit.has_author = bool(author_meta or rel_author or "author" in (audit.schema_types or "").lower())

    # Outbound source links (a factual-evidence signal), counted off-domain only.
    audit.external_link_count = _count_external_links(soup, base)

    # A substantive direct answer near the top: a paragraph with enough words that
    # appears before the first H2.
    audit.answer_upfront = _has_answer_upfront(soup)

    # -- Answer Extractability signals -------------------------------------
    # Lists and tables make facts easy for an answer engine to lift out.
    audit.list_count = len(soup.find_all(["ul", "ol"]))
    tables = soup.find_all("table")
    audit.table_count = len(tables)
    audit.comparison_table_count = sum(1 for t in tables if _looks_like_comparison_table(t))

    # Short answer sections: a question heading followed by a concise answer.
    audit.short_answer_count = _count_short_answer_sections(h1s + h2s + h3s)

    # Heading text is kept so cluster-question coverage can be checked later.
    heading_texts = [h.get_text(" ", strip=True) for h in (h1s + h2s + h3s)]
    audit.headings_text = " | ".join(t for t in heading_texts if t)[:4000]

    # Approximate word count from visible body text.
    body_text = soup.get_text(" ", strip=True)
    audit.word_count = len(body_text.split()) if body_text else 0


_QUESTION_WORDS = ("how", "what", "why", "which", "when", "who", "where", "is", "are", "can", "should", "do", "does")


def _is_question(text: str) -> bool:
    """Heuristic: heading is a question if it ends with '?' or starts with a question word."""
    t = (text or "").strip().lower()
    if not t:
        return False
    return t.endswith("?") or t.split()[0] in _QUESTION_WORDS


def _count_external_links(soup: "BeautifulSoup", base: str) -> int:
    """Count anchor tags linking off the page's own host."""
    host = urlparse(base).netloc.lower().removeprefix("www.")
    count = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http"):
            link_host = urlparse(href).netloc.lower().removeprefix("www.")
            if link_host and link_host != host:
                count += 1
    return count


def _has_answer_upfront(soup: "BeautifulSoup", min_words: int = 20) -> bool:
    """True if a paragraph of >= ``min_words`` occurs before the first H2 heading."""
    first_h2 = soup.find("h2")
    for p in soup.find_all("p"):
        if first_h2 is not None and _appears_after(p, first_h2):
            break
        if len(p.get_text(" ", strip=True).split()) >= min_words:
            return True
    return False


def _appears_after(node, other) -> bool:
    """True if ``node`` occurs after ``other`` in document order (best-effort)."""
    for elem in other.find_all_previous():
        if elem is node:
            return False
    return True


def _looks_like_comparison_table(table) -> bool:
    """Heuristic: a table with a header row and 3+ columns reads as a comparison table."""
    headers = table.find_all("th")
    if len(headers) >= 3:
        return True
    first_row = table.find("tr")
    if first_row is not None and len(first_row.find_all(["td", "th"])) >= 3:
        return True
    return False


def _count_short_answer_sections(headings, max_words: int = 60) -> int:
    """Count question headings immediately followed by a concise (<= max_words) answer.

    A short answer directly under a question heading is the single most extractable
    shape for an answer engine.
    """
    count = 0
    for h in headings:
        if not _is_question(h.get_text(" ", strip=True)):
            continue
        sibling = h.find_next_sibling()
        # Skip over empty wrappers to the first content element.
        hops = 0
        while sibling is not None and hops < 3:
            text = sibling.get_text(" ", strip=True)
            if text:
                if 0 < len(text.split()) <= max_words:
                    count += 1
                break
            sibling = sibling.find_next_sibling()
            hops += 1
    return count


def question_coverage(headings_text: Optional[str], page_title: Optional[str],
                      questions: list[str], min_overlap: float = 0.5) -> dict:
    """Share of a cluster's questions that appear to be covered by the page's headings.

    A question counts as covered when at least ``min_overlap`` of its meaningful words
    (stop-words removed) appear in the page's headings or title. This is a transparent
    word-overlap check on *headings*, not a semantic judgement — it can miss a question
    that is answered in body copy under a differently-worded heading.

    Returns ``covered``, ``total``, ``coverage`` and the per-question ``detail``.
    """
    haystack = f"{headings_text or ''} {page_title or ''}".lower()
    total = len(questions)
    if total == 0:
        return {"covered": 0, "total": 0, "coverage": None, "detail": []}
    detail, covered = [], 0
    for q in questions:
        words = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in _STOPWORDS and len(w) > 2]
        if not words:
            detail.append({"question": q, "covered": False, "overlap": 0.0})
            continue
        hits = sum(1 for w in set(words) if w in haystack)
        overlap = hits / len(set(words))
        is_covered = overlap >= min_overlap
        covered += int(is_covered)
        detail.append({"question": q, "covered": is_covered, "overlap": overlap})
    return {"covered": covered, "total": total, "coverage": covered / total, "detail": detail}


_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "is", "are", "do", "does",
    "what", "which", "how", "why", "when", "who", "should", "can", "my", "our", "we", "i",
    "with", "that", "this", "it", "be", "best", "you", "your", "me", "am", "have", "has",
}


def _collect_schema_types(soup: "BeautifulSoup") -> Optional[str]:
    types: list[str] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for obj in payload if isinstance(payload, list) else [payload]:
            if isinstance(obj, dict) and "@type" in obj:
                t = obj["@type"]
                types.extend(t if isinstance(t, list) else [t])
    unique = sorted(set(str(t) for t in types))
    return ", ".join(unique) if unique else None


def _first_meta(soup: "BeautifulSoup", attrs_list: list[tuple[str, str]]) -> Optional[str]:
    for attr, value in attrs_list:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def summarize_audits(audits: pd.DataFrame, citations: pd.DataFrame) -> pd.DataFrame:
    """Join audits to how often each domain was cited (association, not causation).

    Returns a per-URL frame with ``citation_domain`` and ``times_cited`` so the UI can
    show which frequently-cited pages have which technical traits — without claiming
    the traits caused the citations.
    """
    if audits.empty:
        return audits
    out = audits.copy()
    out["citation_domain"] = out["citation_url"].map(normalize_domain)
    if not citations.empty:
        counts = citations.groupby("citation_url")["run_id"].size().rename("times_cited")
        out = out.merge(counts, on="citation_url", how="left")
        out["times_cited"] = out["times_cited"].fillna(0).astype(int)
    else:
        out["times_cited"] = 0
    return out


# ---------------------------------------------------------------------------
# AI Answer Readiness Audit — each factor reported separately.
# The optional summary score is fully transparent: weights and per-factor points
# are always shown. There is deliberately NO opaque single score.
# ---------------------------------------------------------------------------

# Schema types that matter for AI answers.
READINESS_SCHEMA_TYPES = ["FAQPage", "HowTo", "Product", "Organization", "Article"]

# Explicit weights (sum = 100). Shown in the UI next to the score.
READINESS_WEIGHTS: dict[str, int] = {
    "Direct answer near the beginning": 12,
    "Question-based headings": 10,
    "H1/H2/H3 hierarchy": 10,
    "Answer-friendly schema": 12,
    "Author information": 6,
    "Published & modified dates": 8,
    "Source links / factual evidence": 10,
    "Brand & product entity clarity": 8,
    "Canonical URL": 6,
    "Robots & sitemap access": 6,
    "Content freshness": 6,
    "Topic coverage (depth)": 6,
}

# Status -> fraction of the weight earned. Transparent and simple.
_STATUS_CREDIT = {"pass": 1.0, "partial": 0.5, "fail": 0.0, "unknown": 0.0}


def _get(row, key, default=None):
    """Read a field from a dict or pandas Series uniformly."""
    try:
        val = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


def _year_of(date_str) -> Optional[int]:
    """Best-effort extraction of a 4-digit year from a date string."""
    if not date_str:
        return None
    m = re.search(r"(19|20)\d{2}", str(date_str))
    return int(m.group(0)) if m else None


def readiness_factors(row, reference_year: Optional[int] = None) -> list[dict]:
    """Return the 12 AI-answer-readiness factors for one audited page.

    Each item: ``{"factor", "status" (pass|partial|fail|unknown), "observed"}``. This
    is the primary output — every factor stands on its own. ``reference_year`` lets
    freshness be evaluated deterministically in tests (defaults to the current year).
    """
    if reference_year is None:
        from datetime import datetime

        reference_year = datetime.now().year

    status_ok = _get(row, "audit_status") == "ok"
    schema_types = str(_get(row, "schema_types", "") or "")
    schema_hits = [t for t in READINESS_SCHEMA_TYPES if t.lower() in schema_types.lower()]

    h1 = _get(row, "h1_count")
    h2 = _get(row, "h2_count")
    h3 = _get(row, "h3_count")
    word_count = _get(row, "word_count")
    ext_links = _get(row, "external_link_count")
    q_headings = _get(row, "question_heading_count")
    mod_year = _year_of(_get(row, "modified_date")) or _year_of(_get(row, "published_date"))

    def unknown_if_not_ok(status: str) -> str:
        return status if status_ok else "unknown"

    factors = []

    # 1. Direct answer near the beginning
    ans = _get(row, "answer_upfront")
    factors.append({"factor": "Direct answer near the beginning",
                    "status": unknown_if_not_ok("pass" if ans else "fail"),
                    "observed": "Substantive paragraph before first H2" if ans else "No early answer paragraph found"})

    # 2. Question-based headings
    factors.append({"factor": "Question-based headings",
                    "status": unknown_if_not_ok("pass" if (q_headings or 0) >= 1 else "fail"),
                    "observed": f"{q_headings or 0} question-style heading(s)"})

    # 3. H1/H2/H3 hierarchy
    if h1 is None:
        h_status = "unknown"
    elif h1 == 1 and (h2 or 0) >= 1 and (h3 or 0) >= 1:
        h_status = "pass"
    elif (h1 or 0) >= 1 and (h2 or 0) >= 1:
        h_status = "partial"
    else:
        h_status = "fail"
    factors.append({"factor": "H1/H2/H3 hierarchy", "status": unknown_if_not_ok(h_status),
                    "observed": f"H1={h1}, H2={h2}, H3={h3}"})

    # 4. Answer-friendly schema
    factors.append({"factor": "Answer-friendly schema",
                    "status": unknown_if_not_ok("pass" if schema_hits else "fail"),
                    "observed": ", ".join(schema_hits) if schema_hits else "None of FAQ/HowTo/Product/Organization/Article"})

    # 5. Author information
    author = _get(row, "has_author")
    factors.append({"factor": "Author information",
                    "status": unknown_if_not_ok("pass" if author else "fail"),
                    "observed": "Author present" if author else "No author metadata"})

    # 6. Published & modified dates
    has_pub = bool(_get(row, "published_date"))
    has_mod = bool(_get(row, "modified_date"))
    date_status = "pass" if (has_pub and has_mod) else ("partial" if (has_pub or has_mod) else "fail")
    factors.append({"factor": "Published & modified dates", "status": unknown_if_not_ok(date_status),
                    "observed": f"published={_get(row,'published_date','—')}, modified={_get(row,'modified_date','—')}"})

    # 7. Source links / factual evidence
    if ext_links is None:
        link_status = "unknown"
    elif ext_links >= 3:
        link_status = "pass"
    elif ext_links >= 1:
        link_status = "partial"
    else:
        link_status = "fail"
    factors.append({"factor": "Source links / factual evidence", "status": unknown_if_not_ok(link_status),
                    "observed": f"{ext_links if ext_links is not None else '—'} external link(s)"})

    # 8. Brand & product entity clarity
    entity_clear = ("Organization" in schema_hits) or ("Product" in schema_hits) or bool(_get(row, "page_title"))
    factors.append({"factor": "Brand & product entity clarity",
                    "status": unknown_if_not_ok("pass" if entity_clear else "fail"),
                    "observed": "Org/Product schema or clear title" if entity_clear else "No clear entity signal"})

    # 9. Canonical URL
    canonical = _get(row, "canonical_url")
    factors.append({"factor": "Canonical URL",
                    "status": unknown_if_not_ok("pass" if canonical else "fail"),
                    "observed": canonical or "No canonical link"})

    # 10. Robots & sitemap access (independent of the page fetch, but unknown if we
    # observed neither).
    robots = _get(row, "robots_accessible")
    sitemap = _get(row, "sitemap_found")
    if robots is None and sitemap is None:
        rs_status = "unknown"
    elif robots and sitemap:
        rs_status = "pass"
    elif robots or sitemap:
        rs_status = "partial"
    else:
        rs_status = "fail"
    factors.append({"factor": "Robots & sitemap access", "status": rs_status,
                    "observed": f"robots={robots}, sitemap={sitemap}"})

    # 11. Content freshness
    if mod_year is None:
        fresh_status = "unknown"
        fresh_obs = "No parseable date"
    elif mod_year >= reference_year - 1:
        fresh_status, fresh_obs = "pass", f"Updated {mod_year}"
    elif mod_year >= reference_year - 3:
        fresh_status, fresh_obs = "partial", f"Updated {mod_year}"
    else:
        fresh_status, fresh_obs = "fail", f"Last updated {mod_year}"
    factors.append({"factor": "Content freshness", "status": unknown_if_not_ok(fresh_status) if mod_year is None else fresh_status,
                    "observed": fresh_obs})

    # 12. Topic coverage (depth)
    if word_count is None:
        cov_status, cov_obs = "unknown", "Unknown length"
    elif word_count >= 1200:
        cov_status, cov_obs = "pass", f"~{word_count} words (in-depth)"
    elif word_count >= 300:
        cov_status, cov_obs = "partial", f"~{word_count} words (moderate)"
    else:
        cov_status, cov_obs = "fail", f"~{word_count} words (thin)"
    factors.append({"factor": "Topic coverage (depth)", "status": unknown_if_not_ok(cov_status) if word_count is None else cov_status,
                    "observed": cov_obs})

    return factors


def readiness_score(row, reference_year: Optional[int] = None) -> dict:
    """OPTIONAL transparent readiness score with the exact formula, weights, and parts.

    score = sum(weight * credit(status)) / sum(weight considered) * 100, where
    credit(pass)=1.0, credit(partial)=0.5, credit(fail)=0.0, and 'unknown' factors are
    EXCLUDED from both numerator and denominator (so blocked pages aren't penalised for
    data we couldn't observe). Returns the score, the formula, and per-factor components.
    """
    factors = readiness_factors(row, reference_year=reference_year)
    components = []
    earned = 0.0
    considered = 0.0
    for f in factors:
        weight = READINESS_WEIGHTS.get(f["factor"], 0)
        status = f["status"]
        if status == "unknown":
            components.append({**f, "weight": weight, "credit": None, "points": None})
            continue
        credit = _STATUS_CREDIT[status]
        pts = weight * credit
        earned += pts
        considered += weight
        components.append({**f, "weight": weight, "credit": credit, "points": pts})
    score = (earned / considered * 100) if considered else None
    return {
        "score": score,
        "points_earned": earned,
        "points_considered": considered,
        "formula": "score = sum(weight × credit) / sum(weight of known factors) × 100; "
                   "credit: pass=1.0, partial=0.5, fail=0.0; unknown factors excluded.",
        "components": components,
    }


# ---------------------------------------------------------------------------
# Answer Extractability Analysis (AEO upgrade).
#
# Extends the readiness audit with the shapes an answer engine can most easily lift
# out of a page (short answers, lists, comparison tables) plus coverage of the
# questions in a chosen AEO cluster. Every factor is reported separately, and the
# summary states its exact component RULE — there is no unexplained score.
# ---------------------------------------------------------------------------

# Each rule is (factor, rule_text, weight). The rule text is displayed in the UI so a
# reader can see precisely what was checked and what threshold was applied.
EXTRACTABILITY_RULES: list[tuple[str, str, int]] = [
    ("Direct answer near the beginning", "pass if a paragraph of 20+ words appears before the first H2", 12),
    ("Question-based headings", "pass if ≥3 headings are questions; partial if 1-2", 10),
    ("Short answer sections", "pass if ≥2 question headings are followed by a ≤60-word answer; partial if 1", 12),
    ("Lists", "pass if ≥3 <ul>/<ol> lists; partial if 1-2", 8),
    ("Comparison tables", "pass if ≥1 table with a header row and 3+ columns; partial if any table exists", 8),
    ("Clear brand, product & category entities", "pass if Organization or Product schema is present; partial if only a page title", 8),
    ("Supporting evidence & outbound sources", "pass if ≥3 external links; partial if 1-2", 10),
    ("Answer schema (FAQ/Product/Organization/Article/HowTo)", "pass if ≥2 of these types; partial if exactly 1", 12),
    ("Cluster question coverage", "pass if ≥60% of the selected cluster's questions are matched in headings/title; partial if ≥30%", 10),
    ("Published & modified dates", "pass if both present; partial if one", 4),
    ("Canonical URL", "pass if a canonical link is present", 3),
    ("Crawlability (robots & sitemap)", "pass if robots.txt is reachable and a sitemap is found; partial if one", 3),
]

EXTRACTABILITY_WEIGHTS: dict[str, int] = {name: weight for name, _rule, weight in EXTRACTABILITY_RULES}
_EXTRACTABILITY_RULE_TEXT: dict[str, str] = {name: rule for name, rule, _w in EXTRACTABILITY_RULES}


def _tier(value: Optional[float], pass_at: float, partial_at: float) -> str:
    """Map a numeric observation onto pass/partial/fail (unknown when value is None)."""
    if value is None:
        return "unknown"
    if value >= pass_at:
        return "pass"
    if value >= partial_at:
        return "partial"
    return "fail"


def extractability_factors(row, cluster_questions: Optional[list[str]] = None) -> list[dict]:
    """The 12 Answer Extractability factors for one audited page, each reported separately.

    ``cluster_questions`` are the prompt texts of the selected AEO cluster; when omitted,
    the coverage factor is reported as ``unknown`` rather than guessed.

    Each item: ``{"factor", "status", "observed", "rule", "weight"}``.
    """
    status_ok = _get(row, "audit_status") == "ok"
    schema_types = str(_get(row, "schema_types", "") or "")
    schema_hits = [t for t in READINESS_SCHEMA_TYPES if t.lower() in schema_types.lower()]

    def gate(status: str) -> str:
        """Anything read from the page body is unknown when the fetch didn't succeed."""
        return status if status_ok else "unknown"

    factors: list[dict] = []

    def add(factor: str, status: str, observed: str) -> None:
        factors.append({
            "factor": factor,
            "status": status,
            "observed": observed,
            "rule": _EXTRACTABILITY_RULE_TEXT[factor],
            "weight": EXTRACTABILITY_WEIGHTS[factor],
        })

    # 1. Direct answer near the beginning
    ans = _get(row, "answer_upfront")
    add("Direct answer near the beginning", gate("pass" if ans else "fail"),
        "Substantive paragraph before the first H2" if ans else "No early answer paragraph found")

    # 2. Question-based headings
    q = _get(row, "question_heading_count")
    add("Question-based headings", gate(_tier(q, 3, 1)), f"{q if q is not None else '—'} question-style heading(s)")

    # 3. Short answer sections
    sa = _get(row, "short_answer_count")
    add("Short answer sections", gate(_tier(sa, 2, 1)),
        f"{sa if sa is not None else '—'} question heading(s) followed by a concise answer")

    # 4. Lists
    lc = _get(row, "list_count")
    add("Lists", gate(_tier(lc, 3, 1)), f"{lc if lc is not None else '—'} list element(s)")

    # 5. Comparison tables
    ct, tc = _get(row, "comparison_table_count"), _get(row, "table_count")
    if ct is None and tc is None:
        ct_status, ct_obs = "unknown", "—"
    elif (ct or 0) >= 1:
        ct_status, ct_obs = "pass", f"{ct} comparison-style table(s) of {tc} total"
    elif (tc or 0) >= 1:
        ct_status, ct_obs = "partial", f"{tc} table(s), none with 3+ columns"
    else:
        ct_status, ct_obs = "fail", "No tables"
    add("Comparison tables", gate(ct_status), ct_obs)

    # 6. Clear brand, product & category entities
    has_entity_schema = ("Organization" in schema_hits) or ("Product" in schema_hits)
    has_title = bool(_get(row, "page_title"))
    ent_status = "pass" if has_entity_schema else ("partial" if has_title else "fail")
    add("Clear brand, product & category entities", gate(ent_status),
        ", ".join(schema_hits) if has_entity_schema else ("Title only, no Organization/Product schema" if has_title else "No entity signal"))

    # 7. Supporting evidence & outbound sources
    ext = _get(row, "external_link_count")
    add("Supporting evidence & outbound sources", gate(_tier(ext, 3, 1)),
        f"{ext if ext is not None else '—'} external link(s)")

    # 8. Answer schema
    add("Answer schema (FAQ/Product/Organization/Article/HowTo)", gate(_tier(len(schema_hits) if status_ok else None, 2, 1)),
        ", ".join(schema_hits) if schema_hits else "None of FAQPage/HowTo/Product/Organization/Article")

    # 9. Cluster question coverage
    if not cluster_questions:
        add("Cluster question coverage", "unknown", "No AEO cluster selected — coverage not evaluated")
    else:
        cov = question_coverage(_get(row, "headings_text"), _get(row, "page_title"), cluster_questions)
        pct = cov["coverage"] or 0.0
        add("Cluster question coverage", gate(_tier(pct, 0.6, 0.3)),
            f"{cov['covered']}/{cov['total']} cluster question(s) matched in headings/title ({round(pct*100)}%)")

    # 10. Published & modified dates
    has_pub, has_mod = bool(_get(row, "published_date")), bool(_get(row, "modified_date"))
    add("Published & modified dates", gate("pass" if (has_pub and has_mod) else ("partial" if (has_pub or has_mod) else "fail")),
        f"published={_get(row,'published_date','—')}, modified={_get(row,'modified_date','—')}")

    # 11. Canonical URL
    canonical = _get(row, "canonical_url")
    add("Canonical URL", gate("pass" if canonical else "fail"), canonical or "No canonical link")

    # 12. Crawlability
    robots, sitemap = _get(row, "robots_accessible"), _get(row, "sitemap_found")
    if robots is None and sitemap is None:
        crawl = "unknown"
    elif robots and sitemap:
        crawl = "pass"
    elif robots or sitemap:
        crawl = "partial"
    else:
        crawl = "fail"
    add("Crawlability (robots & sitemap)", crawl, f"robots={robots}, sitemap={sitemap}")

    return factors


def extractability_summary(row, cluster_questions: Optional[list[str]] = None) -> dict:
    """Transparent Answer Extractability summary — component rules always shown.

    Uses the same explicit arithmetic as the readiness score: each factor's rule and
    weight are displayed, credit is pass=1.0 / partial=0.5 / fail=0.0, and factors that
    could not be observed are excluded from both sides of the ratio (so a blocked page
    is reported as *unknown*, never as a misleading 0).

    Returns ``score`` (None if nothing observable), ``points_earned``,
    ``points_considered``, ``formula``, ``rules`` and per-factor ``components``.
    """
    factors = extractability_factors(row, cluster_questions)
    components, earned, considered = [], 0.0, 0.0
    for f in factors:
        weight = f["weight"]
        if f["status"] == "unknown":
            components.append({**f, "credit": None, "points": None})
            continue
        credit = _STATUS_CREDIT[f["status"]]
        pts = weight * credit
        earned += pts
        considered += weight
        components.append({**f, "credit": credit, "points": pts})
    return {
        "score": (earned / considered * 100) if considered else None,
        "points_earned": earned,
        "points_considered": considered,
        "formula": "Answer Extractability = sum(weight × credit) / sum(weight of evaluated factors) × 100; "
                   "credit: pass=1.0, partial=0.5, fail=0.0; factors that could not be observed are excluded.",
        "rules": EXTRACTABILITY_RULES,
        "components": components,
    }
