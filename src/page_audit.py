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
    audit.h1_count = len(soup.find_all("h1"))
    audit.h2_count = len(soup.find_all("h2"))

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

    # Approximate word count from visible body text.
    body_text = soup.get_text(" ", strip=True)
    audit.word_count = len(body_text.split()) if body_text else 0


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
