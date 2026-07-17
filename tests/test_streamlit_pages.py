"""Reproducible Streamlit page tests via AppTest.

Loads the synthetic demo data, executes the home page and every page, and asserts that
none raises and that a key heading is present. Runs fully offline: the live page-audit
network call is mocked, and no page performs network I/O on load anyway.

Run with the normal ``pytest`` command.
"""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import appkit

# (file, a substring expected in the page's title) — also the intended sidebar order.
PAGES: list[tuple[str, str]] = [
    ("app.py", "AI Visibility Explorer"),
    ("pages/01_Data_Input.py", "Data Input"),
    ("pages/02_Visibility_Dashboard.py", "Visibility Dashboard"),
    ("pages/03_Citation_Analysis.py", "Citation Analysis"),
    ("pages/04_Page_Audit.py", "Page Audit"),
    ("pages/05_Customer_Readout.py", "Readout"),
    ("pages/06_Limitations.py", "Limitations"),
    ("pages/07_Entity_Narrative.py", "Entity & Narrative"),
    ("pages/08_Content_Briefs.py", "Content Action Briefs"),
    ("pages/09_AEO_Question_Clusters.py", "AEO Question Clusters"),
    ("pages/10_AEO_Experiments.py", "AEO Experiments"),
]


@pytest.fixture(scope="module")
def demo_data():
    """Load the synthetic demo once and share it (pages don't mutate it on load)."""
    return appkit.load_demo_analysis()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Guarantee offline: stub the page-audit network entry points."""
    empty = pd.DataFrame(columns=["citation_url", "audit_status"])
    monkeypatch.setattr("src.page_audit.audit_urls", lambda urls: empty, raising=True)
    monkeypatch.setattr("src.page_audit.audit_url", lambda url, session=None: None, raising=True)


def _seed(at: AppTest, data) -> AppTest:
    at.session_state["data"] = data
    at.session_state["focal_brand"] = "Trello"
    at.session_state["alias_overrides"] = {"Monday.com": ["Monday"]}
    at.session_state["experiments"] = []
    return at


@pytest.mark.parametrize("path, expected_title", PAGES)
def test_page_runs_without_exception(path, expected_title, demo_data):
    at = _seed(AppTest.from_file(path, default_timeout=90), demo_data)
    at.run()
    assert not at.exception, f"{path} raised: {[getattr(e, 'message', e) for e in at.exception]}"
    titles = " ".join(t.value for t in at.title)
    headers = " ".join(h.value for h in at.header) + " ".join(s.value for s in at.subheader)
    assert expected_title in (titles + headers), f"{path} is missing its heading '{expected_title}'"


def test_pages_appear_in_correct_order():
    """Zero-padded numeric prefixes must yield the intended sidebar order."""
    import os

    page_files = sorted(f for f in os.listdir("pages") if f.endswith(".py"))
    expected = [
        "01_Data_Input.py", "02_Visibility_Dashboard.py", "03_Citation_Analysis.py",
        "04_Page_Audit.py", "05_Customer_Readout.py", "06_Limitations.py",
        "07_Entity_Narrative.py", "08_Content_Briefs.py", "09_AEO_Question_Clusters.py",
        "10_AEO_Experiments.py",
    ]
    assert page_files == expected


def test_dashboard_shows_sql_engine_note(demo_data):
    """The dashboard advertises that headline metrics run in DuckDB SQL."""
    at = _seed(AppTest.from_file("pages/02_Visibility_Dashboard.py", default_timeout=90), demo_data)
    at.run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "DuckDB SQL" in captions


def test_data_input_has_save_load(demo_data):
    """Project persistence (export/import) is surfaced on Data Input."""
    at = _seed(AppTest.from_file("pages/01_Data_Input.py", default_timeout=90), demo_data)
    at.run()
    assert not at.exception
    labels = " ".join(b.label for b in at.button) + " ".join(d.label for d in getattr(at, "download_button", []))
    # The download button label should mention downloading the project.
    assert any("project" in b.label.lower() for b in at.button) or "project" in labels.lower()
