"""
RoadRisk AI — UI Smoke Tests (Phase 13)
=======================================
Guards the Streamlit presentation layer:

* every page and the entry point must at least compile and import cleanly;
* the home page must render end-to-end in Streamlit's script runner without
  raising, so navigation/config regressions are caught automatically.

These tests intentionally avoid heavy computation (DBSCAN, model inference) so
they stay fast; the numerical behaviour is covered by the phase-specific tests.
"""

import py_compile
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_ENTRYPOINT = PROJECT_ROOT / "app.py"
PAGE_FILES = sorted((PROJECT_ROOT / "pages").glob("*.py"))


def test_there_are_ten_pages():
    # The premium UI upgrade must not change the documented 10-page structure.
    assert len(PAGE_FILES) == 10, [p.name for p in PAGE_FILES]


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.stem)
def test_page_compiles(page_file: Path):
    py_compile.compile(str(page_file), doraise=True)


def test_entrypoint_compiles():
    py_compile.compile(str(APP_ENTRYPOINT), doraise=True)


def test_home_page_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_ENTRYPOINT), default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


def test_home_page_does_not_advertise_completed_pages_as_pending():
    """Regression guard: pages 8 and 9 are implemented and must not show as pending."""
    source = APP_ENTRYPOINT.read_text(encoding="utf-8")
    for line in source.splitlines():
        if "_AI_Assistant" in line or "_Model_Explainability" in line:
            assert "pending" not in line, f"completed page still marked pending: {line.strip()}"
