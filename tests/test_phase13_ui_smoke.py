"""
RoadRisk AI — UI Smoke Tests (Phase 13)
=======================================
Guards the Streamlit presentation layer:

* every page and the entry point must at least compile and import cleanly;
* every `from src... import <name>` in the app must resolve to a real attribute;
* every page must render end-to-end in Streamlit's script runner without raising,
  so navigation/config/import/API regressions are caught automatically.

These tests intentionally avoid heavy computation (DBSCAN, model inference) so
they stay fast; the numerical behaviour is covered by the phase-specific tests.
"""

import ast
import importlib
import py_compile
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_ENTRYPOINT = PROJECT_ROOT / "app.py"
PAGE_FILES = sorted((PROJECT_ROOT / "pages").glob("*.py"))
SRC_FILES = sorted((PROJECT_ROOT / "src").glob("*.py"))


def test_there_are_ten_pages():
    # The premium UI upgrade must not change the documented 10-page structure.
    assert len(PAGE_FILES) == 10, [p.name for p in PAGE_FILES]


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.stem)
def test_page_compiles(page_file: Path):
    py_compile.compile(str(page_file), doraise=True)


def test_entrypoint_compiles():
    py_compile.compile(str(APP_ENTRYPOINT), doraise=True)


def _iter_src_imports():
    """Yield (file, module, name) for every `from src.X import name` in the app."""
    for path in [APP_ENTRYPOINT, *PAGE_FILES, *SRC_FILES]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("src."):
                continue
            for alias in node.names:
                yield path, node.module, alias.name


def test_declared_src_imports_resolve():
    """Regression guard: every shared helper a page imports must actually exist.

    Compile-only checks missed real breakages such as `metric_card`,
    `export_dataframe_ui` and `get_db_connection` being imported but undefined.
    """
    failures = []
    for path, module_name, name in _iter_src_imports():
        module = importlib.import_module(module_name)
        if not hasattr(module, name):
            failures.append(f"{path.name}: {module_name} has no attribute '{name}'")
    assert not failures, failures


def test_confusion_matrix_builder_returns_plotly_figure():
    """Regression guard: Plotly 7.x removed `figure_factory.create_annotated_heatmap`."""
    import plotly.graph_objects as go

    from src.evaluate_model import plot_confusion_matrix_plotly

    fig = plot_confusion_matrix_plotly([[37, 10], [9, 44]], ["Accident", "Non Accident"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.stem)
def test_page_renders_without_exception(page_file: Path):
    """Execute every page in Streamlit's script runner and assert it does not raise."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(page_file), default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


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
