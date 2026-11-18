from __future__ import annotations

import importlib.util

from leakwatch.common.paths import REPO_ROOT

_SCRIPT = REPO_ROOT / "scripts" / "run_notebook.py"
_spec = importlib.util.spec_from_file_location("run_notebook", _SCRIPT)
assert _spec is not None and _spec.loader is not None
run_notebook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_notebook)


def test_percent_format_split_into_cells():
    source = "\n".join(
        [
            "# %% [markdown]",
            "# # Title",
            "#",
            "# Some text",
            "",
            "# %%",
            "x = 1",
            "",
            "# %%",
            "",
            "# %%",
            "x + 1",
        ]
    )
    nb = run_notebook.percent_to_notebook(source)
    kinds = [c.cell_type for c in nb.cells]
    assert kinds == ["markdown", "code", "code"]  # empty cell dropped
    assert nb.cells[0].source == "# Title\n\nSome text"
    assert nb.cells[1].source == "x = 1"
    assert nb.metadata["kernelspec"]["name"] == "python3"


def test_eda_notebook_source_parses():
    source = (REPO_ROOT / "notebooks/01_eda_battledim.py").read_text(encoding="utf-8")
    nb = run_notebook.percent_to_notebook(source)
    assert sum(c.cell_type == "code" for c in nb.cells) >= 10
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<cell>", "exec")  # every code cell is valid Python
