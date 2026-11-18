"""Build a Jupyter notebook from a percent-format ``.py`` file and execute it headless.

Cells are separated by ``# %%`` lines; ``# %% [markdown]`` starts a markdown cell whose lines are
written as ``# text``. The executed notebook is saved to ``notebooks/executed/<name>.ipynb``.

Usage: python scripts/run_notebook.py notebooks/01_eda_battledim.py [--timeout 1200]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]


def percent_to_notebook(source: str) -> nbformat.NotebookNode:
    """Split ``# %%`` cells into a notebook (markdown cells have their ``# `` prefix removed)."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        text = "\n".join(lines).strip("\n")
        if kind is None or not text:
            return
        if kind == "markdown":
            md = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines
            )
            nb.cells.append(nbformat.v4.new_markdown_cell(md.strip("\n")))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(text))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        else:
            lines.append(line)
    flush()
    return nb


def main(argv: list[str] | None = None) -> int:
    """Execute the notebook; non-zero exit if any cell fails."""
    parser = argparse.ArgumentParser(description="Execute a percent-format notebook headless.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--timeout", type=int, default=1200, help="per-cell timeout in seconds")
    args = parser.parse_args(argv)

    source = args.source.resolve()
    nb = percent_to_notebook(source.read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "notebooks" / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{source.stem}.ipynb"
    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(source.parent)}},
    )
    t0 = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as exc:
        nbformat.write(nb, out)
        print(f"FAILED after {time.perf_counter() - t0:.0f}s; partial output: {out}\n{exc}")
        return 1
    nbformat.write(nb, out)
    print(f"executed {len(nb.cells)} cells in {time.perf_counter() - t0:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
