"""Dashboard launcher (D-032, D-052): warm up pyarrow, then the screen or the legacy page."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from leakwatch.dashboard import serve

TABLE_SCRIPT = (
    "import pandas as pd\nimport streamlit as st\n\nst.dataframe(pd.DataFrame({'x': [1.0, 2.0]}))\n"
)


def test_legacy_runs_the_streamlit_page_with_the_given_options(monkeypatch):
    seen = {}
    from streamlit.web import cli

    monkeypatch.setattr(cli, "main", lambda: seen.setdefault("argv", list(sys.argv)) and 0)
    monkeypatch.setattr(sys, "argv", ["serve"])
    serve.main(["--legacy", "--server.port", "8600"])
    assert seen["argv"] == ["streamlit", "run", str(serve.APP), "--server.port", "8600"]
    assert serve.APP.name == "app.py" and serve.APP.exists()


def test_main_serves_the_control_room_screen_by_default(monkeypatch):
    from leakwatch.dashboard import api

    seen = {}
    monkeypatch.setattr(api, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    serve.main(["--port", "8601"])
    assert seen["argv"] == ["--port", "8601"]


@pytest.mark.slow
def test_repeated_table_renders_survive_after_the_warm_up(tmp_path):
    """Without the warm-up the second render segfaulted the process; it must now exit cleanly."""
    app = tmp_path / "table_app.py"
    app.write_text(TABLE_SCRIPT, encoding="utf-8")
    code = textwrap.dedent(
        f"""
        from leakwatch.dashboard.serve import warm_up_arrow
        warm_up_arrow()
        from streamlit.testing.v1 import AppTest
        for _ in range(3):
            at = AppTest.from_file({str(app)!r}, default_timeout=60)
            at.run()
            assert not at.exception, at.exception
            assert len(at.dataframe) == 1
        print("rendered 3 times")
        """
    )
    res = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    assert "rendered 3 times" in res.stdout
