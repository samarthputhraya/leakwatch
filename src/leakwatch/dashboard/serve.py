"""``make dashboard`` / ``make demo``: warm up pyarrow in the main thread, then serve the screen.

By default this is the control-room screen (``api.py``, D-052). ``--legacy`` starts the earlier
Streamlit page instead (``make dashboard-legacy``), which is what the warm-up was written for:

Streamlit runs every script rerun in a new, short-lived thread. When pyarrow converts its first
pandas frame inside such a thread, a later conversion in another rerun crashes the whole server
with a segmentation fault (pyarrow 25 / pandas 2.2 / numpy 1.26, found with Python's fault
handler: ``pyarrow/pandas_compat.py convert_column`` under ``st.dataframe``; D-032). One
conversion in the main thread before Streamlit starts prevents it; the regression test runs the
dashboard pattern twice in one process.

Usage: ``python -m leakwatch.dashboard.serve [--host H] [--port P]`` or
``python -m leakwatch.dashboard.serve --legacy [streamlit run options]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def warm_up_arrow() -> None:
    """Convert a small pandas frame (float, UTC timestamp, string) to Arrow in this thread."""
    import pandas as pd
    import pyarrow as pa

    pa.Table.from_pandas(
        pd.DataFrame(
            {
                "x": [1.0],
                "t": pd.to_datetime(["2018-01-01"], utc=True),
                "s": ["a"],
            }
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Warm up pyarrow, then serve the control-room screen (or ``--legacy``: the Streamlit page)."""
    args = sys.argv[1:] if argv is None else list(argv)
    warm_up_arrow()
    if args[:1] == ["--legacy"]:
        from streamlit.web import cli

        sys.argv = ["streamlit", "run", str(APP), *args[1:]]
        return cli.main()
    from leakwatch.dashboard import api

    return api.main(args)


if __name__ == "__main__":
    sys.exit(main())
