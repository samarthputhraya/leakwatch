"""``make dashboard``: start the Streamlit dashboard after warming up pyarrow in the main thread.

Streamlit runs every script rerun in a new, short-lived thread. When pyarrow converts its first
pandas frame inside such a thread, a later conversion in another rerun crashes the whole server
with a segmentation fault (pyarrow 25 / pandas 2.2 / numpy 1.26, found with Python's fault
handler: ``pyarrow/pandas_compat.py convert_column`` under ``st.dataframe``; D-032). One
conversion in the main thread before Streamlit starts prevents it; the regression test runs the
dashboard pattern twice in one process.

Usage: ``python -m leakwatch.dashboard.serve [streamlit run options]``.
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
    """Warm up pyarrow, then hand over to ``streamlit run app.py`` with the given options."""
    warm_up_arrow()
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(APP), *(sys.argv[1:] if argv is None else argv)]
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
