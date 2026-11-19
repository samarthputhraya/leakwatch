"""Demo rehearsal helpers: a fresh demo lake (after a reset) has no alerts yet."""

from __future__ import annotations

import pandas as pd

from leakwatch.demo import check


def test_current_alerts_is_empty_before_the_first_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "demo_lake", lambda: tmp_path)
    df = check.current_alerts(pd.Timestamp("2018-01-01", tz="UTC"))
    assert df.empty
    assert list(df.columns) == check.ALERT_COLUMNS
