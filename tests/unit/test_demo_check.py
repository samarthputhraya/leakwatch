"""Demo rehearsal helpers: a fresh demo lake (after a reset) has no alerts yet."""

from __future__ import annotations

import pandas as pd

from leakwatch.demo import check


def test_current_alerts_is_empty_before_the_first_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "demo_lake", lambda: tmp_path)
    df = check.current_alerts(pd.Timestamp("2018-01-01", tz="UTC"))
    assert df.empty
    assert list(df.columns) == check.ALERT_COLUMNS


def test_wait_for_retries_through_errors_and_times_out():
    answers = iter([RuntimeError("lake not ready"), False, True])

    def predicate():
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert check.wait_for(predicate, timeout_s=5, poll_s=0.01) is not None
    assert check.wait_for(lambda: False, timeout_s=0.05, poll_s=0.01) is None
