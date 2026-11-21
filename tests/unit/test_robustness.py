"""Robustness: sensor dropout, a stuck sensor, the model going down, SQL injection (D-049).

Each test feeds a fault the field produces and checks the system degrades the way it says it
does: a missing reading never becomes an alarm or a crash, the localizer still answers, a dead
LLM leaves a template work order, and no query the model writes can escape the lake views.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.bdt.alerts import AlertRule, AlertTracker
from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
from leakwatch.copilot import dispatch, testing
from leakwatch.copilot.llm import LLMError
from leakwatch.copilot.sql_guard import SQLGuardError, guard_sql
from leakwatch.dl.inversion import Inversion
from leakwatch.features.rebaseline import Rebaseline, rebaselined

COLS = ["pressure__n1__residual", "pressure__n2__residual", "flow__p1__residual"]


def scorer() -> KMeansScorer:
    k = 2
    params = KMeansParams(
        feature_columns=tuple(COLS),
        mean=np.zeros(3),
        std=np.ones(3),
        pc=np.eye(3),
        centers=np.zeros((k, 3)),
        threshold=3.0,
        persistence_windows=2,
        meta={},
    )
    return KMeansScorer(params)


def test_a_dropped_sensor_scores_as_normal_not_as_an_alarm():
    """NaN (sensor offline) is standardized to the mean: it adds nothing to the score."""
    s = scorer()
    x = np.array([[np.nan, 0.1, -0.2], [np.nan, np.nan, np.nan]])
    res = s.score_matrix(x)
    assert np.isfinite(res["score"]).all()
    assert res["score"][1] == pytest.approx(0.0)
    assert res["score"][0] == pytest.approx(np.hypot(0.1, 0.2))


def test_a_dropout_mid_leak_does_not_close_or_reopen_the_alert():
    rule = AlertRule(
        threshold=1.0,
        persistence_windows=2,
        close_after_windows=4,
        top_n=1,
        series_area={"pressure__n1": "A", "pressure__n2": "A", "flow__p1": "A"},
        origin="stream",
        model="kmeans",
    )
    tracker = AlertTracker(rule, "battledim", None, ["pressure__n1", "pressure__n2", "flow__p1"])
    ends = np.arange(10) * 900 + 1_546_300_800
    scores = np.array([5, 5, 5, 0.0, 5, 5, 5, 5, 5, 5])  # one window lost to a dropout (score 0)
    shares = np.tile([1.0, 0.0, 0.0], (10, 1))
    rows = tracker.update(ends, scores, shares, [None] * 10)
    assert len({r["alert_id"] for r in rows}) == 1  # one alert, not closed and reopened
    assert rows[-1]["status"] == "open"


def test_a_stuck_sensor_rebaselines_to_zero():
    """A sensor frozen at one value has a constant residual; re-baselining maps it to 0."""
    t = pd.date_range("2019-01-01", periods=4 * 24 * 10, freq="15min", tz="UTC")
    df = pd.DataFrame({"window_start": t, "stuck": 3.7, "live": np.sin(np.arange(len(t)))})
    out = rebaselined(df, ["stuck"], Rebaseline(window_days=7, gap_days=1))
    later = out[out["window_start"] >= t[0] + pd.Timedelta(days=8)]
    assert later["stuck"].abs().max() == pytest.approx(0.0)


def test_localizer_still_ranks_the_leak_near_the_top_with_a_sensor_offline():
    """A missing sensor enters the localizer as 0 (``benchmark_inputs`` / live: nan_to_num)."""
    n, sensors = 30, 6
    pos = np.linspace(0, n - 1, sensors)
    sig = -np.exp(-np.abs(np.arange(n)[:, None] - pos[None, :]) / 3.0)
    inv = Inversion(sig, np.eye(sensors), 0.01, np.ones(n, dtype=bool))

    def rank_of(x: np.ndarray) -> int:
        return int((np.argsort(-inv.loglik(x)[0]) == 12).nonzero()[0][0])

    x = 2.0 * sig[[12]]
    assert rank_of(x) == 0
    x[0, 1] = 0.0  # a sensor next to the leak is offline: worse, but still near the top
    assert rank_of(x) < n // 3
    assert np.isfinite(inv.loglik(np.zeros((1, sensors)))[0]).all()  # all offline: no NaN


@pytest.fixture
def ctx(tmp_path):
    return testing.tiny_context(testing.build_tiny_lake(tmp_path / "lake"))


class DownLLM:
    """A model endpoint that is down (connection refused / 503)."""

    model = "down"

    def chat(self, *args, **kwargs):
        raise LLMError("APIConnectionError: connection refused", transient=True)


def test_llm_down_gives_a_template_work_order_in_demo_safe_mode(ctx):
    gen = dispatch.generate_work_order(ctx, testing.ALERT_A1, llm=DownLLM())
    assert gen.generator == "template"
    assert any("template fallback" in n for n in gen.notes)
    assert gen.order.pipes_to_inspect  # the crew still gets the pipes to walk


def test_llm_down_is_an_error_without_demo_safe_mode(ctx):
    ctx.full_cfg["copilot"]["dispatch"]["demo_safe_mode"] = False
    with pytest.raises(LLMError):
        dispatch.generate_work_order(ctx, testing.ALERT_A1, llm=DownLLM())


def test_copilot_answer_explains_a_down_model(ctx):
    from leakwatch.copilot.agent import CopilotAgent

    ans = CopilotAgent(DownLLM(), ctx, "v3").ask("How many alerts were raised in 2019?")
    assert ans.error and ans.error.startswith("llm_unavailable")
    assert "did not respond" in ans.answer and not ans.refused


VIEWS = {"alerts", "features", "clean_readings"}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM alerts; ATTACH 'evil.db'",
        "SELECT * FROM alerts --\n; DROP TABLE alerts",
        "SELECT * FROM alerts /* ; */ ; COPY alerts TO '/tmp/x.csv'",
        "WITH x AS (SELECT 1) SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM alerts UNION ALL SELECT * FROM read_parquet('/root/.ssh/*')",
        "SELECT * FROM alerts WHERE alert_id IN (SELECT content FROM read_blob('/etc/shadow'))",
        "SELECT * FROM 'data/../../etc/hostname'",
        "SELECT * FROM sqlite_scan('x.db', 't')",
        "SELECT * FROM postgres_scan('host=x', 'public', 't')",
        "SELECT * FROM alerts, LATERAL (SELECT * FROM read_csv('/etc/passwd'))",
        "PRAGMA enable_external_access",
        "SELECT * FROM system.main.alerts",
        "CALL pragma_version()",
        "EXPORT DATABASE '/tmp/x'",
        "SELECT 1 FROM alerts WHERE 1 = 1; SET enable_external_access = true",
    ],
)
def test_sql_guard_blocks_injection_patterns(query):
    with pytest.raises(SQLGuardError):
        guard_sql(query, VIEWS)


def test_sql_guard_keeps_a_quoted_semicolon_inside_a_string():
    out = guard_sql("SELECT count(*) FROM alerts WHERE area = 'A; DROP TABLE alerts'", VIEWS)
    assert "DROP" in out.sql and out.sql.upper().startswith("SELECT")  # data, not a statement
