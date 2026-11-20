"""Shared persistence tuning (eval.alerting): the false-alarm rule and the official-score rule."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from leakwatch.common.config import load_config
from leakwatch.eval.alerting import tune_persistence

SERIES = ["pressure__n1"]
AREAS = {"pressure__n1": "A"}


def setup():
    cfg = copy.deepcopy(load_config(environ={}))
    cfg["bdt"]["persistence_candidates"] = [1, 2, 4, 32]
    start = pd.Timestamp("2018-01-01", tz="UTC")
    ws = pd.date_range(start, periods=20 * 96, freq="15min")
    score = np.zeros(len(ws))
    score[[100, 400]] = 5.0  # two one-window bursts on normal validation days
    leak = pd.Timestamp("2018-01-12", tz="UTC")
    onset = int((leak - start) / pd.Timedelta(minutes=15))
    score[onset : onset + 20] = 5.0  # 5 h above tau after the leak starts
    scored = pd.DataFrame(
        {
            "source": "battledim",
            "scenario_id": None,
            "window_start": ws,
            "window_end": ws + pd.Timedelta(hours=1),
            "score": score,
        }
    )
    shares = np.ones((len(ws), 1))
    val = np.asarray(ws < leak)
    gt = pd.DataFrame(
        {
            "leak_id": ["L1"],
            "pipe_id": ["p1"],
            "type": ["abrupt"],
            "start_ts": [leak],
            "end_ts": pd.Series([pd.NaT], dtype="datetime64[ns, UTC]"),
            "peak_outflow_m3h": [10.0],
        }
    )
    return cfg, scored, shares, val, gt, leak


def run(official=None):
    cfg, scored, shares, val, gt, _ = setup()
    return tune_persistence(
        1.0, scored, shares, SERIES, val, None, [], cfg, gt, AREAS, "t", 3600, 900, official
    )


def test_false_alarm_rule_takes_the_smallest_m_within_the_limit():
    m, rows = run()
    assert [r["m"] for r in rows] == [1, 2, 4, 32]
    assert rows[0]["validation_false_alarms"] == 2 and rows[1]["validation_false_alarms"] == 0
    assert m == 1  # 2 false alarms in 11 days is within 2 per week
    assert "official_2018" not in rows[0]


def test_official_rule_takes_the_best_score_and_the_smaller_m_on_ties():
    _, _, _, _, _, leak = setup()
    seen = []

    def official(alerts: pd.DataFrame) -> dict:
        seen.append(len(alerts))
        hit = bool((alerts["start_ts"] >= leak).any())
        return {
            "total_score_eur": 100.0 * hit - 10.0 * len(alerts),
            "true_positives": int(hit),
            "false_positives": int((alerts["start_ts"] < leak).sum()),
        }

    m, rows = run(official)
    assert seen == [3, 1, 1, 0]
    assert [r["official_2018"]["total_score_eur"] for r in rows] == [70.0, 90.0, 90.0, 0.0]
    assert m == 2
