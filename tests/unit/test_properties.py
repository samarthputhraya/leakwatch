"""Property-based tests (hypothesis): invariants that must hold for every input (ST, D-040).

Each test states a property of a core function and lets hypothesis search for a counterexample:
re-baselining ignores constant offsets, the official score ignores the order detections arrive
in and never charges a repeat, the SQL guard never lets a write statement through, and the
number guardrail accepts every value it was given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from leakwatch.eval import battledim_score as bs  # noqa: E402
from leakwatch.features.rebaseline import Rebaseline, rebaselined  # noqa: E402

FAST = settings(max_examples=60, deadline=None)
RB = Rebaseline(window_days=1, gap_days=0.5)


def series_frame(values: list[float]) -> pd.DataFrame:
    ts = pd.date_range("2018-01-01", periods=len(values), freq="15min", tz="UTC")
    return pd.DataFrame({"window_start": ts, "a": values})


values = st.lists(
    st.floats(-50, 50, allow_nan=False, allow_infinity=False), min_size=150, max_size=260
)


@FAST
@given(values, st.floats(-100, 100, allow_nan=False))
def test_rebaseline_ignores_a_constant_offset(v, offset):
    a = rebaselined(series_frame(v), ["a"], RB)["a"].to_numpy()
    b = rebaselined(series_frame([x + offset for x in v]), ["a"], RB)["a"].to_numpy()
    assert np.allclose(a, b, equal_nan=True, atol=1e-6)


@FAST
@given(st.floats(-50, 50, allow_nan=False), st.integers(150, 260))
def test_rebaseline_of_a_constant_series_is_zero(c, n):
    out = rebaselined(series_frame([c] * n), ["a"], RB)["a"].dropna().to_numpy()
    assert len(out) > 0 and np.allclose(out, 0.0, atol=1e-9)


class Line:
    """Five 100 m pipes in a line: a - b - c - d - e - f."""

    node_name_list = ("a", "b", "c", "d", "e", "f")

    def links(self):
        from types import SimpleNamespace

        nodes = self.node_name_list
        for i in range(5):
            yield (
                f"p{i}",
                SimpleNamespace(
                    start_node_name=nodes[i],
                    end_node_name=nodes[i + 1],
                    link_type="Pipe",
                    length=100.0,
                ),
            )


DIST = bs.LinkDistances(Line())
LEAKS = [
    bs.OfficialLeak("p0", 10, 5_000, np.full(105_120, 6.0)),
    bs.OfficialLeak("p4", 500, 3_000, np.full(105_120, 12.0)),
]
detection = st.tuples(st.sampled_from([f"p{i}" for i in range(5)]), st.integers(-100, 6_000))


def frame(dets: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"link_id": [d[0] for d in dets], "ts": [bs.YEAR_START + bs.STEP * d[1] for d in dets]}
    )


@FAST
@given(st.lists(detection, max_size=12), st.randoms(use_true_random=False))
def test_official_score_ignores_submission_order(dets, rnd):
    shuffled = list(dets)
    rnd.shuffle(shuffled)
    a = bs.score_detections(frame(dets), LEAKS, DIST, bs.ScoringParams())
    b = bs.score_detections(frame(shuffled), LEAKS, DIST, bs.ScoringParams())
    assert a["true_positives"] == b["true_positives"]
    assert a["false_positives"] == b["false_positives"]
    # equal times may resolve in a different order; the total only depends on what was found when
    if len({d[1] for d in dets}) == len(dets):
        assert a["total_score_eur"] == pytest.approx(b["total_score_eur"])


@FAST
@given(st.lists(detection, max_size=12))
def test_official_score_bookkeeping(dets):
    r = bs.score_detections(frame(dets), LEAKS, DIST, bs.ScoringParams())
    in_year = sum(1 for d in dets if d[1] >= 0)
    assert r["true_positives"] + r["false_positives"] + r["repeated_detections"] == in_year
    assert r["true_positives"] <= len(LEAKS)
    assert r["true_positives"] + r["false_negatives"] == len(LEAKS)
    # a copy of a report that hit nothing hits nothing again; a copy of a hit is a repeat or
    # claims another running leak within 300 m (as in the official code), never a false positive
    twice = bs.score_detections(frame(dets + dets), LEAKS, DIST, bs.ScoringParams())
    assert twice["false_positives"] == 2 * r["false_positives"]
    assert twice["true_positives"] >= r["true_positives"]


@FAST
@given(st.sampled_from([f"p{i}" for i in range(5)]), st.sampled_from([f"p{i}" for i in range(5)]))
def test_link_distance_is_symmetric(a, b):
    assert DIST.between(a, b) == pytest.approx(DIST.between(b, a))
    assert DIST.between(a, b) >= 0.0


write_words = st.sampled_from(
    [
        "DROP TABLE alerts",
        "DELETE FROM alerts",
        "INSERT INTO alerts VALUES (1)",
        "UPDATE alerts SET area = 'A'",
        "CREATE TABLE x (a INT)",
        "COPY alerts TO 'x.csv'",
        "ATTACH 'x.db'",
        "PRAGMA version",
        "INSTALL httpfs",
    ]
)


@FAST
@given(write_words, st.sampled_from(["", "SELECT 1; ", "select * from alerts;\n"]))
def test_sql_guard_never_passes_a_write(stmt, prefix):
    pytest.importorskip("sqlglot")
    from leakwatch.copilot.sql_guard import SQLGuardError, guard_sql

    with pytest.raises(SQLGuardError):
        guard_sql(prefix + stmt, ["alerts"])


@FAST
@given(st.floats(0.01, 1e6, allow_nan=False), st.integers(0, 3))
def test_number_guardrail_accepts_the_value_it_was_given(value, digits):
    from leakwatch.copilot.guardrails import NumberMention, number_supported

    shown = round(value, digits)
    mention = NumberMention(text=str(shown), value=shown, tolerance=0.5 * 10**-digits)
    assert number_supported(mention, [value])
