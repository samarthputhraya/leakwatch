"""The demo runner says "running" only once the scored stream is warm (D-043)."""

from __future__ import annotations

from types import SimpleNamespace

from leakwatch.demo.runner import DemoRunner

CFG = {"demo": {"topic": "demo.readings", "warm_batches": 2}}


def runner_with(progress):
    r = DemoRunner(CFG, 720.0)
    r.queries = [SimpleNamespace(recentProgress=progress)]
    return r


def test_warm_after_enough_batches_with_data():
    assert not runner_with([]).stream_warm()
    assert not runner_with([{"numInputRows": 10}, {"numInputRows": 0}]).stream_warm()
    assert runner_with(
        [{"numInputRows": 10}, {"numInputRows": 0}, {"numInputRows": 3}]
    ).stream_warm()


def test_no_query_counts_as_warm_and_a_failing_query_does_not():
    assert DemoRunner(CFG, 720.0).stream_warm()

    class Broken:
        @property
        def recentProgress(self):  # noqa: N802 (PySpark's name)
            raise RuntimeError("query stopped")

    r = DemoRunner(CFG, 720.0)
    r.queries = [Broken()]
    assert not r.stream_warm()
