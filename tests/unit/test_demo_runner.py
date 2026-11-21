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


def test_dispatch_writes_the_template_first_then_the_model_order(tmp_path, monkeypatch):
    """D-046: a work order exists as soon as the alert does; the model's replaces it later."""
    import json

    from leakwatch.common import lake
    from leakwatch.copilot import testing
    from leakwatch.copilot.llm import ScriptedLLM, reply
    from leakwatch.demo import runner as runner_mod

    root = testing.build_tiny_lake(tmp_path / "lake")
    monkeypatch.setattr(runner_mod, "demo_lake", lambda: root)
    order = {
        "priority": "P3",
        "summary": "Likely leak in area A.",
        "pipes_to_inspect": ["p2"],
        "evidence": ["get_alert: alert in area A"],
        "recommended_actions": ["Walk the pipes in order."],
        "manual_citations": [],
    }
    r = DemoRunner(CFG, 720.0)
    r.llm = ScriptedLLM([reply(json.dumps(order))])
    r.dispatch([testing.ALERT_A1])
    con = lake.connect(lake_root=root)
    rows = con.execute(
        "SELECT generator FROM work_orders WHERE alert_id = ? ORDER BY generated_at",
        [testing.ALERT_A1],
    ).fetchall()
    assert [g for (g,) in rows] == ["template", "llm"]
    shown = con.execute(
        "SELECT work_order_generator FROM alerts WHERE alert_id = ?", [testing.ALERT_A1]
    ).fetchone()
    assert shown == ("llm",)
    assert r.llm.requests[0]["fast"] is True


def test_dispatch_without_a_model_writes_only_the_template(tmp_path, monkeypatch):
    from leakwatch.common import lake
    from leakwatch.copilot import testing
    from leakwatch.demo import runner as runner_mod

    root = testing.build_tiny_lake(tmp_path / "lake")
    monkeypatch.setattr(runner_mod, "demo_lake", lambda: root)
    DemoRunner(CFG, 720.0).dispatch([testing.ALERT_A1])
    n = lake.connect(lake_root=root).execute("SELECT count(*) FROM work_orders").fetchone()
    assert n == (1,)
