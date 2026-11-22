"""The demo runner says "running" only once the scored stream is warm (D-043)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


def test_evidence_path_is_loaded_once(monkeypatch):
    """D-058: network index and manual search before the streams start, not again on reset."""
    from leakwatch.copilot import tools

    seen = []
    monkeypatch.setattr(tools.ToolContext, "get_network", lambda self: seen.append("network"))
    monkeypatch.setattr(tools, "search_manual", lambda ctx, args: seen.append(args["query"]) or {})
    r = DemoRunner(CFG, 720.0)
    r.warm_evidence()
    r.warm_evidence()
    assert seen[0] == "network" and len(seen) == 2 and r.evidence_warm


def test_a_failed_evidence_warm_up_does_not_stop_the_runner(monkeypatch):
    from leakwatch.copilot import tools

    def broken(self):
        raise OSError("network file missing")

    monkeypatch.setattr(tools.ToolContext, "get_network", broken)
    r = DemoRunner(CFG, 720.0)
    r.warm_evidence()
    assert r.evidence_warm


class FakeQuery:
    """A streaming query as ``stream_drained`` sees it."""

    def __init__(self, rows=0, at="2026-11-22T05:00:10Z", active=True, busy=False):
        self.isActive = active  # PySpark's names
        self.status = {"isTriggerActive": busy}
        self.lastProgress = None if at is None else {"numInputRows": rows, "timestamp": at}


SENT = datetime(2026, 11, 22, 5, 0, 0, tzinfo=UTC)  # the last reading went out


def drained(*queries):
    r = DemoRunner(CFG, 720.0)
    r.queries = list(queries)
    return r.stream_drained(SENT)


def test_streams_are_drained_only_after_an_empty_trigger_that_began_after_the_last_reading():
    """D-060: the replay is called finished only once the stream has analysed what it was sent."""
    assert drained(FakeQuery(), FakeQuery())
    assert not drained(FakeQuery(rows=2400))  # still reading
    assert not drained(FakeQuery(at="2026-11-22T04:59:58Z"))  # empty, but from before the end
    assert not drained(FakeQuery(busy=True))  # a batch is running
    assert not drained(FakeQuery(at=None))  # no batch reported yet
    assert not drained(FakeQuery(), FakeQuery(rows=10))  # every query counts
    assert drained(FakeQuery(rows=10, active=False))  # a stopped query has nothing left


class Inbox:
    """A command consumer that hands out the queued messages, then nothing."""

    def __init__(self, *commands):
        self.queue = [
            SimpleNamespace(error=lambda: None, value=lambda c=c: c.encode()) for c in commands
        ]
        self.polls = 0

    def poll(self, timeout):
        self.polls += 1
        return self.queue.pop(0) if self.queue else None


def finishing_runner(limit, queries):
    r = DemoRunner({"demo": {**CFG["demo"], "drain_timeout_s": limit}}, 720.0)
    r.queries = queries
    r.write_state = lambda: None
    return r


def test_finish_waits_for_the_streams_and_a_reset_or_the_limit_ends_the_wait(monkeypatch):
    from leakwatch.demo import runner as runner_mod

    later = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
    done = finishing_runner(60, [FakeQuery(at=later)])
    inbox = Inbox()
    done.finish_streams(inbox)
    assert done.status == "finishing" and inbox.polls == 0  # caught up at once

    behind = finishing_runner(60, [FakeQuery(rows=500)])
    behind.finish_streams(Inbox('{"command": "reset", "params": {}}'))
    assert behind._reset_flag  # a restart ends the wait

    clock = iter(range(0, 1000, 30))
    monkeypatch.setattr(runner_mod.time, "perf_counter", lambda: float(next(clock)))
    stuck = finishing_runner(60, [FakeQuery(rows=500)])
    inbox = Inbox()
    stuck.finish_streams(inbox)
    assert inbox.polls == 1 and not stuck._reset_flag  # gave up after the limit


def test_a_new_speed_is_written_at_once(tmp_path, monkeypatch):
    """After the replay has ended the runner writes no state of its own; the header must still
    show a speed set from the screen."""
    from leakwatch.demo import runner as runner_mod
    from leakwatch.ingest.replayer import Pacer

    monkeypatch.setattr(runner_mod, "demo_root", lambda: tmp_path)
    r = DemoRunner({"demo": {**CFG["demo"], "background_start": "2018-11-19"}}, 2880.0)
    r.pacer = Pacer(2880.0)
    r.status = "finished"
    r._commands(Inbox('{"command": "set_speed", "params": {"speed": 1440}}'))
    assert runner_mod.read_state()["speed"] == 1440.0
    assert runner_mod.read_state()["status"] == "finished"


def test_dispatch_without_a_model_writes_only_the_template(tmp_path, monkeypatch):
    from leakwatch.common import lake
    from leakwatch.copilot import testing
    from leakwatch.demo import runner as runner_mod

    root = testing.build_tiny_lake(tmp_path / "lake")
    monkeypatch.setattr(runner_mod, "demo_lake", lambda: root)
    DemoRunner(CFG, 720.0).dispatch([testing.ALERT_A1])
    n = lake.connect(lake_root=root).execute("SELECT count(*) FROM work_orders").fetchone()
    assert n == (1,)
