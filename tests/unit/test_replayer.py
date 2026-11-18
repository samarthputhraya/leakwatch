from __future__ import annotations

import numpy as np
import orjson
import pandas as pd
import pytest

from leakwatch.common.schemas import Reading
from leakwatch.ingest import replayer as rp


def wide_frame(n_steps: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2019-03-01", periods=n_steps, freq="5min", tz="UTC", name="ts")
    return pd.DataFrame(
        {
            "pressure__n54": np.linspace(40.0, 41.0, n_steps),
            "flow__p227": np.linspace(80.0, 82.0, n_steps),
            "amr__n1": [np.nan] + [12.5] * (n_steps - 1),
        },
        index=idx,
    )


class FakeProducer:
    def __init__(self, fail_first: int = 0):
        self.messages: list[tuple[str, bytes, bytes]] = []
        self.fail_first = fail_first
        self.polls = 0

    def produce(self, topic, value, key):
        if self.fail_first:
            self.fail_first -= 1
            raise BufferError("queue full")
        self.messages.append((topic, key, value))

    def poll(self, timeout):
        self.polls += 1
        return 0

    def flush(self, timeout):
        return 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [("max", None), (None, None), ("360", 360.0), ("60x", 60.0), (1, 1.0), ("MAX", None)],
)
def test_parse_speed(text, expected):
    assert rp.parse_speed(text) == expected


@pytest.mark.parametrize("bad", ["0", "-5", "fast"])
def test_parse_speed_rejects(bad):
    with pytest.raises(ValueError):
        rp.parse_speed(bad)


def test_encoded_messages_satisfy_reading_contract():
    wide = wide_frame()
    metas = rp.sensor_meta(list(wide.columns))
    msgs = list(
        rp.encode_step(
            "2019-03-01T00:05:00Z", 2019, wide.iloc[1].to_numpy(), metas, "battledim", None
        )
    )
    assert [k for k, _ in msgs] == [b"pressure__n54", b"flow__p227", b"amr__n1"]
    for _, payload in msgs:
        reading = Reading.model_validate(orjson.loads(payload))
        assert reading.ts == pd.Timestamp("2019-03-01 00:05", tz="UTC")
    first = orjson.loads(msgs[0][1])
    assert list(first) == list(Reading.model_fields)  # field order of the contract
    assert first["unit"] == "m" and first["value"] == pytest.approx(40.5)


def test_missing_values_are_not_published():
    wide = wide_frame()
    metas = rp.sensor_meta(list(wide.columns))
    msgs = list(rp.encode_step("t", 2019, wide.iloc[0].to_numpy(), metas, "battledim", None))
    assert [k for k, _ in msgs] == [b"pressure__n54", b"flow__p227"]  # amr NaN skipped


def test_replayer_publishes_in_time_order_and_counts():
    producer = FakeProducer(fail_first=2)  # BufferError path retried
    replayer = rp.Replayer(producer, "t", rp.Pacer(None), progress_every_steps=0)
    stats = replayer.run(wide_frame(4).iloc[::-1])  # unsorted input
    assert stats.steps == 4
    assert stats.readings == 4 * 3 - 1 == len(producer.messages)
    ts = [orjson.loads(v)["ts"] for _, _, v in producer.messages]
    assert ts == sorted(ts)
    assert stats.first_ts == "2019-03-01T00:00:00Z"
    assert stats.delivery_errors == 0


def test_delivery_error_counter():
    stats = rp.ReplayStats()
    cb = rp.delivery_error_counter(stats)
    cb(None, object())
    cb("boom", object())
    assert stats.delivery_errors == 1


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_pacer_keeps_simulated_to_wall_ratio():
    fc = FakeClock()
    pacer = rp.Pacer(speed=300.0, clock=fc.clock, sleep=fc.sleep)  # 5 sim-minutes per wall second
    t0 = pd.Timestamp("2019-01-01 00:00", tz="UTC")
    assert pacer.wait_for(t0) == 0.0
    assert pacer.wait_for(t0 + pd.Timedelta(minutes=5)) == pytest.approx(1.0)
    fc.now += 3.0  # we fell behind: no extra sleeping to "catch up" backwards
    assert pacer.wait_for(t0 + pd.Timedelta(minutes=10)) == 0.0
    assert pacer.wait_for(t0 + pd.Timedelta(minutes=30)) == pytest.approx(2.0)


def test_pacer_max_speed_never_sleeps():
    fc = FakeClock()
    pacer = rp.Pacer(speed=None, clock=fc.clock, sleep=fc.sleep)
    for minutes in (0, 5, 10):
        pacer.wait_for(pd.Timestamp("2019-01-01", tz="UTC") + pd.Timedelta(minutes=minutes))
    assert fc.slept == []


def test_pacer_speed_change_reanchors():
    fc = FakeClock()
    pacer = rp.Pacer(speed=300.0, clock=fc.clock, sleep=fc.sleep)
    t0 = pd.Timestamp("2019-01-01", tz="UTC")
    pacer.wait_for(t0)
    pacer.set_speed(600.0, t0 + pd.Timedelta(minutes=5))
    assert pacer.wait_for(t0 + pd.Timedelta(minutes=15)) == pytest.approx(1.0)


def test_load_input_slices_by_date(tmp_path):
    path = tmp_path / "w.parquet"
    wide_frame(600).to_parquet(path)
    wide = rp.load_input(None, path, "2019-03-02", "2019-03-02")
    assert wide.index[0] == pd.Timestamp("2019-03-02", tz="UTC")
    assert (wide.index.date == pd.Timestamp("2019-03-02").date()).all()
    with pytest.raises(ValueError):
        rp.load_input(None, path, "2020-01-01", "2020-01-02")
