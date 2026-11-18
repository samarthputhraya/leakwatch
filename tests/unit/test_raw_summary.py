"""Streaming-run summary statistics (pure Python; no Spark needed)."""

from __future__ import annotations

import pytest

from leakwatch.stream.raw_ingest import summarise_run


def progress(batch_id: int, rows: int, ms: int, ts: str, dup: int = 0, late: int = 0) -> dict:
    return {
        "batchId": batch_id,
        "numInputRows": rows,
        "timestamp": ts,
        "durationMs": {"triggerExecution": ms},
        "stateOperators": [
            {"numRowsDroppedByWatermark": late, "customMetrics": {"numDroppedDuplicateRows": dup}}
        ],
    }


def marker(batch_id: int, rows: int, committed: float, oldest: float, newest: float) -> dict:
    return {
        "batch_id": batch_id,
        "rows": rows,
        "rejects": 0,
        "out_of_range": 0,
        "committed_epoch_s": committed,
        "oldest_kafka_epoch_s": oldest,
        "newest_kafka_epoch_s": newest,
    }


def test_summary_separates_warmup_batch():
    prog = [
        progress(0, 100, 20_000, "2026-01-01T00:00:00.000Z", dup=1),
        progress(1, 100, 2_000, "2026-01-01T00:00:25.000Z", late=2),
        progress(2, 0, 50, "2026-01-01T00:00:30.000Z"),  # idle trigger: ignored
        progress(3, 100, 3_000, "2026-01-01T00:00:35.000Z"),
    ]
    marks = [
        marker(0, 99, committed=1020.0, oldest=1000.0, newest=1001.0),
        marker(1, 98, committed=1027.0, oldest=1021.0, newest=1024.0),
        marker(3, 100, committed=1038.0, oldest=1030.0, newest=1035.0),
    ]
    s = summarise_run(prog, marks)
    assert s["batches_with_data"] == 3
    assert s["kafka_records_read"] == 300
    assert s["rows_written"] == 297
    assert s["duplicates_dropped"] == 1
    assert s["late_rows_dropped_by_watermark"] == 2
    assert s["batch_duration_s"]["max"] == 20.0
    assert s["batches_over_target"] == 1
    assert s["first_batch_duration_s"] == 20.0
    steady = s["steady_state_excluding_first_batch"]
    assert steady["batches"] == 2
    assert steady["batch_duration_s"]["max"] == 3.0
    assert steady["batches_over_target"] == 0
    assert steady["publish_to_lake_s"]["oldest_in_batch"]["p95"] == pytest.approx(8.0)
    assert steady["publish_to_lake_s"]["newest_in_batch"]["p50"] == pytest.approx(3.0)
    assert s["processing_span_s"] == pytest.approx(35 + 3)


def test_summary_of_empty_run():
    s = summarise_run([], [])
    assert s["batches_with_data"] == 0
    assert s["batch_duration_s"]["p95"] is None
    assert s["steady_state_excluding_first_batch"]["batches"] == 0
