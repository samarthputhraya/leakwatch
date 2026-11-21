"""Live K-Means scoring and alerting inside the streaming job (PRD §7.3, Phase 3; D-022).

:class:`LiveScoring` is plugged into the feature stream's ``foreachBatch`` writer
(``stream.features_stream``). For each micro-batch it receives the newly emitted feature rows and

1. for a re-baselined detector (D-038), replaces each residual by its re-baselined value, using
   the stream's own recent windows as reference (seeded from the batch feature table for the days
   before the stream starts), then scores them with :class:`~leakwatch.bdt.scorer.KMeansScorer`
   (the batch evaluation's code);
2. runs one :class:`~leakwatch.bdt.alerts.AlertTracker` per (source, scenario), resuming from the
   state committed with the previous batch;
3. returns score rows and changed alert versions, which the writer stages next to the features
   and commits atomically with them; after the commit, alert versions go to Kafka ``alerts``.

The rows of one micro-batch are tiny (a few windows), so they are written by the Spark driver with
pyarrow inside the same commit protocol rather than by two extra Spark jobs per batch (latency).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from leakwatch.bdt.alerts import AlertRule, AlertTracker, epoch_seconds, fast_threshold
from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
from leakwatch.common.schemas import ALERT_LAKE_COLUMNS, SCORE_COLUMNS

Localizer = Callable[[dict[str, Any], pd.DataFrame], list[dict[str, Any]] | None]
#: (first window start of a stream, how far back) -> earlier feature rows to use as reference
Seeder = Callable[[pd.Timestamp, pd.Timedelta], pd.DataFrame]
UTC_US = pa.timestamp("us", tz="UTC")


def alerts_arrow_schema(include_origin: bool = False) -> pa.Schema:
    """Arrow schema of ``lake/alerts`` files (``origin`` is the partition directory)."""
    top = pa.list_(pa.struct([("sensor_id", pa.string()), ("contribution", pa.float64())]))
    gnn = pa.list_(pa.struct([("node_id", pa.string()), ("prob", pa.float64())]))
    types = {
        "alert_id": pa.string(),
        "version": pa.int32(),
        "origin": pa.string(),
        "model": pa.string(),
        "source": pa.string(),
        "scenario_id": pa.string(),
        "start_ts": UTC_US,
        "end_ts": UTC_US,
        "status": pa.string(),
        "kmeans_score_max": pa.float64(),
        "threshold": pa.float64(),
        "top_sensors": top,
        "area": pa.string(),
        "lstm_score_max": pa.float64(),
        "gnn_top5": gnn,
        "est_loss_m3h": pa.float64(),
        "windows_above": pa.int32(),
        "last_window_end": UTC_US,
        "created_at": UTC_US,
        "updated_at": UTC_US,
    }
    assert list(types) == list(ALERT_LAKE_COLUMNS)
    return pa.schema([(k, v) for k, v in types.items() if include_origin or k != "origin"])


@dataclass
class LiveScoring:
    """Score emitted feature windows and keep alerts up to date (see module docstring)."""

    scorer: KMeansScorer
    rule: AlertRule
    inflow_columns: Sequence[str]
    model: str = "kmeans"
    publish: Callable[[list[dict[str, Any]]], None] | None = None
    localize: Localizer | None = None
    stats: dict[str, int] = field(default_factory=lambda: {"windows": 0, "alerts": 0})
    # Re-localize an open alert until its windows reach this long after its start (hours; 0 =
    # localize once). A fast alert opens while its first hour is still mostly pre-leak (D-033).
    refine_hours: float = 0.0
    history_windows: int = 16  # feature rows kept per stream for the localizer (4 h at 15 min)
    recent: dict[str, pd.DataFrame] = field(default_factory=dict)
    # D-038: re-baselined residuals (None = score the features as they arrive)
    rebaseline: Any = None  # features.rebaseline.Rebaseline
    rebaseline_columns: Sequence[str] = ()
    seed: Seeder | None = None
    reference: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def from_model_dir(
        cls,
        directory: Path,
        cfg: Mapping[str, Any],
        series_area: Mapping[str, str],
        origin: str = "stream",
        publish: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> LiveScoring:
        """Load a trained detector (``bdt.train``) for live use."""
        params = KMeansParams.load(directory)
        if params.threshold is None or params.persistence_windows is None:
            raise ValueError(f"{directory}: model has no threshold/persistence (run bdt-train)")
        bdt = cfg["bdt"]
        rule = AlertRule(
            threshold=float(params.threshold),
            persistence_windows=int(params.persistence_windows),
            close_after_windows=int(bdt["close_after_windows"]),
            top_n=int(bdt["top_sensors"]),
            series_area=dict(series_area),
            origin=origin,
            model=str(bdt["model"]),
            fast_threshold=fast_threshold(cfg, directory.name, float(params.threshold)),
        )
        inflow = [f"{s}__residual" for s in bdt["inflow_series"]]
        live = cls(KMeansScorer(params), rule, inflow, str(bdt["model"]), publish)
        gnn = cfg.get("dl", {}).get("gnn", {})
        if "live_refine_hours" in gnn:
            live.refine_hours = float(gnn["live_refine_hours"])
        if params.meta.get("feature_view") == "rebaselined":
            from leakwatch.features.rebaseline import Rebaseline

            live.rebaseline = Rebaseline.from_config(cfg)
            live.rebaseline_columns = list(params.feature_columns)
            live.seed = batch_seeder(live.rebaseline_columns)
        return live

    def _rebaselined(self, key: str, part: pd.DataFrame) -> pd.DataFrame:
        """``part`` with its residual columns re-baselined against this stream's history."""
        from leakwatch.features.rebaseline import rebaselined

        cols = list(self.rebaseline_columns)
        hist = self.reference.get(key)
        if hist is None:
            hist = pd.DataFrame(columns=["window_start", *cols])
            if self.seed is not None:
                first = pd.Timestamp(part["window_start"].min())
                seeded = self.seed(first, self.rebaseline.history)
                if not seeded.empty:
                    hist = seeded[["window_start", *cols]]
        raw = part[["window_start", *cols]]
        both = pd.concat([hist, raw], ignore_index=True)
        both["window_start"] = pd.to_datetime(both["window_start"], utc=True)
        both = both.drop_duplicates("window_start", keep="last").sort_values("window_start")
        out = rebaselined(both, cols, self.rebaseline).set_index("window_start")
        cutoff = both["window_start"].max() - self.rebaseline.history
        self.reference[key] = both[both["window_start"] >= cutoff].reset_index(drop=True)
        new = part.copy()
        starts = pd.to_datetime(new["window_start"], utc=True)
        new[cols] = out.loc[starts, cols].fillna(0.0).to_numpy()  # no reference yet -> no change
        return new

    def _history(self, key: str, part: pd.DataFrame) -> pd.DataFrame:
        """This stream's latest feature rows including ``part`` (in memory, for the localizer)."""
        old = self.recent.get(key)
        rows = part if old is None else pd.concat([old, part], ignore_index=True)
        rows = rows.drop_duplicates("window_end", keep="last").sort_values("window_end")
        self.recent[key] = rows.tail(self.history_windows).reset_index(drop=True)
        return self.recent[key]

    def _wants_location(self, row: Mapping[str, Any]) -> bool:
        """Localize a new alert, and refine an open one during its first ``refine_hours``."""
        if row["gnn_top5"] is None:
            return True
        if row["status"] != "open" or self.refine_hours <= 0:
            return False
        age = pd.Timestamp(row["last_window_end"]) - pd.Timestamp(row["start_ts"])
        return age <= pd.Timedelta(hours=self.refine_hours)

    def process(
        self, features: pd.DataFrame, state: Mapping[str, Any] | None
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        """Score ``features`` (one micro-batch); return scores, changed alerts, new state."""
        new_state = dict(state or {})
        if features.empty:
            return pd.DataFrame(columns=list(SCORE_COLUMNS)), [], new_state
        features = features.sort_values(["source", "scenario_id", "window_start"]).reset_index(
            drop=True
        )
        if self.rebaseline is not None:
            keys0 = features["scenario_id"].fillna("")
            parts = [
                self._rebaselined(f"{source}|{scenario}", features.loc[np.asarray(idx)])
                for (source, scenario), idx in features.groupby(
                    [features["source"], keys0]
                ).groups.items()
            ]
            features = pd.concat(parts).sort_index()
        scored, shares = self.scorer.score_frame(features)
        excess_all = (
            features[list(self.inflow_columns)].sum(axis=1, min_count=1)
            if all(c in features for c in self.inflow_columns)
            else pd.Series(np.nan, index=features.index)
        )
        rows: list[dict[str, Any]] = []
        keys = features["scenario_id"].fillna("")
        for (source, scenario), idx in features.groupby([features["source"], keys]).groups.items():
            key = f"{source}|{scenario}"
            tracker = AlertTracker(
                self.rule, str(source), scenario or None, self.scorer.series, new_state.get(key)
            )
            pos = np.asarray(idx)
            changed = tracker.update(
                epoch_seconds(features.loc[pos, "window_end"]),
                scored.loc[pos, "score"].to_numpy(),
                shares[pos],
                [None if pd.isna(v) else float(v) for v in excess_all.loc[pos]],
            )
            if self.localize is not None:
                recent = self._history(key, features.loc[pos])
                for row in changed:
                    if self._wants_location(row):
                        cands = self.localize(row, recent)
                        if cands:
                            row["gnn_top5"] = cands
                            tracker.keep_candidates(row["alert_id"], cands)
            rows.extend(changed)
            new_state[key] = tracker.state()
        self.stats["windows"] += len(scored)
        self.stats["alerts"] += len(rows)
        return scored, rows, new_state


def batch_seeder(columns: Sequence[str]) -> Seeder:
    """Reference rows from the batch feature table (BattLeDIM) before a stream's first window."""

    def seed(first: pd.Timestamp, back: pd.Timedelta) -> pd.DataFrame:
        from leakwatch.common import lake

        col_sql = ", ".join(f'"{c}"' for c in columns)
        lo, hi = (first - back).isoformat(), first.isoformat()
        try:
            return lake.query_df(
                lake.connect(),
                f"SELECT window_start, {col_sql} FROM features WHERE source = 'battledim' "
                f"AND window_start >= TIMESTAMPTZ '{lo}' AND window_start < TIMESTAMPTZ '{hi}' "
                "ORDER BY window_start",
            )
        except Exception:  # no batch features (fresh lake): the stream builds its own history
            return pd.DataFrame(columns=["window_start", *columns])

    return seed


def write_scores_staged(scored: pd.DataFrame, stage: Path, model: str, origin: str) -> int:
    """Write score rows under ``stage/model=…/origin=…/date=…/`` (hive layout of lake/scores)."""
    if scored.empty:
        return 0
    df = scored[list(SCORE_COLUMNS)].copy()
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True)
    df["cluster"] = df["cluster"].astype("int32")
    df["above"] = df["above"].astype(bool)
    schema = pa.schema(
        [
            ("source", pa.string()),
            ("scenario_id", pa.string()),
            ("window_start", UTC_US),
            ("window_end", UTC_US),
            ("score", pa.float64()),
            ("threshold", pa.float64()),
            ("above", pa.bool_()),
            ("cluster", pa.int32()),
            ("top_sensor", pa.string()),
        ]
    )
    for day, part in df.groupby(df["window_start"].dt.strftime("%Y-%m-%d")):
        target = stage / f"model={model}" / f"origin={origin}" / f"date={day}"
        target.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(part, schema=schema, preserve_index=False, safe=False)
        pq.write_table(table, target / "part-00000.parquet", compression="zstd")
    return len(df)


def write_alerts_staged(rows: list[dict[str, Any]], stage: Path, origin: str) -> int:
    """Write alert versions under ``stage/origin=…/`` (hive layout of lake/alerts)."""
    if not rows:
        return 0
    schema = alerts_arrow_schema()
    clean = []
    for row in rows:
        rec = {}
        for name in schema.names:
            v = row.get(name)
            if isinstance(v, pd.Timestamp):
                v = (
                    v.tz_localize("UTC") if v.tzinfo is None else v.tz_convert("UTC")
                ).to_pydatetime()
            rec[name] = v
        clean.append(rec)
    target = stage / f"origin={origin}"
    target.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(clean, schema=schema), target / "part-00000.parquet")
    return len(rows)


def alert_message(row: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe alert version for the Kafka ``alerts`` topic."""
    out = {}
    for k, v in row.items():
        out[k] = v.isoformat() if isinstance(v, pd.Timestamp) else v
    return out
