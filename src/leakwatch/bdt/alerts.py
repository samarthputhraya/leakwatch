"""Alert rule and alert records (PRD §6.3, §7.3; docs/DECISIONS.md D-020).

Per (source, scenario), windows are processed in time order:

- a window is *above* when its score exceeds the threshold τ;
- ``m`` consecutive windows above τ open an alert; its ``start_ts`` is the end of the m-th window,
  i.e. the event time at which the system could raise it (time to detection uses this);
- while open, every window above τ adds to the alert (max score, sensor shares, estimated loss);
- ``close_after`` consecutive windows below τ close it; ``end_ts`` = end of its last window above τ.

The tracker's state is a JSON-serialisable dict, so the streaming job commits it with each
micro-batch (like the feature tail, D-018) and the batch evaluation runs the same code over a year.

Explanations: the top sensors are the series with the largest mean share of the squared distance
over the alert's windows (``bdt.scorer``); the area is the top sensor's area. The estimated loss
is the mean excess inflow (sum of the inflow series' residuals against their hour-of-week
baselines, m³/h) over the alert's windows, floored at 0 and labelled as an estimate.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

ALERT_NAMESPACE = uuid.UUID("6f1d4f0e-5a2b-4c3e-9d7f-1a2b3c4d5e6f")


def alert_id_for(
    origin: str, model: str, source: str, scenario_id: str | None, start_s: int
) -> str:
    """Deterministic alert id (the same alert gets the same id if a batch is replayed)."""
    key = f"{origin}|{model}|{source}|{scenario_id or ''}|{start_s}"
    return str(uuid.uuid5(ALERT_NAMESPACE, key))


@dataclass(frozen=True)
class AlertRule:
    """Parameters of the alert rule and of the explanation."""

    threshold: float
    persistence_windows: int
    close_after_windows: int
    top_n: int
    series_area: Mapping[str, str]
    origin: str
    model: str

    def __post_init__(self) -> None:
        if self.persistence_windows < 1 or self.close_after_windows < 1:
            raise ValueError("persistence and close-after must be at least 1 window")


def epoch_seconds(ts: pd.Series) -> np.ndarray:
    """UTC timestamps of any resolution -> integer epoch seconds."""
    since = pd.to_datetime(ts, utc=True) - pd.Timestamp(0, tz="UTC")
    return (since // pd.Timedelta(seconds=1)).to_numpy()


def _ts(epoch_s: int | None) -> pd.Timestamp | None:
    return None if epoch_s is None else pd.Timestamp(epoch_s, unit="s", tz="UTC")


class AlertTracker:
    """Runs the alert rule for one (source, scenario) stream of scored windows."""

    def __init__(
        self,
        rule: AlertRule,
        source: str,
        scenario_id: str | None,
        series: Sequence[str],
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.rule = rule
        self.source = source
        self.scenario_id = scenario_id
        self.series = list(series)
        st = dict(state or {})
        self.run: list[dict[str, Any]] = st.get("run", [])  # windows above tau, alert not open
        self.below: int = int(st.get("below", 0))
        self.alert: dict[str, Any] | None = st.get("alert")
        self.last_end_s: int | None = st.get("last_end_s")

    # ------------------------------------------------------------------------------- state

    def state(self) -> dict[str, Any]:
        """JSON-serialisable state (``AlertTracker(..., state=tracker.state())`` resumes)."""
        return {
            "run": self.run,
            "below": self.below,
            "alert": self.alert,
            "last_end_s": self.last_end_s,
        }

    # ------------------------------------------------------------------------------ update

    def update(
        self,
        window_end_s: Sequence[int],
        scores: Sequence[float],
        shares: np.ndarray,
        inflow_excess: Sequence[float | None],
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Feed windows (in time order); return one row per alert that changed (its latest state).

        Args:
            window_end_s: window end times, epoch seconds, increasing.
            scores: anomaly score of each window.
            shares: ``(n, n_series)`` share of each series in the squared distance.
            inflow_excess: excess inflow of each window (m³/h), or None if unknown.
            now: wall-clock time stamped on the rows (default: now, UTC).

        """
        now = now or datetime.now(UTC)
        changed: dict[str, dict[str, Any]] = {}
        thr = self.rule.threshold
        for i, end_s in enumerate(window_end_s):
            end_s = int(end_s)
            if self.last_end_s is not None and end_s <= self.last_end_s:
                continue  # already seen (replayed batch)
            self.last_end_s = end_s
            score = float(scores[i])
            excess = inflow_excess[i]
            if score > thr:
                self.below = 0
                window = {
                    "end_s": end_s,
                    "score": score,
                    "share": [float(v) for v in shares[i]],
                    "excess": None if excess is None or pd.isna(excess) else float(excess),
                }
                if self.alert is None:
                    self.run.append(window)
                    if len(self.run) >= self.rule.persistence_windows:
                        self._open(now)
                        changed[self.alert["alert_id"]] = self.alert  # type: ignore[index]
                else:
                    self._add(self.alert, window)
                    changed[self.alert["alert_id"]] = self.alert
            else:
                self.run = []
                if self.alert is not None:
                    self.below += 1
                    if self.below >= self.rule.close_after_windows:
                        self.alert["status"] = "closed"
                        self.alert["end_s"] = self.alert["last_window_end_s"]
                        changed[self.alert["alert_id"]] = self.alert
                        self.alert = None
                        self.below = 0
        rows = []
        for alert in changed.values():
            alert["version"] += 1
            alert["updated_at"] = now.isoformat()
            rows.append(self.to_row(alert))
        return rows

    def keep_candidates(
        self, alert_id: str, candidates: Sequence[Mapping[str, Any]] | None
    ) -> None:
        """Remember the localizer's candidates on the open alert, so later versions carry them.

        The live localizer needs the windows just before the alert start, which only the
        opening micro-batch has; without this, every update of the alert dropped them.
        """
        if candidates and self.alert is not None and self.alert["alert_id"] == alert_id:
            self.alert["gnn_top5"] = [
                {"node_id": str(c["node_id"]), "prob": float(c["prob"])} for c in candidates
            ]

    def _open(self, now: datetime) -> None:
        first = self.run[-1]  # the m-th window: the alert fires at its end
        alert = {
            "alert_id": alert_id_for(
                self.rule.origin, self.rule.model, self.source, self.scenario_id, first["end_s"]
            ),
            "version": 0,
            "start_s": first["end_s"],
            "end_s": None,
            "status": "open",
            "score_max": 0.0,
            "windows_above": 0,
            "share_sum": [0.0] * len(self.series),
            "excess_sum": 0.0,
            "excess_n": 0,
            "last_window_end_s": first["end_s"],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        for window in self.run:
            self._add(alert, window)
        self.run = []
        self.alert = alert

    @staticmethod
    def _add(alert: dict[str, Any], window: dict[str, Any]) -> None:
        alert["score_max"] = max(alert["score_max"], window["score"])
        alert["windows_above"] += 1
        alert["share_sum"] = [
            a + b for a, b in zip(alert["share_sum"], window["share"], strict=True)
        ]
        if window["excess"] is not None:
            alert["excess_sum"] += window["excess"]
            alert["excess_n"] += 1
        alert["last_window_end_s"] = max(alert["last_window_end_s"], window["end_s"])

    # --------------------------------------------------------------------------- output row

    def to_row(self, alert: Mapping[str, Any]) -> dict[str, Any]:
        """Alert dict -> row with ``schemas.ALERT_LAKE_COLUMNS``."""
        n = max(alert["windows_above"], 1)
        mean_share = np.asarray(alert["share_sum"]) / n
        order = np.argsort(mean_share)[::-1][: self.rule.top_n]
        top = [
            {"sensor_id": self.series[j], "contribution": round(float(mean_share[j]), 4)}
            for j in order
        ]
        area = self.rule.series_area.get(top[0]["sensor_id"]) if top else None
        loss = max(0.0, alert["excess_sum"] / alert["excess_n"]) if alert["excess_n"] else None
        return {
            "alert_id": alert["alert_id"],
            "version": int(alert["version"]),
            "origin": self.rule.origin,
            "model": self.rule.model,
            "source": self.source,
            "scenario_id": self.scenario_id,
            "start_ts": _ts(alert["start_s"]),
            "end_ts": _ts(alert["end_s"]),
            "status": alert["status"],
            "kmeans_score_max": float(alert["score_max"]),
            "threshold": float(self.rule.threshold),
            "top_sensors": top,
            "area": area,
            "lstm_score_max": None,
            "gnn_top5": alert.get("gnn_top5"),
            "est_loss_m3h": None if loss is None else round(float(loss), 3),
            "windows_above": int(alert["windows_above"]),
            "last_window_end": _ts(alert["last_window_end_s"]),
            "created_at": pd.Timestamp(alert["created_at"]),
            "updated_at": pd.Timestamp(alert["updated_at"]),
        }


def run_alerts(
    scored: pd.DataFrame,
    shares: np.ndarray,
    series: Sequence[str],
    rule: AlertRule,
    inflow_columns: Sequence[str] = (),
    features: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Batch mode: run the rule over scored windows (all sources/scenarios); final alert states.

    Args:
        scored: ``source, scenario_id, window_start, window_end, score`` (any order).
        shares: ``series_share`` matrix aligned with ``scored`` rows.
        series: series names of the share columns.
        rule: alert rule.
        inflow_columns: residual feature columns whose sum is the excess inflow (m³/h).
        features: feature rows aligned with ``scored`` (needed for ``inflow_columns``).
        now: wall-clock time stamped on the rows.

    """
    df = scored.reset_index(drop=True).copy()
    df["_row"] = np.arange(len(df))
    if inflow_columns and features is not None:
        df["_excess"] = features.reset_index(drop=True)[list(inflow_columns)].sum(
            axis=1, min_count=1
        )
    else:
        df["_excess"] = np.nan
    rows: list[dict[str, Any]] = []
    keys = df["scenario_id"].fillna("\x00")
    for (_, _), part in df.groupby([df["source"], keys], sort=False):
        part = part.sort_values("window_start")
        scenario = part["scenario_id"].iloc[0]
        tracker = AlertTracker(
            rule, part["source"].iloc[0], None if pd.isna(scenario) else scenario, series
        )
        ends = epoch_seconds(part["window_end"])
        latest = tracker.update(
            ends,
            part["score"].to_numpy(),
            shares[part["_row"].to_numpy()],
            [None if pd.isna(v) else float(v) for v in part["_excess"]],
            now=now,
        )
        rows.extend(latest)
    from leakwatch.common.schemas import ALERT_LAKE_COLUMNS

    return pd.DataFrame(rows, columns=list(ALERT_LAKE_COLUMNS))
