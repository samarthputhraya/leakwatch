"""Feature specification: which series, windows and statistics make up one feature vector.

Built once from config (plus the BattLeDIM sensor list and the network areas) and passed to every
feature function, so the batch job and the streaming job cannot disagree about the layout of a
feature row (PRD §7.2).

Naming (all joined with ``schemas.COLUMN_SEP``, i.e. ``__``):

- a *series* is what statistics are computed on: one per pressure/flow/level sensor
  (``pressure__n54``, the wide-table column name) plus one summed AMR series per area
  (``amr_sum__C``);
- a *feature column* is ``<series>__<stat>``, e.g. ``pressure__n54__slope``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from leakwatch.common.schemas import COLUMN_SEP, SENSOR_TYPES, sensor_column

#: Statistics a window can carry, in the order they appear in a feature row.
STATS: tuple[str, ...] = ("mean", "std", "min", "max", "slope", "residual")
#: Per-window data-quality counts stored next to the features (not model inputs).
QUALITY_COLUMNS: tuple[str, ...] = (
    "n_readings",
    "n_filled",
    "n_stuck",
    "n_out_of_range",
    "n_series_missing",
)
AMR_SUM_PREFIX = "amr_sum"


def amr_sum_series(area: str) -> str:
    """Series id of the summed AMR consumption of one area, e.g. ``amr_sum__C``."""
    return f"{AMR_SUM_PREFIX}{COLUMN_SEP}{area}"


def feature_column(series_id: str, stat: str) -> str:
    """Feature column name, e.g. ``pressure__n54__mean``."""
    if stat not in STATS:
        raise ValueError(f"unknown stat {stat!r}")
    return f"{series_id}{COLUMN_SEP}{stat}"


def _natural(name: str) -> tuple[str, int, str]:
    digits = "".join(ch for ch in name if ch.isdigit())
    return (name.rstrip("0123456789"), int(digits) if digits else -1, name)


def window_label(window_s: int) -> str:
    """Partition label of a window length: ``1h`` for 3600 s, ``30min`` for 1800 s."""
    if window_s % 3600 == 0:
        return f"{window_s // 3600}h"
    if window_s % 60 == 0:
        return f"{window_s // 60}min"
    return f"{window_s}s"


@dataclass(frozen=True)
class FeatureSpec:
    """Everything that fixes how readings become feature rows (see module docstring)."""

    step_s: int
    window_s: int
    slide_s: int
    stats: tuple[str, ...]
    sensor_series: tuple[str, ...]
    amr_areas: Mapping[str, str]
    max_ffill_gap_steps: int
    stuck_min_steps: int
    stuck_exempt_types: tuple[str, ...]
    exclude_out_of_range: bool
    lateness_s: int

    def __post_init__(self) -> None:
        if min(self.step_s, self.window_s, self.slide_s) <= 0:
            raise ValueError("step, window and slide must be positive")
        if self.window_s % self.slide_s or self.slide_s % self.step_s:
            raise ValueError("window must be a multiple of slide, slide a multiple of step")
        if unknown := set(self.stats) - set(STATS):
            raise ValueError(f"unknown stats {sorted(unknown)}")
        if unknown := set(self.stuck_exempt_types) - set(SENSOR_TYPES):
            raise ValueError(f"unknown sensor types {sorted(unknown)}")
        if len(set(self.series)) != len(self.series):
            raise ValueError("series ids must be unique")
        # A window may only be emitted once the reading that closes a fillable gap after its last
        # step has arrived; otherwise the stream would skip a fill that the batch job makes.
        if self.lateness_s < (self.max_ffill_gap_steps + 1) * self.step_s:
            raise ValueError("lateness must cover max_ffill_gap_steps + 1 steps")

    @property
    def sensor_types(self) -> tuple[str, ...]:
        """Sensor types that have per-sensor series, in first-seen order."""
        return tuple(dict.fromkeys(s.split(COLUMN_SEP, 1)[0] for s in self.sensor_series))

    @property
    def area_series(self) -> tuple[str, ...]:
        """One summed AMR series per area that has AMRs, sorted by area."""
        return tuple(amr_sum_series(a) for a in sorted(set(self.amr_areas.values())))

    @property
    def series(self) -> tuple[str, ...]:
        """All series in feature-row order: per-sensor series, then AMR area sums."""
        return (*self.sensor_series, *self.area_series)

    @property
    def history_s(self) -> int:
        """Readings this far before a window are needed to clean the window's own readings."""
        return max(self.stuck_min_steps - 1, self.max_ffill_gap_steps + 1) * self.step_s

    @property
    def steps_per_window(self) -> int:
        """Time steps in one window (12 for 1 h of 5-minute data)."""
        return self.window_s // self.step_s

    @property
    def window_label(self) -> str:
        """Value of the ``window=`` partition, e.g. ``1h``."""
        return window_label(self.window_s)

    def feature_columns(self) -> list[str]:
        """Model input columns of a feature row, in order (series-major, then stat)."""
        return [feature_column(s, stat) for s in self.series for stat in self.stats]

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        sensors: Mapping[str, Sequence[str]],
        node_area: Mapping[str, str],
    ) -> FeatureSpec:
        """Build the spec from the merged config, the sensor lists and each node's area.

        Args:
            cfg: merged configuration (``cleaning`` and ``features`` sections are used).
            sensors: sensor ids per sensor type (``data.network.parse_sensor_config``).
            node_area: area label of every node (``data.network.node_areas``).

        """
        clean, feat = cfg["cleaning"], cfg["features"]
        series = [
            sensor_column(t, s)
            for t in feat["sensor_types"]
            for s in sorted(sensors.get(t, []), key=_natural)
        ]
        amr_areas = {s: str(node_area[s]) for s in sensors.get("amr", [])}
        return cls(
            step_s=int(clean["step_seconds"]),
            window_s=int(feat["window_minutes"]) * 60,
            slide_s=int(feat["slide_minutes"]) * 60,
            stats=tuple(feat["stats"]),
            sensor_series=tuple(series),
            amr_areas=amr_areas,
            max_ffill_gap_steps=int(clean["max_ffill_gap_steps"]),
            stuck_min_steps=int(clean["stuck_min_steps"]),
            stuck_exempt_types=tuple(clean.get("stuck_exempt_types") or ()),
            exclude_out_of_range=bool(feat["exclude_out_of_range"]),
            lateness_s=int(feat["stream"]["lateness_minutes"]) * 60,
        )


def load_feature_spec(cfg: Mapping[str, Any] | None = None) -> FeatureSpec:
    """Spec for L-Town: sensors from ``dataset_configuration.yaml``, areas from ``L-TOWN.inp``."""
    from leakwatch.common.config import load_config
    from leakwatch.data import network

    cfg = cfg if cfg is not None else load_config()
    wn = network.load_network()
    areas = network.node_areas(wn, cfg["network"]["area_anchors"])
    return FeatureSpec.from_config(cfg, network.parse_sensor_config(), areas.to_dict())
