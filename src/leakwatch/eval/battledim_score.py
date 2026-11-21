"""Official BattLeDIM 2020 score of a list of leak detections (D-037).

A port of the competition's own scoring code (KIOS-Research/BattLeDIM, ``Scoring Algorithm/
Scoring_Algorithm.m`` and ``functions/scoring_function.m``, Vrachimis et al. 2022, JWRPM,
doi:10.1061/(ASCE)WR.1943-5452.0001601), so LeakWatch can be placed on the competition's results
table. A *detection* is a pipe ID and a time. Rules, as in the MATLAB code:

- times are 5-minute steps from 2019-01-01 00:00; detections dated in 2018 are ignored (and,
  when a year is scored from a longer list, those after it);
- detections are scored in time order (stable sort). For every detection and leak, the distance
  ``x`` is the shortest pipe-length path between the two links' end nodes plus half of each
  link's length (0 for the same link). Pumps and valves are not part of the distance graph;
- a detection hits a leak if the leak is running (``start ≤ t ≤ end``) and ``x ≤ 300 m``. It is
  a true positive for the closest leak it hits that has not been detected yet, worth the water
  that leak loses from then to its end at €0.80/m³ (``sum(flow[t:]) / 12``) minus a crew cost
  ``x / 300 · €500``. A detection that only hits leaks already detected scores 0 (not a false
  positive); a detection that hits nothing is a false positive and costs €500;
- the 23 leaks are those running in 2019 (19 that start in 2019 and 4 left over from 2018).

The same rules applied to 2018 (its 14 leaks, steps from 2018-01-01) give a development score:
every design choice is made on 2018 and simulated leaks; 2019 is scored as the blind test.

``make battledim-score`` scores the system's batch alerts of both years (one detection per alert)
with two placements — the pipe at the localizer's top candidate, and the pipe at the pressure
sensor that dropped most (``bdt.report``) — and reports the system with the one that scores better
on 2018 (D-041); also the perfect submission (a self-check: the paper's Table 4 gives €523,154).

Then two dispatch rules (D-042, ``bdt.report``): no second crew to a pipe within ``d`` days (``d``
from ``battledim_score.revisit_days_grid``, best 2018 score, ties to the shorter window), and the
2018 reports that no repair in the utility's 2018 repair report explains carried into 2019 as open
work orders on 1 January. The carry-forward has nothing to tune and cannot be tried on 2018 (there
is no 2017); the 2019 score is reported with and without it (``steps_2019``).

Random-timing baseline: ``random_baseline_draws`` sets of alert times drawn uniformly from each
year (as many as the real alerts), placed and filtered by the same rules, without the backlog.
The one-sided p-value says how often chance timing scores at least as well as the system.

Writes ``reports/metrics/battledim_score.json``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from leakwatch.common.logging import get_logger

log = get_logger("leakwatch.eval.battledim_score")

STEP = pd.Timedelta(minutes=5)
YEAR_START = pd.Timestamp("2019-01-01 00:00", tz="UTC")
YEAR_LAST = pd.Timestamp("2019-12-31 23:55", tz="UTC")
#: BattLeDIM results, Table 4 of Vrachimis et al. (2022): the six top-ranked of 18 teams.
LEADERBOARD: tuple[dict[str, Any], ...] = (
    {"team": "Tongji-Team", "tp_rate": 0.5652, "false_positives": 3, "score_eur": 264_873},
    {"team": "Under Pressure", "tp_rate": 0.6522, "false_positives": 4, "score_eur": 260_562},
    {"team": "IRI", "tp_rate": 0.4347, "false_positives": 1, "score_eur": 210_772},
    {"team": "Leakbusters", "tp_rate": 0.4783, "false_positives": 7, "score_eur": 195_490},
    {"team": "Tsinghua", "tp_rate": 0.4783, "false_positives": 5, "score_eur": 167_981},
    {"team": "UNIFE", "tp_rate": 0.4347, "false_positives": 4, "score_eur": 127_626},
)
PERFECT_SCORE_EUR = 523_154  # Table 4, "PERFECT"


@dataclass(frozen=True)
class ScoringParams:
    """The competition's scoring parameters (``SP`` in ``Scoring_Algorithm.m``)."""

    max_distance_m: float = 300.0
    cost_water_eur_m3: float = 0.80
    cost_crew_eur: float = 500.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> ScoringParams:
        """Read ``battledim_score`` from the config."""
        s = cfg["battledim_score"]
        return cls(
            float(s["max_distance_m"]), float(s["cost_water_eur_m3"]), float(s["cost_crew_eur"])
        )


@dataclass(frozen=True)
class OfficialLeak:
    """One scored leak: pipe, start/end step and its 2019 outflow (m³/h every 5 min)."""

    link_id: str
    start_step: int
    end_step: int
    flow_m3h: np.ndarray


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First instant and last 5-minute step of a scored year (UTC)."""
    start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
    return start, pd.Timestamp(f"{year + 1}-01-01 00:00", tz="UTC") - STEP


def to_step(ts: Any, year: int = 2019) -> int:
    """5-minute steps from 1 January of ``year`` (``time2steps.m``; negative before it)."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return round((t - year_bounds(year)[0]) / STEP)


def official_leaks(gt: pd.DataFrame, flows: pd.DataFrame, year: int = 2019) -> list[OfficialLeak]:
    """The leaks running in ``year``, in start order; one never repaired runs to the year's end."""
    first, last = year_bounds(year)
    running = gt[
        (gt["start_ts"] <= last) & (gt["end_ts"].isna() | (gt["end_ts"] >= first))
    ].sort_values("start_ts", kind="stable")
    out = []
    for leak in running.itertuples(index=False):
        end = last if pd.isna(leak.end_ts) else min(leak.end_ts, last)
        out.append(
            OfficialLeak(
                link_id=str(leak.pipe_id),
                start_step=to_step(leak.start_ts, year),
                end_step=to_step(end, year),
                flow_m3h=flows[leak.pipe_id].to_numpy(dtype="float64"),
            )
        )
    return out


class LinkDistances:
    """Pipe-length distances between link centres (``nodeTopologicalDistance.m`` + scoring)."""

    def __init__(self, wn: Any) -> None:
        """Shortest paths over pipes only: links of length 0 are no edge in the MATLAB graph."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        nodes = list(wn.node_name_list)
        index = {n: i for i, n in enumerate(nodes)}
        self.links: dict[str, tuple[int, int, float]] = {}
        edges: dict[tuple[int, int], float] = {}
        for name, link in wn.links():
            a, b = index[link.start_node_name], index[link.end_node_name]
            length = float(link.length) if link.link_type == "Pipe" else 0.0
            self.links[name] = (a, b, length)
            if length > 0:  # parallel pipes: the MATLAB matrix keeps the last length written
                edges[(min(a, b), max(a, b))] = length
        n = len(nodes)
        keys = list(edges)
        graph = coo_matrix(
            ([edges[k] for k in keys], ([k[0] for k in keys], [k[1] for k in keys])), shape=(n, n)
        ).tocsr()
        self.node_distance = dijkstra(graph, directed=False)
        self.nodes = nodes
        self.index = index
        self._incident: dict[str, list[str]] = {}
        for name, (a, b, length) in self.links.items():
            if length > 0:
                self._incident.setdefault(nodes[a], []).append(name)
                self._incident.setdefault(nodes[b], []).append(name)

    def between(self, link_a: str, link_b: str) -> float:
        """Distance between two link centres (m); 0 for the same link."""
        if link_a == link_b:
            return 0.0
        a1, a2, la = self.links[link_a]
        b1, b2, lb = self.links[link_b]
        d = self.node_distance
        return float(min(d[a1, b1], d[a1, b2], d[a2, b1], d[a2, b2]) + la / 2 + lb / 2)

    def link_at_node(self, node: str) -> str | None:
        """The shortest pipe touching ``node`` (its centre is nearest to the node)."""
        pipes = self._incident.get(node)
        if not pipes:
            return None
        return min(pipes, key=lambda p: (self.links[p][2], p))


def score_detections(
    detections: pd.DataFrame,
    leaks: Sequence[OfficialLeak],
    dist: LinkDistances,
    params: ScoringParams,
    year: int = 2019,
) -> dict[str, Any]:
    """Score ``detections`` (columns ``link_id``, ``ts``) like ``Scoring_Algorithm.m``."""
    det = detections.assign(step=[to_step(t, year) for t in detections["ts"]])
    last = to_step(year_bounds(year)[1], year)
    in_year = (det["step"] >= 0) & (det["step"] <= last)  # the official files hold one year only
    det = det[in_year].sort_values("step", kind="stable").reset_index(drop=True)
    detected: list[int] = []
    rows = []
    for d in det.itertuples(index=False):
        x = np.array([dist.between(leak.link_id, d.link_id) for leak in leaks])
        hits = [
            j
            for j, leak in enumerate(leaks)
            if leak.start_step <= d.step <= leak.end_step and abs(x[j]) <= params.max_distance_m
        ]
        new = [j for j in hits if j not in detected]
        row: dict[str, Any] = {
            "link_id": d.link_id,
            "ts": pd.Timestamp(d.ts).isoformat(),
            "leak": None,
            "distance_m": None,
            "score_eur": -params.cost_crew_eur,
            "label": "fp",
        }
        if hits and not new:
            row.update(score_eur=0.0, label="repeat")
        elif new:
            j = min(new, key=lambda k: (x[k], k))
            leak = leaks[j]
            td = max(int(d.step), 1)
            water_m3 = float(leak.flow_m3h[td - 1 :].sum() / 12)
            score = water_m3 * params.cost_water_eur_m3 - (x[j] / params.max_distance_m) * (
                params.cost_crew_eur
            )
            detected.append(j)
            row.update(
                leak=leak.link_id,
                distance_m=round(float(x[j]), 1),
                score_eur=round(float(score), 2),
                label="tp",
                delay_h=round((d.step - leak.start_step) * STEP.total_seconds() / 3600, 2),
                water_saved_m3=round(water_m3, 1),
            )
        rows.append(row)
    tp = len(detected)
    fp = sum(r["label"] == "fp" for r in rows)
    total = round(float(sum(r["score_eur"] for r in rows)), 2)
    return {
        "total_score_eur": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": len(leaks) - tp,
        "repeated_detections": sum(r["label"] == "repeat" for r in rows),
        "tp_rate": round(tp / len(leaks), 4) if leaks else None,
        "leaks": len(leaks),
        "detections": len(rows),
        "detected_leaks": [leaks[j].link_id for j in detected],
        "missed_leaks": [leak.link_id for j, leak in enumerate(leaks) if j not in detected],
        "per_detection": rows,
    }


def perfect_detections(leaks: Sequence[OfficialLeak], year: int = 2019) -> pd.DataFrame:
    """Every leak at its own pipe at its start (older leaks at 00:05 on 1 January, as KIOS do)."""
    start = year_bounds(year)[0]
    return pd.DataFrame(
        {
            "link_id": [leak.link_id for leak in leaks],
            "ts": [start + STEP * max(leak.start_step, 1) for leak in leaks],
        }
    )


def rank(score_eur: float) -> dict[str, Any]:
    """Where a score would place among the six tabulated teams (of 18)."""
    better = sum(t["score_eur"] > score_eur for t in LEADERBOARD)
    last = min(t["score_eur"] for t in LEADERBOARD)
    return {
        "rank_among_top6": better + 1 if score_eur >= last else None,
        "above_all_tabulated": better == 0,
        "below_tabulated": score_eur < last,
        "sixth_place_eur": last,
        "teams_in_competition": 18,
    }


def detections_from_alerts(alerts: pd.DataFrame, dist: LinkDistances) -> pd.DataFrame:
    """One detection per alert: its start time and the pipe at the localizer's top candidate.

    An alert without localizer candidates falls back to its top sensor's node (a pressure sensor
    is a junction); alerts without either are left out and counted.
    """
    rows = []
    for a in alerts.itertuples(index=False):
        node, how = None, None
        cands = getattr(a, "gnn_top5", None)
        if cands is not None and len(cands):
            node, how = _first_node(cands), "localizer"
        if node is None:
            sensors = getattr(a, "top_sensors", None)
            if sensors is not None and len(sensors):
                node, how = _first_node(sensors, key="sensor_id"), "top_sensor"
        link = dist.link_at_node(node) if node else None
        if link is not None:
            rows.append(
                {"alert_id": str(a.alert_id), "ts": a.start_ts, "link_id": link, "how": how}
            )
    return pd.DataFrame(rows, columns=["alert_id", "ts", "link_id", "how"])


def _first_node(items: Any, key: str = "node_id") -> str | None:
    first = next(iter(items))
    return str(first[key]) if isinstance(first, Mapping) else str(getattr(first, key, first))


class YearScorer:
    """Scores detections of 2018 (development) or 2019 (test) with the official rules."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        """Load the ground truth, both years' leak flows and the network distances."""
        from leakwatch.common import paths
        from leakwatch.data import network
        from leakwatch.data.ground_truth import load_ground_truth

        self.params = ScoringParams.from_config(cfg)
        self.dist = LinkDistances(network.load_network())
        gt = load_ground_truth()
        self.leaks = {
            year: official_leaks(
                gt,
                pd.read_parquet(paths.battledim_parquet_dir() / f"leak_flows_{year}.parquet"),
                year,
            )
            for year in (2018, 2019)
        }

    def score(self, detections: pd.DataFrame, year: int) -> dict[str, Any]:
        """Official score of ``detections`` (``link_id``, ``ts``) in ``year``."""
        return score_detections(detections, self.leaks[year], self.dist, self.params, year)

    def detections(
        self,
        alerts: pd.DataFrame,
        year: int,
        place: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The alerts that start in ``year`` and their detections (one per alert)."""
        first, last = year_bounds(year)
        in_year = alerts[(alerts["start_ts"] >= first) & (alerts["start_ts"] <= last)]
        dets = place(in_year) if place is not None else detections_from_alerts(in_year, self.dist)
        return in_year, dets

    def score_alerts(
        self,
        alerts: pd.DataFrame,
        year: int,
        place: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """One detection per alert, scored in ``year``.

        ``place`` turns alerts into detections; default :func:`detections_from_alerts` (the
        localizer's top candidate, else the top sensor).
        """
        in_year, dets = self.detections(alerts, year, place)
        out = self.score(dets, year)
        out["alerts"] = len(in_year)
        how = dets["how"] if "how" in dets else pd.Series(["sensor_drop"] * len(dets))
        out["localized_by"] = how.value_counts().to_dict() if len(dets) else {}
        return out


#: How an alert becomes a report. The system uses the one with the better 2018 score (D-041).
PLACEMENTS = ("localizer", "sensor_drop")


def choose_placement(scores_2018: Mapping[str, float]) -> str:
    """The placement with the best 2018 (development) score; the localizer wins a tie."""
    return max(PLACEMENTS, key=lambda p: (scores_2018[p], p == "localizer"))


def choose_revisit_days(scores_2018: Mapping[float, float]) -> float:
    """The revisit window with the best 2018 score; ties go to the shorter window."""
    return max(scores_2018, key=lambda d: (scores_2018[d], -d))


def random_alert_times(year: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n`` distinct alert start times drawn uniformly from the year's 15-minute grid."""
    first, last = year_bounds(year)
    slots = int((last - first) / pd.Timedelta(minutes=15))  # slots 1..slots, never 00:00 1 Jan
    k = np.sort(rng.choice(np.arange(1, slots + 1), size=min(n, slots), replace=False))
    return pd.DataFrame(
        {
            "alert_id": [f"random-{i}" for i in range(len(k))],
            "start_ts": first + pd.to_timedelta(k * 15, unit="min"),
        }
    )


def empirical_p_value(observed: float, draws: Sequence[float]) -> float:
    """One-sided p of ``observed`` against random draws: (1 + #draws ≥ observed) / (n + 1)."""
    return (1 + sum(d >= observed for d in draws)) / (len(draws) + 1)


def main(argv: list[str] | None = None) -> int:
    """Score the lake's batch alerts of 2018 and 2019 and the perfect submission."""
    parser = argparse.ArgumentParser(description="Official BattLeDIM score of the alerts.")
    parser.parse_args(argv)
    from leakwatch.bdt.data import load_features
    from leakwatch.bdt.report import carry_forward, revisit_filter, sensor_drop_detections
    from leakwatch.bdt.train import feature_view
    from leakwatch.common import lake
    from leakwatch.common.config import load_config
    from leakwatch.common.metrics import write_metrics
    from leakwatch.data.battledim import read_fixed_leakages
    from leakwatch.features.spec import load_feature_spec

    cfg = load_config()
    bcfg = cfg["battledim_score"]
    scorer = YearScorer(cfg)
    perfect = scorer.score(perfect_detections(scorer.leaks[2019]), 2019)
    alerts = lake.query_df(
        lake.connect(),
        "SELECT alert_id, start_ts, model, top_sensors, gnn_top5 FROM alerts "
        "WHERE origin = 'batch'",
    )
    spec = load_feature_spec(cfg)
    feats = pd.concat([load_features(2018), load_features(2019)], ignore_index=True)
    drops = feature_view(feats, spec, "rebaselined", cfg)
    place = {
        "localizer": None,
        "sensor_drop": lambda a: sensor_drop_detections(a, drops, spec, scorer.dist),
    }
    results = {
        p: {year: scorer.score_alerts(alerts, year, place[p]) for year in (2018, 2019)}
        for p in PLACEMENTS
    }
    chosen = choose_placement({p: results[p][2018]["total_score_eur"] for p in PLACEMENTS})

    # Dispatch rules (D-042): the revisit window on 2018, then the 2018 backlog into 2019.
    placed = {year: scorer.detections(alerts, year, place[chosen]) for year in (2018, 2019)}
    grid = [float(d) for d in bcfg["revisit_days_grid"]]
    revisit_2018 = {
        d: scorer.score(revisit_filter(placed[2018][1], d), 2018)["total_score_eur"] for d in grid
    }
    days = choose_revisit_days(revisit_2018)
    final = {2018: revisit_filter(placed[2018][1], days)}
    no_carry_2019 = revisit_filter(placed[2019][1], days)
    carry: dict[str, Any] = {"enabled": bool(bcfg["carry_forward"])}
    carried = pd.DataFrame(columns=["ts", "link_id"])
    if carry["enabled"]:
        carried, summary = carry_forward(
            final[2018],
            read_fixed_leakages(),
            scorer.dist.between,
            scorer.params.max_distance_m,
            year_bounds(2019)[0] + STEP,
        )
        carry.update(summary)
    final[2019] = revisit_filter(pd.concat([carried, placed[2019][1]], ignore_index=True), days)

    years = {}
    for year in (2018, 2019):
        res = scorer.score(final[year], year)
        per = res.pop("per_detection")
        years[str(year)] = {
            "role": "development (rules applied to 2018)" if year == 2018 else "test (official)",
            "placement": chosen,
            "revisit_days": days,
            **res,
            "alerts": len(placed[year][0]),
            "carried_forward": len(carried) if year == 2019 else 0,
            **(rank(res["total_score_eur"]) if year == 2019 else {}),
            "per_detection": per,
        }
    keep = ("total_score_eur", "true_positives", "false_positives", "detections", "alerts")
    placements = {
        p: {str(y): {k: results[p][y][k] for k in keep} for y in (2018, 2019)} for p in PLACEMENTS
    }
    steps_2019 = {
        "placement_only": results[chosen][2019]["total_score_eur"],
        "with_revisit_rule": scorer.score(no_carry_2019, 2019)["total_score_eur"],
        "with_carry_forward": years["2019"]["total_score_eur"],
    }

    # Random-timing baseline: same number of alerts, same placement and revisit rule, no backlog.
    draws = int(bcfg["random_baseline_draws"])
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    observed = {2018: years["2018"]["total_score_eur"], 2019: steps_2019["with_revisit_rule"]}
    random_baseline: dict[str, Any] = {}
    for year in (2018, 2019):
        n = len(placed[year][0])
        sets = [random_alert_times(year, n, rng) for _ in range(draws)]
        if chosen == "sensor_drop":
            dets = [place["sensor_drop"](s) for s in sets]
        else:
            dets = _localize_random(sets, scorer.dist, cfg)
        totals = [scorer.score(revisit_filter(d, days), year)["total_score_eur"] for d in dets]
        random_baseline[str(year)] = {
            "draws": draws,
            "alerts_per_draw": n,
            "mean_eur": round(float(np.mean(totals)), 2),
            "p95_eur": round(float(np.percentile(totals, 95)), 2),
            "max_eur": round(float(np.max(totals)), 2),
            "system_eur": observed[year],
            "p_value": round(empirical_p_value(observed[year], totals), 4),
        }

    payload = {
        "source": "KIOS-Research/BattLeDIM Scoring_Algorithm.m, ported (D-037)",
        "params": scorer.params.__dict__,
        "detector": cfg["bdt"]["live_model"],
        "perfect_submission_2019": {
            "total_score_eur": perfect["total_score_eur"],
            "paper_table4_eur": PERFECT_SCORE_EUR,
        },
        "leaderboard_top6": list(LEADERBOARD),
        "placement_rule": "the placement with the best 2018 score (localizer on a tie), D-041",
        "placements": placements,
        "placement": chosen,
        "revisit_rule": {
            "rule": "no second report at a pipe within d days of the last one kept there; d with "
            "the best 2018 score, ties to the shorter window (D-042)",
            "scores_2018": {f"{d:g}": v for d, v in revisit_2018.items()},
            "chosen_days": days,
        },
        "carry_forward": carry,
        "steps_2019": steps_2019,
        "random_baseline": random_baseline,
        "years": years,
    }
    path = write_metrics("battledim_score", payload)
    for year, r in years.items():
        log.info(
            "%s: %.0f EUR, TP %d of %d, FP %d (random timing p = %s)",
            year,
            r["total_score_eur"],
            r["true_positives"],
            r["leaks"],
            r["false_positives"],
            random_baseline[year]["p_value"],
        )
    log.info(
        "perfect 2019 %.0f EUR (paper %d) -> %s",
        perfect["total_score_eur"],
        PERFECT_SCORE_EUR,
        path,
    )
    return 0


def _localize_random(
    sets: Sequence[pd.DataFrame], dist: LinkDistances, cfg: Mapping[str, Any]
) -> list[pd.DataFrame]:
    """Detections at the system localizer's top node for each set of random alert times."""
    from leakwatch.dl import gnn

    times = [t for s in sets for t in s["start_ts"]]
    nodes = gnn.localize_times(times, gnn.SystemLocalizer(gnn.localizer_config(cfg)))
    out, i = [], 0
    for s in sets:
        rows = []
        for ts in s["start_ts"]:
            link = dist.link_at_node(nodes[i])
            i += 1
            if link is not None:
                rows.append({"ts": ts, "link_id": link})
        out.append(pd.DataFrame(rows, columns=["ts", "link_id"]))
    return out


if __name__ == "__main__":
    sys.exit(main())
