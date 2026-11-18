"""Metamorphic and simulator-as-oracle tests (PRD §11) on real L-Town data and trained models.

Leaks are simulated with EPANET (``demo.signature``: leak run minus no-leak run) and added to a
quiet BattLeDIM 2018 day, the same way the live demo does it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakwatch.common import paths

pytestmark = [
    pytest.mark.spark,
    pytest.mark.needs_data,
    pytest.mark.needs_features,
    pytest.mark.slow,
]

QUIET_START = pd.Timestamp("2018-11-14 00:00", tz="UTC")  # normal under D-015 (Nov 8 - Dec 31)
LEAK_AT = QUIET_START + pd.Timedelta(hours=6)
NODE = "n100"


def long_from_wide(wide: pd.DataFrame) -> pd.DataFrame:
    from leakwatch.data import battledim

    return battledim.to_long(wide).assign(out_of_range=False)


def spark_readings(spark, wide: pd.DataFrame):
    from pyspark.sql import functions as F

    long = long_from_wide(wide)
    rows = [
        (
            r.source,
            r.scenario_id,
            r.sensor_type,
            r.sensor_id,
            int(r.ts.timestamp()),
            float(r.value),
            False,
        )
        for r in long.itertuples(index=False)
    ]
    ddl = (
        "source string, scenario_id string, sensor_type string, sensor_id string,"
        " t long, value double, out_of_range boolean"
    )
    return spark.createDataFrame(rows, ddl).withColumn("ts", F.timestamp_seconds("t")).drop("t")


@pytest.fixture(scope="module")
def setup(spark):
    from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
    from leakwatch.common.config import load_config
    from leakwatch.data import battledim
    from leakwatch.features.baselines import read_baselines
    from leakwatch.features.spec import load_feature_spec

    cfg = load_config()
    model_dir = paths.models_dir() / str(cfg["bdt"]["live_model"])
    if not (model_dir / "params.json").exists():
        pytest.skip("needs the trained live model (`make bdt-train`)")
    spec = load_feature_spec(cfg)
    wide = battledim.load_scada(2018).loc[QUIET_START : QUIET_START + pd.Timedelta(hours=14)]
    baselines = read_baselines(spark, paths.lake_baselines_dir()).cache()
    return {
        "spec": spec,
        "wide": wide,
        "baselines": baselines,
        "scorer": KMeansScorer(KMeansParams.load(model_dir)),
        "cfg": cfg,
    }


def features_of(spark, setup, wide: pd.DataFrame) -> pd.DataFrame:
    from leakwatch.features.windows import complete_window_range, select_windows, window_features

    spec = setup["spec"]
    t0, t1 = int(wide.index[0].timestamp()), int(wide.index[-1].timestamp())
    first, last = complete_window_range(t0, t1, spec)
    df = select_windows(
        window_features(spark_readings(spark, wide), setup["baselines"], spec), first, last
    )
    out = df.toPandas()
    out["window_start"] = pd.to_datetime(out["window_start"], utc=True)
    out["window_end"] = pd.to_datetime(out["window_end"], utc=True)
    return out.sort_values("window_start").reset_index(drop=True)


def with_leak(wide: pd.DataFrame, diameter_m: float) -> pd.DataFrame:
    from leakwatch.demo.signature import apply_signature, leak_signature

    delta, _ = leak_signature(NODE, diameter_m, LEAK_AT, hours=4)
    out = wide.copy()
    cols = list(out.columns)
    start = out.index.get_loc(LEAK_AT)
    values = out.to_numpy(dtype="float64")
    for i in range(start, len(out)):
        values[i] = apply_signature(values[i], cols, delta, i - start)
    return pd.DataFrame(values, index=out.index, columns=cols)


def test_shifting_time_by_whole_weeks_keeps_features(spark, setup):
    a = features_of(spark, setup, setup["wide"].iloc[:96])
    shifted = setup["wide"].iloc[:96].copy()
    shifted.index = shifted.index + pd.Timedelta(days=14)
    b = features_of(spark, setup, shifted)
    cols = setup["spec"].feature_columns()
    assert len(a) == len(b) > 0
    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-9, atol=1e-9)


def test_bigger_leak_never_lowers_the_score(spark, setup):
    scorer = setup["scorer"]
    after = lambda f: f[f["window_start"] >= LEAK_AT + pd.Timedelta(hours=1)]  # noqa: E731
    base = scorer.score_frame(after(features_of(spark, setup, setup["wide"])))[0]["score"]
    small = scorer.score_frame(after(features_of(spark, setup, with_leak(setup["wide"], 0.012))))[
        0
    ]["score"]
    large = scorer.score_frame(after(features_of(spark, setup, with_leak(setup["wide"], 0.030))))[
        0
    ]["score"]
    assert large.mean() >= small.mean() >= base.mean() * 0.98
    assert (large.to_numpy() >= small.to_numpy() - 1e-6).mean() > 0.9


def test_large_simulated_leak_raises_an_alert_quickly(spark, setup):
    """Simulator as oracle: a ~30 m3/h leak on a quiet day is flagged within 4 windows."""
    from leakwatch.bdt.alerts import AlertRule, run_alerts

    feats = features_of(spark, setup, with_leak(setup["wide"], 0.030))
    scorer = setup["scorer"]
    scored, shares = scorer.score_frame(feats)
    params = scorer.params
    rule = AlertRule(
        threshold=float(params.threshold),
        persistence_windows=int(params.persistence_windows),
        close_after_windows=4,
        top_n=3,
        series_area={s: "A" for s in scorer.series},
        origin="batch",
        model="test",
    )
    alerts = run_alerts(scored, shares, scorer.series, rule)
    fired = alerts[alerts["start_ts"] >= LEAK_AT]
    assert not fired.empty
    assert fired["start_ts"].min() <= LEAK_AT + pd.Timedelta(hours=1, minutes=15) + pd.Timedelta(
        minutes=15 * 4
    )
    quiet = alerts[alerts["start_ts"] < LEAK_AT]
    assert quiet.empty  # the quiet morning itself raises nothing


def test_appending_later_days_does_not_change_earlier_alerts(spark, setup):
    from leakwatch.bdt.alerts import AlertRule, run_alerts

    feats = features_of(spark, setup, with_leak(setup["wide"], 0.030))
    scorer = setup["scorer"]
    p = scorer.params
    rule = AlertRule(
        float(p.threshold),
        int(p.persistence_windows),
        4,
        3,
        {s: "A" for s in scorer.series},
        "batch",
        "test",
    )
    head = feats.iloc[: len(feats) // 2]
    a = run_alerts(*scorer.score_frame(head), scorer.series, rule)
    b = run_alerts(*scorer.score_frame(feats), scorer.series, rule)
    started_early = b[b["start_ts"] <= head["window_end"].max()]
    assert sorted(a["start_ts"]) == sorted(started_early["start_ts"])


def test_localizer_ranks_junctions_near_a_large_leak(spark, setup):
    """Simulator as oracle for the GNN: one of the top 5 lies within 500 m of the leak."""
    import networkx as nx

    from leakwatch.data import network
    from leakwatch.dl.gnn import make_localizer

    localize = make_localizer()
    if localize is None:
        pytest.skip("needs the trained localizer (`make gnn`)")
    feats = features_of(spark, setup, with_leak(setup["wide"], 0.030))
    alert_time = LEAK_AT + pd.Timedelta(hours=2)
    top = localize({"start_ts": alert_time}, feats)
    assert top is not None and len(top) == 5
    graph = network.pipe_graph(network.load_network())
    dist = nx.single_source_dijkstra_path_length(graph, NODE, weight="length")
    assert min(dist[c["node_id"]] for c in top) <= 500.0
