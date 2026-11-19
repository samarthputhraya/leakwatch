"""``make bdt-scaling``: the batch detection pipeline at 12.5M → 50M → 100M+ readings (PRD §7.5).

At each scale, one Spark job reads raw readings from the lake (BattLeDIM 2019; then both
BattLeDIM years plus the first simulated scenarios; then everything), runs the shared feature
function (``features.windows.window_features``: cleaning, 1 h windows every 15 min, 228 features
per (source, scenario, window)) and scores every window with the live K-Means model inside Spark
(``mapInPandas`` with the numpy scorer), writing scores to a scratch directory. Recorded per
scale: readings, windows, wall time, readings/s, and the peak resident memory of the whole
process tree (Python driver + Spark JVM + Python workers), sampled every second.

Writes ``reports/metrics/bdt_scaling.json`` and ``reports/figures/bdt_scaling.png``.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import threading
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from leakwatch.bdt.scorer import KMeansParams, KMeansScorer
from leakwatch.common import paths
from leakwatch.common.config import load_config
from leakwatch.common.logging import get_logger
from leakwatch.common.metrics import write_metrics
from leakwatch.features.baselines import read_baselines
from leakwatch.features.spec import load_feature_spec
from leakwatch.features.windows import window_features
from leakwatch.stream.spark import get_spark

log = get_logger("leakwatch.bdt.scaling")

SCORE_SCHEMA = StructType(
    [
        StructField("source", StringType()),
        StructField("scenario_id", StringType()),
        StructField("window_start", TimestampType()),
        StructField("score", DoubleType()),
        StructField("cluster", IntegerType()),
        StructField("above", BooleanType()),
    ]
)


class PeakMemory:
    """Samples the resident memory of this process and all its children (Spark JVM, workers)."""

    def __init__(self, every_s: float = 1.0) -> None:
        self.every_s = every_s
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        me = psutil.Process()
        while not self._stop.is_set():
            total = 0
            for p in [me, *me.children(recursive=True)]:
                with contextlib.suppress(psutil.Error):
                    total += p.memory_info().rss
            self.peak = max(self.peak, total)
            self._stop.wait(self.every_s)

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def partition_scorer(params: KMeansParams) -> Any:
    """A ``mapInPandas`` function: feature batches -> ``SCORE_SCHEMA`` rows (numpy scorer)."""

    def score_partitions(batches: Any) -> Any:
        scorer = KMeansScorer(params)
        for pdf in batches:
            if pdf.empty:
                continue
            scored, _ = scorer.score_frame(pdf)
            out = pdf[["source", "scenario_id", "window_start"]].reset_index(drop=True)
            out["score"] = scored["score"].to_numpy()
            out["cluster"] = scored["cluster"].astype("int32").to_numpy()
            out["above"] = scored["above"].to_numpy()
            yield out

    return score_partitions


def scale_filters(spark: Any) -> list[tuple[str, Any, int]]:
    """(label, raw-lake filter, readings) for the three scales, from the lake's actual content."""
    raw = spark.read.parquet(str(paths.lake_raw_dir()))
    counts = (
        raw.groupBy("source", "year")
        .count()
        .toPandas()
        .assign(year=lambda d: d["year"].astype(int))
    )
    sim_years = sorted(counts.loc[counts["source"] == "sim", "year"])
    bd = counts[counts["source"] == "battledim"].set_index("year")["count"]
    out = [
        (
            "BattLeDIM 2019",
            (F.col("source") == "battledim") & (F.col("year") == 2019),
            int(bd.get(2019, 0)),
        )
    ]
    # 50M: both BattLeDIM years + simulated years until ~50M
    total, years = int(bd.sum()), []
    sim = counts[counts["source"] == "sim"].set_index("year")["count"]
    for y in sim_years:
        if total >= 50_000_000:
            break
        years.append(y)
        total += int(sim[y])
    out.append(
        (
            "BattLeDIM 2018-2019 + simulated " + (f"{years[0]}-{years[-1]}" if years else "none"),
            (F.col("source") == "battledim")
            | ((F.col("source") == "sim") & F.col("year").isin(*years)),
            total,
        )
    )
    out.append(("all (BattLeDIM + all simulated)", F.lit(True), int(counts["count"].sum())))
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the three scales and write the benchmark."""
    parser = argparse.ArgumentParser(description="Batch pipeline scaling benchmark.")
    parser.add_argument("--master", default=None, help="override spark.master (default config)")
    args = parser.parse_args(argv)
    cfg = load_config()
    spec = load_feature_spec(cfg)
    model = str(cfg["bdt"]["live_model"])
    params = KMeansParams.load(paths.models_dir() / model)
    spark = get_spark(cfg["spark"], "leakwatch-bdt-scaling", master=args.master)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    baselines = read_baselines(spark, paths.lake_baselines_dir()).cache()
    out_root = paths.data_dir() / "bench" / "scaling"
    cols = list(params.feature_columns)
    score_partitions = partition_scorer(params)
    results = []
    for label, cond, readings in scale_filters(spark):
        shutil.rmtree(out_root, ignore_errors=True)
        raw = spark.read.parquet(str(paths.lake_raw_dir())).where(cond)
        log.info("scale %s: %d readings", label, readings)
        t0 = time.perf_counter()
        with PeakMemory() as mem:
            feats = window_features(raw, baselines, spec).select(
                "source", "scenario_id", "window_start", *cols
            )
            feats.mapInPandas(score_partitions, SCORE_SCHEMA).write.mode("overwrite").parquet(
                str(out_root)
            )
        wall = time.perf_counter() - t0
        scored = spark.read.parquet(str(out_root))
        stats = scored.agg(
            F.count(F.lit(1)).alias("n"), F.sum(F.col("above").cast("int")).alias("above")
        ).first()
        results.append(
            {
                "scale": label,
                "readings": readings,
                "windows_scored": int(stats["n"]),
                "windows_above_threshold": int(stats["above"] or 0),
                "wall_s": round(wall, 1),
                "readings_per_s": round(readings / wall, 1),
                "peak_rss_gb": round(mem.peak / 1e9, 2),
            }
        )
        log.info("%s", results[-1])
    shutil.rmtree(out_root, ignore_errors=True)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [r["readings"] / 1e6 for r in results]
    ax.plot(x, [r["wall_s"] / 60 for r in results], "o-", label="wall time (min)")
    ax.set_xlabel("readings in the job (millions)")
    ax.set_ylabel("wall time (minutes)")
    ax2 = ax.twinx()
    ax2.plot(
        x, [r["readings_per_s"] / 1e3 for r in results], "s--", color="#d62728", label="throughput"
    )
    ax2.set_ylabel("thousand readings per second", color="#d62728")
    master = args.master or cfg["spark"]["master"]
    ax.set_title(f"Batch features + K-Means scoring on one laptop ({master})")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "bdt_scaling.png", dpi=130)
    plt.close(fig)

    path = write_metrics(
        "bdt_scaling",
        {
            "pipeline": (
                "raw lake -> clean -> 1 h windows (228 features) -> K-Means score (mapInPandas)"
            ),
            "model": model,
            "spark": {k: cfg["spark"][k] for k in ("master", "driver_memory", "shuffle_partitions")}
            | ({"master": args.master} if args.master else {}),
            "machine": {
                "logical_cpus": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
            },
            "scales": results,
            "figure": "bdt_scaling.png",
        },
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
