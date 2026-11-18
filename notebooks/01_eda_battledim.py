# %% [markdown]
# # BattLeDIM L-Town: exploratory data analysis
#
# Phase 0 EDA (PRD §12): what the 2018 (training) and 2019 (evaluation) SCADA years contain,
# where the sensors and leaks are, how leaks show up in the readings, and data quality.
# Every number below is computed from the files built by `make data`; the headline numbers are
# also saved in `reports/metrics/data_summary.json` and `reports/metrics/network_stats.json`.
# Run headless with `make eda`; figures go to `reports/figures/`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from leakwatch.common.metrics import read_metrics
from leakwatch.common.paths import FIGURES_DIR, ensure_dir
from leakwatch.common.schemas import split_sensor_column
from leakwatch.data import battledim, ground_truth, network, quality

ensure_dir(FIGURES_DIR)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
AREA_COLORS = {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"}
TYPE_COLORS = {"abrupt": "#D62728", "incipient": "#FF9F1C"}

scada = {y: battledim.load_scada(y) for y in battledim.YEARS}
leak_flows = {y: battledim.load_leak_flows(y) for y in battledim.YEARS}
gt = ground_truth.load_ground_truth()
wn = network.load_network()
areas = network.node_areas(wn)
pipe_areas = network.pipe_areas(wn, areas)
sensors = network.parse_sensor_config()
summary = read_metrics("data_summary")
stats = read_metrics("network_stats")

# %% [markdown]
# ## 1. What is in the data

# %%
rows = []
for year, wide in scada.items():
    types = pd.Series([split_sensor_column(c)[0] for c in wide.columns], index=wide.columns)
    for sensor_type, cols in types.groupby(types):
        block = wide[cols.index]
        rows.append(
            {
                "year": year,
                "sensor_type": sensor_type,
                "sensors": block.shape[1],
                "readings": int(block.notna().sum().sum()),
                "missing": int(block.isna().sum().sum()),
            }
        )
overview = pd.DataFrame(rows)
print("readings per year:", overview.groupby("year")["readings"].sum().to_dict())
overview

# %% [markdown]
# ## 2. Network, areas, sensors and leak locations

# %%
coords = network.node_coordinates(wn)
xy = {n: (float(r.x), float(r.y)) for n, r in coords.iterrows()}


def pipe_midpoint(pipe_id: str) -> tuple[float, float]:
    link = wn.get_link(pipe_id)
    (x1, y1), (x2, y2) = xy[link.start_node_name], xy[link.end_node_name]
    return (x1 + x2) / 2, (y1 + y2) / 2


fig, ax = plt.subplots(figsize=(10, 9))
for area, color in AREA_COLORS.items():
    segments = [
        [xy[wn.get_link(p).start_node_name], xy[wn.get_link(p).end_node_name]]
        for p in pipe_areas.index[pipe_areas.eq(area)]
    ]
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.9, label=f"Area {area}"))
amr = coords.loc[sensors["amr"]]
ax.scatter(amr.x, amr.y, s=8, c="#B279A2", label=f"AMR ({len(amr)})", zorder=2)
pressure = coords.loc[sensors["pressure"]]
ax.scatter(
    pressure.x,
    pressure.y,
    s=40,
    c="black",
    marker="^",
    label=f"pressure sensor ({len(pressure)})",
    zorder=4,
)
for year, marker in ((2018, "o"), (2019, "X")):
    pts = np.array([pipe_midpoint(p) for p in gt.loc[gt.year_started == year, "pipe_id"]])
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=70,
        marker=marker,
        facecolors="none" if marker == "o" else "#D62728",
        edgecolors="#D62728",
        label=f"leaks started {year} ({len(pts)})",
        zorder=5,
    )
ax.autoscale()
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
ax.set_title("L-Town: areas, sensors and the 33 BattLeDIM leaks")
fig.savefig(FIGURES_DIR / "eda_network_map.png", bbox_inches="tight")
pd.DataFrame(stats["areas"]).T[
    ["junctions", "n_pressure_sensors", "amrs", "pipes", "pipe_length_km"]
]

# %%
leak_areas = gt.assign(area=gt["pipe_id"].map(pipe_areas))
pd.crosstab(leak_areas["year_started"], leak_areas["area"])

# %% [markdown]
# ## 3. Pressure in a leak-free week (2018-01-01 to 2018-01-07)
# The first 2018 leak starts on 8 January, so this week has no leak.

# %%
week = scada[2018].loc["2018-01-01":"2018-01-07"]
picks = {
    "A": ["pressure__n54", "pressure__n105", "pressure__n415", "pressure__n740"],
    "B": ["pressure__n215"],
    "C": ["pressure__n1", "pressure__n4", "pressure__n31"],
}
fig, axes = plt.subplots(len(picks), 1, figsize=(12, 8), sharex=True)
for ax, (area, cols) in zip(axes, picks.items(), strict=True):
    for col in cols:
        ax.plot(week.index, week[col], lw=0.8, label=col.split("__")[1])
    ax.set_ylabel("pressure (m)")
    ax.set_title(f"Area {area}", loc="left", fontsize=10)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_pressure_week.png", bbox_inches="tight")

# %% [markdown]
# ## 4. Inflow with leak bands
# Inflow = the two reservoir inlet meters (p227 + p235). The lower panel shows each leak's active
# period; the dashed line is the total leak outflow published as ground truth.

# %%
for year, wide in scada.items():
    inflow = (wide["flow__p227"] + wide["flow__p235"]).resample("1D").mean()
    leak_total = leak_flows[year].sum(axis=1).resample("1D").mean()
    active = ground_truth.active_in_year(gt, year).sort_values("start_ts")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]}
    )
    ax1.plot(inflow.index, inflow, color="#4C78A8", label="daily mean inflow (m³/h)")
    ax1.plot(
        leak_total.index,
        leak_total,
        color="#D62728",
        ls="--",
        label="daily mean total leak outflow (m³/h)",
    )
    ax1.set_ylabel("m³/h")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"{year}: inflow and leaks ({len(active)} leaks active during the year)")
    year_end = wide.index[-1]
    for i, leak in enumerate(active.itertuples()):
        start = max(leak.start_ts, wide.index[0])
        end = leak.end_ts if pd.notna(leak.end_ts) else year_end
        end = min(end, year_end)
        ax2.barh(i, end - start, left=start, color=TYPE_COLORS[leak.type], height=0.7)
        ax2.text(start, i, f" {leak.pipe_id}", va="center", fontsize=6)
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.legend(
        handles=[Patch(color=c, label=t) for t, c in TYPE_COLORS.items()],
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"eda_inflow_leak_bands_{year}.png", bbox_inches="tight")

# %% [markdown]
# ## 5. How much of each year is leak-free?
# PRD §7.3 trains K-Means on "2018 windows that do not overlap any known leak". Several 2018 leaks
# are never repaired, so the truly leak-free part of 2018 is short. This matters for Phases 2–3.

# %%
pd.DataFrame(summary["leak_activity"]).T

# %%
fig, ax = plt.subplots(figsize=(12, 3))
for year, flows in leak_flows.items():
    n_active = flows.gt(0).sum(axis=1).resample("1D").max()
    ax.step(n_active.index.dayofyear, n_active, where="post", label=str(year))
ax.set_xlabel("day of year")
ax.set_ylabel("leaks flowing")
ax.legend()
ax.set_title("Number of leaks flowing at the same time (daily max)")
fig.savefig(FIGURES_DIR / "eda_simultaneous_leaks.png", bbox_inches="tight")

# %% [markdown]
# ## 6. Weekly pattern: why baselines are per hour-of-week
# Median and 10–90% band of inflow for each hour of the week, 2018.

# %%
inflow18 = scada[2018]["flow__p227"] + scada[2018]["flow__p235"]
how = inflow18.index.dayofweek * 24 + inflow18.index.hour
profile = inflow18.groupby(how).quantile([0.1, 0.5, 0.9]).unstack()
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.fill_between(profile.index, profile[0.1], profile[0.9], alpha=0.25, label="10–90%")
ax.plot(profile.index, profile[0.5], label="median")
ax.set_xticks(range(0, 168, 24), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_ylabel("inflow (m³/h)")
ax.legend()
ax.set_title("2018 inflow by hour of week")
fig.savefig(FIGURES_DIR / "eda_hour_of_week_inflow.png", bbox_inches="tight")

# %% [markdown]
# ## 7. One abrupt 2019 leak, seen by the sensors
# Signal = reading minus the reading at the same time one week earlier (2,016 steps), which removes
# most of the weekly demand pattern. Shown ±3 days around the first abrupt 2019 leak.

# %%
leak = gt[(gt.year_started == 2019) & (gt.type == "abrupt")].iloc[0]
lo, hi = leak.start_ts - pd.Timedelta(days=3), leak.start_ts + pd.Timedelta(days=3)
week_diff = scada[2019].diff(2016).loc[lo:hi]
pressure_cols = [c for c in week_diff.columns if c.startswith("pressure__")]
after = week_diff.loc[leak.start_ts : leak.start_ts + pd.Timedelta(hours=12), pressure_cols].mean()
top3 = after.sort_values().index[:3]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(week_diff.index, week_diff["flow__p227"] + week_diff["flow__p235"], color="#4C78A8")
ax1.set_ylabel("Δ inflow vs last week (m³/h)")
for col in top3:
    ax2.plot(week_diff.index, week_diff[col], lw=0.8, label=col.split("__")[1])
ax2.set_ylabel("Δ pressure vs last week (m)")
ax2.legend(fontsize=8)
for ax in (ax1, ax2):
    ax.axvline(leak.start_ts, color="#D62728", ls="--")
area = pipe_areas[leak.pipe_id]
ax1.set_title(
    f"Leak {leak.pipe_id} (area {area}, abrupt, peak {leak.peak_outflow_m3h:.1f} m³/h) "
    f"starts {leak.start_ts:%Y-%m-%d %H:%M}"
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "eda_abrupt_leak_example.png", bbox_inches="tight")
print("pressure sensors with the largest mean drop in the 12 h after the start:")
after.sort_values().head(3).round(3)

# %% [markdown]
# ## 8. Missing data and stuck sensors
# PRD §7.2 flags a sensor as stuck when it repeats the same value for 12 or more steps (1 hour).

# %%
pd.json_normalize(
    [
        {"year": s["year"], "sensor_type": t, **v}
        for s in summary["scada"]
        for t, v in s["by_type"].items()
    ]
)

# %%
stuck18 = quality.stuck_summary(scada[2018], min_steps=12)
stuck18["sensor_type"] = [split_sensor_column(c)[0] for c in stuck18.index]
stuck18.sort_values("stuck_steps", ascending=False).head(10)

# %% [markdown]
# ## 9. Findings carried into later phases
# The cell below prints each finding from the computed values.

# %%
la18 = summary["leak_activity"]["2018"]
equal = summary["share_of_readings_equal_across_years"]["2018_vs_2019"]
missing = sum(s["missing"] for s in summary["scada"])
findings = [
    f"Missing values in 2018+2019 SCADA: {missing}. Gap filling (PRD §7.2) is still built for "
    "streamed/simulated data but has nothing to fill in the benchmark files.",
    f"Only {la18['share_of_steps_with_no_leak_flowing']:.1%} of 2018 time steps have no leak "
    f"flowing ({la18['steps_with_no_leak_flowing']} steps); up to "
    f"{la18['max_simultaneous_leaks']} leaks flow at once. Training on strictly leak-free 2018 "
    "windows leaves very little data: Phase 2/3 must define 'normal' differently (logged in "
    "DECISIONS.md when chosen).",
    f"Leaks are {la18['leak_share_of_inflow']:.1%} of 2018 inflow volume "
    f"(mean inflow {la18['mean_inflow_m3h']} m³/h).",
    f"AMR readings: {equal['amr']:.1%} of 2019 values equal 2018 at the same position "
    f"(pressure: {equal['pressure']:.1%}), so AMR consumption is largely replayed between years.",
    f"Network: {stats['counts']['junctions']} junctions, {stats['counts']['pipes']} pipes, "
    f"{stats['total_pipe_length_km']} km of pipe, mean pipe {stats['pipe_length_m']['mean']} m.",
    f"Top-5 search area: {stats['search_area']['pipe_length_to_inspect_m']} m of pipe, "
    f"{stats['search_area']['reduction_factor_vs_whole_network']}x less than the whole network.",
]
for f in findings:
    print("-", f)
