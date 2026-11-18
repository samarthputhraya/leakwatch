---
short: LeakWatch methodology
title: How LeakWatch detects and locates leaks
---

# How LeakWatch detects and locates leaks

This document explains, in plain words, how the system turns sensor readings into alerts, loss
estimates, candidate locations and work orders. Measured results (detection rate, false alarms per
week, localization accuracy) are not repeated here: they are in the files under `reports/metrics/`.

## 1. Data flow

Sensor readings (recorded BattLeDIM data or simulator output) are replayed into Kafka, the only
entry point for readings. Spark Structured Streaming writes them to a Parquet data lake, cleans
them and computes features. The detector scores each feature window and writes alerts to the lake.
The dashboard and the copilot read the lake with DuckDB, read-only.

## 2. Cleaning and features

- Readings arrive every 5 minutes. Short gaps of one or two missing readings are filled with the
  last value; a pressure sensor that repeats the same value for an hour is flagged as stuck.
  Flow, tank level and AMR meters are exempt from the stuck rule because constant values are
  normal for them (a pump that is off, a full tank).
- Features are computed for one-hour windows that slide every 15 minutes. For each pressure, flow
  and tank-level sensor, and for the summed AMR consumption of area C, each window has six
  statistics: mean, standard deviation, minimum, maximum, slope (change per hour) and residual.
- The residual is the window mean minus the hour-of-week baseline: the median value of that
  sensor at the same hour of the week in normal 2018 data. It removes the daily and weekly demand
  pattern, so a leak shows up as a lasting negative pressure residual and a positive inflow
  residual.

## 3. K-Means detector

The detector follows the BDT project title, "Applying K-Means Clustering for Anomaly Detection in
Water Resource Management".

- Training data: 2018 windows that contain no significant leak. Small, slowly growing background
  leaks are allowed (they are part of normal operation in the benchmark).
- Each feature vector is standardised, reduced with PCA (keeping the components that explain most
  of the variance) and clustered with K-Means in Spark MLlib. The number of clusters is chosen
  from the elbow of the within-cluster error, with the silhouette score reported alongside.
- Anomaly score: the distance from a window to the nearest cluster centre. Normal windows sit
  close to a centre; a leak moves the window away from all of them.
- Threshold: a high percentile of the scores of held-out normal 2018 windows. The year 2019 is
  never used to tune anything; it is the test year.

## 4. Alert rule

- A window is "above" when its score exceeds the threshold.
- An alert fires when a run of consecutive windows is above the threshold (the run length is
  tuned on 2018 so that false alarms per week stay low). The alert's start time is the end of the
  window that completed the run, the moment the system could have raised it.
- While the alert is open, every window above the threshold adds to it (highest score, sensor
  shares, loss estimate). The alert closes after a run of windows back below the threshold; its
  end time is the end of its last window above the threshold.

## 5. Explanation: top sensors and area

The distance of a window from its cluster centre is split into the share of each feature, and the
shares of a sensor's six statistics are added. The three sensors with the largest average share
over the alert's windows are its top sensors. The alert's area is the area of the top sensor.
Areas come from the network itself: A, B and C are the zones separated by the pump and the
pressure-reducing valves.

## 6. Estimated loss

The estimated loss rate is the mean excess inflow at the two reservoir inlets (flow sensors p227
and p235) above their hour-of-week baseline, over the alert's windows above the threshold, in
m3/h, and never below zero. It is an estimate: demand that differs from the usual week, and leaks
that already existed when the baseline was built, both change it. The water lost so far is this
rate times the time from when the alert fired to its end (or to its latest window if still open);
water lost before the alert fired is not counted.

## 7. Leak localization and the pipe list

- The K-Means detector says when something is wrong. The localizer (a graph neural network on the
  pipe network, the deep learning module) says where: it gives the five most likely nodes with a
  probability each, stored with the alert as its localizer candidates.
- A work order lists the pipes attached to those five nodes, most probable node first. A crew then
  walks a few hundred metres of pipe instead of the whole network.
- Until candidates exist for an alert, the work order lists the pipes nearest the alert's top
  sensors, measured along the pipes, starting with the top sensor. This fallback is less precise
  and is labelled as such in the work order.

## 8. Copilot and its guardrails

- The copilot answers through tools: SQL on the lake (read-only, one SELECT, capped rows, with a
  time limit), alert lookups, sensor charts, loss estimates, manual search and complaint matching.
- Every number in an answer must appear in that turn's tool results. If it does not, the answer
  is regenerated once with a correction; if it still fails, the copilot shows the tool results
  instead.
- Answers about procedures must cite a passage from the indexed manuals, or say that the manuals
  do not cover the question. Questions unrelated to the water network get a short refusal.
- Work orders are validated against a fixed schema. If the language model is unavailable, the
  same evidence is rendered with a template marked "template fallback".

## 9. Limits

- L-Town is a benchmark network, not Bengaluru: results show the method works on realistic
  sensor data, not how it would perform on a particular city.
- Slowly growing (incipient) leaks are hard for a window-based detector; expect them to be found
  late or missed. Results are reported separately for abrupt and incipient leaks.
- Area B has one pressure sensor, so leaks there are hard to see and harder to place.
