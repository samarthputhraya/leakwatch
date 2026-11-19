# LeakWatch DL module: LSTM autoencoder (when) and GNN localizer (where)

Draft generated from the metrics files.

## 1. LSTM autoencoder detector

- Input: 72-step (6 h) sequences of 37 standardized residual channels; trained on 8,848 normal 2018 sequences, 30 epochs (`reports/metrics/dl_lstm_ae.json` (code 8aa05d5)).
- τ = 0.6906 (p99 of threshold-day errors), m = 3.
- 2019: 7/19 leaks within 7 days (chance level 76.3%), 75 alerts, precision 0.589, 0.44 false alarms/week, windows above τ 89.9%.

**K-Means vs LSTM-AE at matched false-alarm rates.** Each detector is swept over the same threshold levels (quantiles of its own 2018 threshold-day scores, then multiples of their maximum) at its own 2018-tuned m; for each budget the table shows its best 2019 operating point with at most that many false alarms per week, next to the chance level of that many random alerts:

| detector | FA/week budget | detected ≤ 7 d | chance level | FA/week | precision | median TTD (h) |
|---|---|---|---|---|---|---|
| LSTM-AE | ≤ 0.5 | 36.8% | 77.2% | 0.44 | 0.589 | 4.1 |
| LSTM-AE | ≤ 1 | 42.1% | 80.4% | 0.65 | 0.528 | 4.8 |
| LSTM-AE | ≤ 2 | 79.0% | 95.2% | 1.71 | 0.437 | 23.4 |
| LSTM-AE | ≤ 4 | 94.7% | 96.7% | 2.15 | 0.391 | 25.4 |
| K-Means (all PRD features) | ≤ 0.5 | 0.0% | 0.0% | 0.00 | n/a | n/a |
| K-Means (all PRD features) | ≤ 1 | 0.0% | 0.0% | 0.00 | n/a | n/a |
| K-Means (all PRD features) | ≤ 2 | 42.1% | 91.5% | 1.90 | 0.238 | 23.3 |
| K-Means (all PRD features) | ≤ 4 | 100.0% | 99.6% | 2.90 | 0.437 | 18.1 |
| K-Means (residual features) | ≤ 0.5 | 36.8% | 82.5% | 0.48 | 0.390 | 56.8 |
| K-Means (residual features) | ≤ 1 | 36.8% | 82.5% | 0.48 | 0.390 | 56.8 |
| K-Means (residual features) | ≤ 2 | 68.4% | 98.8% | 1.76 | 0.533 | 5.0 |
| K-Means (residual features) | ≤ 4 | 94.7% | 99.1% | 2.57 | 0.449 | 18.1 |
| K-Means (residual, combined index) | ≤ 0.5 | 15.8% | 71.8% | 0.29 | 0.250 | 82.9 |
| K-Means (residual, combined index) | ≤ 1 | 36.8% | 90.5% | 0.86 | 0.489 | 2.4 |
| K-Means (residual, combined index) | ≤ 2 | 36.8% | 90.5% | 0.86 | 0.489 | 2.4 |
| K-Means (residual, combined index) | ≤ 4 | 94.7% | 99.0% | 2.63 | 0.434 | 17.8 |

Figure: `reports/figures/dl_detector_comparison_2019.png` (dotted lines: chance level).

## 2. GNN leak localizer

- Training data: 5,000 single-leak scenarios over 782 junctions (leak diameter 5–30 mm, ±10% demand noise, 2 cm sensor noise) (`reports/metrics/gnn_dataset.json` (code f3bbb4f)).

Held-out simulated leaks: 669 test scenarios; 549 more with leak diameters 14–17 mm that were never trained on (`reports/metrics/dl_gnn.json` (code 909b99f)).

| model | inputs | top-1 | top-5 | top-5 medium + large | mean distance top-1 (m) | within 300 m (best of 5) | top-5 unseen sizes | training (min) |
|---|---|---|---|---|---|---|---|---|
| nearest_sensor | heuristic: pipe distance to the sensor with the largest pressure drop | 0.036 | 0.190 | 0.202 | 280 | 0.806 | 0.193 | — |
| mlp | sensor residuals only (no graph) | 0.058 | 0.229 | 0.281 | 232 | 0.957 | 0.288 | 1.3 |
| gcn | GCN, residuals at sensor nodes + closeness to each sensor + global sensor context | 0.076 | 0.309 | 0.386 | 219 | 0.888 | 0.444 | 31.2 |
| gat | GAT, residuals at sensor nodes + closeness to each sensor + global sensor context | 0.073 | 0.339 | 0.402 | 206 | 0.915 | 0.448 | 253.4 |
| gcn_no_context | GCN, residuals at sensor nodes + closeness to each sensor | 0.030 | 0.102 | 0.115 | 557 | 0.646 | 0.069 | 7.0 |
| gcn_no_closeness | GCN, residuals at sensor nodes + global sensor context | 0.033 | 0.176 | 0.179 | 308 | 0.846 | 0.206 | 4.7 |
| gcn_local_only | GCN, residuals at sensor nodes | 0.030 | 0.097 | 0.115 | 586 | 0.652 | 0.064 | 8.6 |

By leak size (buckets: `dl.gnn.size_buckets_m` in `configs/default.yaml`), test set:

| model | top-5 small | top-5 medium | top-5 large | within 300 m small (best of 5) | within 300 m medium | within 300 m large |
|---|---|---|---|---|---|---|
| nearest_sensor | 0.173 | 0.196 | 0.209 | 0.719 | 0.877 | 0.855 |
| mlp | 0.155 | 0.242 | 0.331 | 0.910 | 0.982 | 1.000 |
| gcn | 0.201 | 0.384 | 0.390 | 0.766 | 0.968 | 0.983 |
| gat | 0.252 | 0.343 | 0.477 | 0.838 | 0.954 | 0.988 |
| gcn_no_context | 0.083 | 0.069 | 0.174 | 0.547 | 0.566 | 0.907 |
| gcn_no_closeness | 0.173 | 0.192 | 0.163 | 0.770 | 0.904 | 0.895 |
| gcn_local_only | 0.072 | 0.064 | 0.180 | 0.540 | 0.598 | 0.901 |

Chosen by validation top-5: **gat**. GNN beats MLP on top-5: True.

**Target (PRD §10): top-5 ≥ 60% on held-out medium and large leaks, and better than the MLP.** Measured: 40.2% (gat) vs 28.1% (MLP): missed on the 60% level, better than the MLP.

**Ablation (GCN, test top-5):** full model 0.309; without the global sensor context 0.102 (-20.8 points); without sensor closeness 0.176 (-13.3 points); without both (the first design: each junction only sees sensors a few hops away) 0.097 (D-029).

**BattLeDIM 2019 benchmark (gat):** 7 leaks with a batch alert of the live detector starting within 7 days after the leak; localized at the first such alert. Median distance from the top-1 node to the leaking pipe 1,682 m (nearest of top 5: 1,647 m); nearest-sensor heuristic top-1: 750 m; within 300 m (best of top 5): 28.6%. On 2019 the trained localizer is worse than the nearest-sensor heuristic: it was trained on one leak at a time on the nominal model, while 2019 has many leaks running at once and a real network that differs from the model.

| leak | type | peak outflow (m³/h) | alert start (UTC) | top-1 distance (m) | best of top 5 (m) | nearest sensor top-1 (m) |
|---|---|---|---|---|---|---|
| L2019_p523 | abrupt | 28.4 | 2019-01-15 23:45 | 300 | 225 | 63 |
| L2019_p280 | abrupt | 5.3 | 2019-02-10 16:30 | 1,008 | 682 | 750 |
| L2019_p653 | incipient | 18.3 | 2019-03-03 16:15 | 1,682 | 1,647 | 1,648 |
| L2019_p710 | abrupt | 5.6 | 2019-03-25 04:45 | 376 | 269 | 113 |
| L2019_p514 | abrupt | 15.6 | 2019-04-03 04:00 | 2,535 | 2,309 | 51 |
| L2019_p193 | incipient | 10.4 | 2019-05-20 04:30 | 2,781 | 2,691 | 2,176 |
| L2019_p277 | incipient | 7.4 | 2019-05-31 06:00 | 2,165 | 2,145 | 2,198 |

## 3. Model cards

**LSTM-AE** — data: BattLeDIM 2018 normal periods (D-015); intended use: anomaly score per 1 h window alongside K-Means; limits: trained on one year of one benchmark network; drift in accumulated leakage raises scores like K-Means.

**GNN localizer** — data: EPANET single-leak scenarios on nominal L-Town (D-024); inputs: two hourly pressure residuals at the 33 sensor nodes, each junction's closeness (pipe distance) to every sensor, and a global context vector computed from all sensor residuals (D-029); output: probability per junction; limits: one leak at a time, nominal model (the real network differs by about ±10%), junction leaks (pipe leaks are scored by the nearer pipe end); see the 2019 benchmark above for how it transfers to the recorded data.
