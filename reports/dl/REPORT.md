# LeakWatch DL module: LSTM autoencoder (when) and GNN localizer (where)

Draft generated from the metrics files.

## 1. LSTM autoencoder detector

- Input: 72-step (6 h) sequences of 37 standardized residual channels; trained on 8,848 normal 2018 sequences, 30 epochs (`reports/metrics/dl_lstm_ae.json` (code fa0f692)).
- τ = 1.0266 (p99 of threshold-day errors), m = 48; inputs re-baselined (minus each channel's median of the previous week, D-038).
- Official BattLeDIM score of its alerts (placed at the most-dropped pressure sensor): 2018 €61,916 (6 of 14 leaks, 22 false positives); 2019 €150,786 (13 of 23 leaks, 125 false positives).
- 2019: 19/19 leaks within 7 days (chance level 96.9%), 180 alerts, precision 0.391, 2.09 false alarms/week, windows above τ 73.6%.

**K-Means vs LSTM-AE at matched false-alarm rates.** Each detector is swept over the same threshold levels (quantiles of its own 2018 threshold-day scores, then multiples of their maximum) at its own 2018-tuned m; for each budget the table shows its best 2019 operating point with at most that many false alarms per week, next to the chance level of that many random alerts:

| detector | FA/week budget | detected ≤ 7 d | chance level | FA/week | precision | median TTD (h) |
|---|---|---|---|---|---|---|
| LSTM-AE | ≤ 0.5 | 52.6% | 57.7% | 0.38 | 0.500 | 16.9 |
| LSTM-AE | ≤ 1 | 63.2% | 70.0% | 0.59 | 0.456 | 17.3 |
| LSTM-AE | ≤ 2 | 100.0% | 95.2% | 1.78 | 0.408 | 14.8 |
| LSTM-AE | ≤ 4 | 100.0% | 95.2% | 1.78 | 0.408 | 14.8 |
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
| K-Means (re-baselined, combined index) | ≤ 0.5 | 47.4% | 56.1% | 0.40 | 0.364 | 69.0 |
| K-Means (re-baselined, combined index) | ≤ 1 | 68.4% | 65.5% | 0.63 | 0.411 | 40.8 |
| K-Means (re-baselined, combined index) | ≤ 2 | 100.0% | 93.5% | 1.71 | 0.382 | 26.0 |
| K-Means (re-baselined, combined index) | ≤ 4 | 100.0% | 93.5% | 1.71 | 0.382 | 26.0 |

Figure: `reports/figures/dl_detector_comparison_2019.png` (dotted lines: chance level).

## 2. GNN leak localizer

- Training data: 5,000 leak scenarios over 782 junctions (leak diameter 5–30 mm, ±10% demand noise, 2 cm sensor noise) (`reports/metrics/gnn_dataset.json` (code 758379a)).
- Each scenario runs on one of 120 perturbed networks (D-038): pipe roughness × U(1 ± 0.10), 0–6 small background leaks already running, whole-network demand × N(1, 0.05) per run; the reference is the median of 3 leak-free runs of the same network. A sensitivity matrix (one 15 mm leak per junction on the nominal model) gives each node the cosine between the observed residuals and its own signature.

Held-out simulated leaks: 667 test scenarios; 561 more with leak diameters 14–17 mm that were never trained on (`reports/metrics/dl_gnn.json` (code d5931bf)).

| model | inputs | top-1 | top-5 | top-5 medium + large | mean distance top-1 (m) | within 300 m (best of 5) | top-5 unseen sizes | training (min) |
|---|---|---|---|---|---|---|---|---|
| nearest_sensor | heuristic: pipe distance to the sensor with the largest pressure drop | 0.036 | 0.154 | 0.191 | 485 | 0.654 | 0.212 | — |
| sensitivity_corr | classic: cosine with each node's sensitivity signature (no learning) | 0.086 | 0.264 | 0.390 | 486 | 0.642 | 0.321 | — |
| mlp | sensor residuals only (no graph) | 0.033 | 0.145 | 0.202 | 313 | 0.894 | 0.189 | 18.4 |
| gcn | GCN, residuals at sensor nodes + closeness to each sensor + global sensor context | 0.076 | 0.307 | 0.393 | 262 | 0.844 | 0.344 | 98.0 |
| gcn_sens | GCN, residuals at sensor nodes + closeness to each sensor + global sensor context + sensitivity cosines | 0.066 | 0.252 | 0.330 | 324 | 0.799 | 0.291 | 77.4 |
| gat_sens | GAT, residuals at sensor nodes + closeness to each sensor + global sensor context + sensitivity cosines | 0.072 | 0.267 | 0.348 | 313 | 0.796 | 0.344 | 150.5 |
| gcn_sens_no_context | GCN, residuals at sensor nodes + closeness to each sensor + sensitivity cosines | 0.051 | 0.211 | 0.304 | 372 | 0.771 | 0.257 | 19.7 |

By leak size (buckets: `dl.gnn.size_buckets_m` in `configs/default.yaml`), test set:

| model | top-5 small | top-5 medium | top-5 large | within 300 m small (best of 5) | within 300 m medium | within 300 m large |
|---|---|---|---|---|---|---|
| nearest_sensor | 0.105 | 0.182 | 0.201 | 0.456 | 0.739 | 0.872 |
| sensitivity_corr | 0.095 | 0.291 | 0.503 | 0.368 | 0.768 | 0.933 |
| mlp | 0.070 | 0.187 | 0.218 | 0.789 | 0.956 | 0.989 |
| gcn | 0.193 | 0.389 | 0.397 | 0.688 | 0.946 | 0.978 |
| gcn_sens | 0.147 | 0.281 | 0.386 | 0.642 | 0.877 | 0.961 |
| gat_sens | 0.158 | 0.296 | 0.408 | 0.663 | 0.857 | 0.939 |
| gcn_sens_no_context | 0.088 | 0.246 | 0.369 | 0.593 | 0.872 | 0.939 |

Chosen (median top-1 distance on the 2018 BattLeDIM leaks (ties: validation top-5)): **gcn_sens_no_context**. GNN beats MLP on top-5: True.

**Target (PRD §10): top-5 ≥ 60% on held-out medium and large leaks, and better than the MLP.** Measured: 30.4% (gcn_sens_no_context) vs 20.2% (MLP): missed on the 60% level, better than the MLP.

**Ablation (sensitivity input, test top-5):** GCN 0.307; with the sensitivity cosines 0.252; with them but without the global context 0.211; the cosine alone, no learning 0.264 (D-038).

**Recorded BattLeDIM leaks (D-038).** Every leak is localized 2 h after its outflow first reaches half its peak, from re-baselined pressure residuals. 2018 (14 leaks) is the development set the live localizer is chosen on; 2019 (19 leaks) is only reported. Distance: top-1 node to the leaking pipe.

| localizer | 2018 median (m) | 2018 ≤ 300 m | 2018 ≤ 300 m (best of 5) | 2019 median (m) | 2019 ≤ 300 m | 2019 ≤ 300 m (best of 5) |
|---|---|---|---|---|---|---|
| nearest_sensor | 1,004 | 14.3% | 35.7% | 906 | 31.6% | 36.8% |
| sensitivity_corr | 1,052 | 28.6% | 28.6% | 1,300 | 26.3% | 31.6% |
| mlp | 674 | 21.4% | 35.7% | 852 | 26.3% | 36.8% |
| gcn | 622 | 35.7% | 35.7% | 1,360 | 21.1% | 26.3% |
| gcn_sens | 920 | 21.4% | 28.6% | 978 | 21.1% | 31.6% |
| gat_sens | 760 | 21.4% | 21.4% | 1,025 | 26.3% | 26.3% |
| gcn_sens_no_context | 458 | 28.6% | 42.9% | 1,025 | 15.8% | 26.3% |

**Posterior localizer (D-042).** The network's node probabilities times a prior exp(−d / L) around the pressure sensor that fell most (d: pipe distance). L is chosen on the official 2018 score of the live model's alerts, each reported at the posterior's top node; 2019 is shown for every L but never used (`reports/metrics/dl_gnn_prior.json` (code 63e95b6)):

| prior length L | 2018 (development) | 2019 (reported only) |
|---|---|---|
| network alone | €14,054 (5 TP) | €-3,394 (8 TP) |
| 1200 m | €30,419 (6 TP) | €66,859 (9 TP) |
| 600 m | €31,419 (6 TP) | €72,084 (10 TP) |
| 300 m | €73,937 (7 TP) | €160,577 (13 TP) |
| 150 m | €73,497 (7 TP) | €172,172 (13 TP) |
| 75 m | €65,298 (6 TP) | €183,236 (14 TP) |
| 37.5 m | €65,204 (6 TP) | €107,675 (12 TP) |

Chosen L = 300 m.

Recorded leaks, localized 2 h after the outflow first reaches half its peak:

| year | localizer | median (m) | ≤ 300 m | ≤ 300 m (best of 5) |
|---|---|---|---|---|
| 2018 | nearest_sensor | 1,004 | 14.3% | 35.7% |
| 2018 | network | 458 | 28.6% | 42.9% |
| 2018 | posterior | 631 | 21.4% | 28.6% |
| 2019 | nearest_sensor | 906 | 31.6% | 36.8% |
| 2019 | network | 1,025 | 15.8% | 26.3% |
| 2019 | posterior | 906 | 36.8% | 42.1% |

Gain of the posterior in leaks found within 300 m (top-1), paired bootstrap over leaks, 95% interval: 2018 vs nearest_sensor: 7.1% [0.0%, 21.4%]; 2018 vs network: -7.1% [-21.4%, 0.0%]; 2019 vs nearest_sensor: 5.3% [0.0%, 15.8%]; 2019 vs network: 21.1% [5.3%, 42.1%].

## 3. Model cards

**LSTM-AE** — data: BattLeDIM 2018 normal periods (D-015); intended use: anomaly score per 1 h window alongside K-Means; limits: trained on one year of one benchmark network; drift in accumulated leakage raises scores like K-Means.

**GNN localizer** — data: EPANET leak scenarios on perturbed copies of L-Town (roughness, background leaks, day-to-day demand; D-024, D-038); inputs: two hourly re-baselined pressure residuals at the 33 sensor nodes, each junction's closeness (pipe distance) to every sensor, the cosine with its sensitivity signature, and a global context vector computed from all sensor residuals (D-029); output: probability per junction; limits: one new leak at a time, junction leaks (pipe leaks are scored by the nearer pipe end); the recorded-leak table above shows how it transfers to the BattLeDIM data.
