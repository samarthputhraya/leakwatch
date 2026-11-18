# Decision log

Every non-obvious technical decision, newest at the bottom. Format: date · decision · reason · alternatives considered.

---

### D-001 · 2026-11-18 · Run everything inside WSL2 Ubuntu 26.04
- **Decision:** All build, data and service commands run in WSL2 (`Ubuntu`, kernel 2, systemd enabled). Windows-side Python 3.12 and Java 25 are not used.
- **Reason:** PRD §0 ("If the OS is Windows, use WSL2"). Spark, Kafka and Docker are far less painful on Linux.
- **Alternatives:** native Windows (needs winutils for Spark, Docker Desktop with admin rights) — rejected.

### D-002 · 2026-11-18 · Source in the OneDrive workspace, venv and data on the WSL disk
- **Decision:** The git repo is the VS Code workspace folder (`…/OneDrive/Documents/mega project`, seen from WSL as `/mnt/c/...`). The Python venv lives at `/root/.venvs/leakwatch` (`UV_PROJECT_ENVIRONMENT`) and data lives at `/root/leakwatch-data` (`LEAKWATCH_DATA_DIR` in `.env`). If `LEAKWATCH_DATA_DIR` is unset, code falls back to `<repo>/data/` (gitignored) as the PRD layout says.
- **Reason:** (1) the team edits in VS Code on Windows, so source must stay visible there; (2) a 100M-reading Parquet lake and a venv with tens of thousands of files must not be synced by OneDrive; (3) file I/O across the `/mnt/c` 9P bridge is several times slower than the WSL ext4 disk, which matters for Spark and Parquet.
- **Risks:** the repo path contains a space (`mega project`) — the Makefile uses only relative paths and quotes; data paths have no spaces. OneDrive can briefly lock files during sync; if git ever reports lock errors, pause OneDrive sync or move the repo to WSL (`~/leakwatch`) and open it with VS Code Remote-WSL.
- **Alternatives:** whole repo in WSL (`~/leakwatch`) + VS Code Remote-WSL — the best-performing option and a good later move; not chosen now because the team's workspace is already the OneDrive folder.

### D-003 · 2026-11-18 · Toolchain installed with apt + uv inside WSL (no Windows admin needed)
- **Decision:** `openjdk-17-jdk-headless`, `docker.io`, `docker-compose-v2`, `make`, `build-essential` from Ubuntu apt; `uv` from the official Astral installer; Python 3.11 via `uv python install 3.11` (python-build-standalone).
- **Reason:** Ubuntu 26.04 ships Python 3.14 only; PySpark 3.5.x is safest on Python 3.11 (the PRD's choice). Docker Engine inside WSL avoids Docker Desktop (Windows admin rights + licence terms). Java 17 is the newest LTS that Spark 3.5 supports.
- **Alternatives:** deadsnakes PPA (no 26.04 builds), Docker Desktop (admin install) — rejected.

### D-004 · 2026-11-18 · Dependencies pinned with uv (pyproject `==` pins + `uv.lock`), installed per phase
- **Decision:** Direct dependencies use exact `==` pins in `pyproject.toml`; `uv.lock` pins the full transitive tree. Heavy stacks are optional extras added when their phase starts: `spark` (Phase 1), `dl` (Phase 5), `copilot` (Phase 6), `dashboard` (Phase 7).
- **Reason:** PRD §0 (pin everything) and laptop constraints: Phase 0 should not download ~2 GB of torch/CUDA wheels it does not use.

### D-005 · 2026-11-18 · BattLeDIM loaded straight from Zenodo; `water-benchmark-hub` not used
- **What we verified (PRD §4.1 step 1–3):**
  - `water-benchmark-hub` 0.5.3 registers the benchmark under the key `"BattLeDIM"` (`load("BattLeDIM")`).
  - **Bug found:** `BattLeDIM.load_data(return_test_scenario=True)` downloads `2018_SCADA.xlsx` and `False` downloads `2019_SCADA.xlsx` (inverted), while its labels use the correct year's leak config. Using it would silently pair 2018 readings with 2019 labels. `epyt_flow.data.benchmarks.battledim.load_data` (0.17.3) has the same inversion.
  - The leak configurations in both packages are byte-identical to each other and match the official `dataset_configuration.yaml` on Zenodo.
  - Installing `water-benchmark-hub` pulls HuggingFace `datasets` (needs pyarrow ≥ 24) plus ~10 more packages we do not use.
- **Decision:** `make data` downloads the original files from Zenodo record 4017659 (doi:10.5281/zenodo.4017659) and verifies each file's MD5 against the Zenodo API. We use the per-type CSVs (not the 92 MB xlsx files). Ground truth comes from the official `dataset_configuration.yaml` and is cross-checked in tests against the epyt-flow leak config and `2018_Fixed_Leakages_Report.txt`; peak outflows come from `2018_Leakages.csv` / `2019_Leakages.csv`. epyt-flow (already needed for simulation) also provides the official BattLeDIM scoring routine for Phase 3.

### D-006 · 2026-11-18 · numpy 1.26.4 + pandas 2.2.3 + WNTR 1.3.2 + pyarrow 25.0.0 (+ setuptools 80.9.0, six)
- **Problem:** PySpark 3.5.9 (`ml`, `mllib`, `sql` extras) requires `numpy<2`. WNTR ≥ 1.4.0 requires `numpy>=2.2.6`. pyarrow 26 refuses numpy 1.x at import time (not declared in its metadata). Spark's Python workers will run the simulator (`mapPartitions`), so both must live in one environment.
- **Decision:** pin `numpy==1.26.4`, `pandas==2.2.3` (well tested with PySpark 3.5), `wntr==1.3.2` (newest release with `numpy<2`, Mar 2025), `pyarrow==25.0.0` (newest that imports with numpy 1.26). WNTR 1.3.2 imports `pkg_resources` and `six` without declaring them, so we also pin `setuptools==80.9.0` (last line that ships `pkg_resources`) and `six==1.17.0`.
- **Alternatives:** Spark 4.x (supports numpy 2) — rejected because the PRD fixes Spark 3.5.x; separate venvs for simulation and Spark — rejected (Spark workers need the simulator in-process).
- **Revisit if:** PySpark's Arrow path fails with pyarrow 25 in Phase 1 → downgrade pyarrow (no other package constrains it now).

### D-007 · 2026-11-18 · Scaling data from epyt-flow; GNN training scenarios from WNTR
- **Measured** (`reports/metrics/sim_benchmark.json` at commit 22ba83c; one 15 mm abrupt leak on p461, 5-minute steps, one process each): WNTR WNTRSimulator (PDD, split-pipe leak) ran 48 h in 22.3 s and 7 days in 74.0 s (3,077–3,243 equivalent readings/s, peak 410–756 MB). epyt-flow (EPANET) ran 48 h in 15.1 s and 7 days in 51.4 s (4,535–4,666 readings/s, peak 262–358 MB). "Equivalent readings" = time steps × 119 BattLeDIM sensors. An earlier run of the same code on the same machine measured epyt-flow at 13.0 s / 45.4 s, so expect ±15% run-to-run variance; the ranking was the same both times.
- **Decision:** bulk simulation for the 100M-reading scaling story (Phase 4) uses **epyt-flow**; single-leak scenarios for GNN training (Phase 5) use **WNTR** as PRD §4.2 describes (PDD, leak on a junction inserted mid-pipe).
- **Reason:** epyt-flow is ~1.4–1.7× faster and uses about half the memory per worker. With WSL capped at 7 GB, memory per Spark worker limits parallelism more than CPU does (18 logical cores). Projected single-core time for 100M readings: ~6.0 h (epyt-flow) vs ~8.6 h (WNTR); both divide by the number of parallel workers.
- **Sanity check:** WNTR's 15 mm leak reached 15.1 m³/h, consistent with the benchmark's own leaks (p461: 21.3 mm, peak 30.5 m³/h; outflow scales with diameter²).
- **Note:** neither engine is "too slow" outright; the choice is about memory headroom. Revisit in Phase 4 with the real Spark `mapPartitions` throughput.

### D-008 · 2026-11-18 · Network areas = zones between the pump and PRVs
- **Decision:** area A/B/C of each node is computed (not hard-coded per node): cut L-Town at PUMP_1 and PRV-1/2/3, take connected zones, name each by an anchor node in config (`A: n54`, `B: n215`, `C: T1`); reservoir stubs inherit the zone their PRV feeds.
- **Measured** (`reports/metrics/network_stats.json`): A has 659 junctions (657 in the main zone plus the reservoir inlet nodes n303 and n336) and 29 pressure sensors, B has 31 junctions and 1 sensor (n215), C has 92 junctions, tank T1, 3 sensors and all 82 AMRs; 659 + 31 + 92 = 782. The PRD quotes 655 junctions for A (from a paper); we report our measured numbers.
- **Also measured:** total pipe length 43.16 km (PRD quotes ~42.6 km); mean pipe length 47.7 m.

### D-009 · 2026-11-18 · Notebooks are percent-format `.py` files
- **Decision:** notebooks are committed as `# %%` scripts (`notebooks/01_eda_battledim.py`); `make eda` builds and executes a real `.ipynb` into `notebooks/executed/` (gitignored) with nbclient, and saves figures to `reports/figures/`.
- **Reason:** readable diffs and code review, no output blobs in git, VS Code runs `# %%` cells natively. No extra dependency (jupytext) needed.

### D-010 · 2026-11-18 · K-Means "normal" = background leakage allowed; strict leak-free run as ablation (confirmed by the team)
- **Problem:** PRD §7.3 trains K-Means on 2018 windows that overlap no known leak. Measured (`data_summary.json`): only 2.2% of 2018 steps (2,310, about 8 days in early January) have no leak flowing; 2019 has none (mean 10.3 leaks flowing at once).
- **Decision (confirmed by the user on 2026-11-18):** the main BDT model treats small, long-running leaks as background leakage, as utilities do, and excludes from training only windows near leak onsets and during abrupt leaks (exact exclusion rule set and logged in Phase 3). A second model trained on the strict ~8 leak-free days is reported alongside as an ablation.
- **Impact:** the BDT report must state how "normal" was defined and show both results.

### D-011 · 2026-11-18 · Raw lake = readings as received; stateful cleaning moves to the cleaned layer (Phase 2)
- **Decision:** the Phase 1 stream parses, validates (contract), deduplicates and *flags* out-of-range values, then writes `lake/raw`. Forward-filling gaps of ≤ 2 steps and stuck-sensor flags (PRD §7.2) are applied when building the cleaned/feature layer in Phase 2, using `leakwatch.data.quality` as the reference implementation and the batch/stream parity test.
- **Reason:** forward-filling creates readings that never arrived and stuck flags depend on per-sensor history across micro-batches; neither belongs in a "raw" layer, and both need the same state handling as the windowed features. Keeping raw = as received also makes the lake auditable against the source (contract tests compare every value).
- **Also:** messages that break the `scada.readings` contract are not dropped silently; they go to `lake/_rejects/raw`.

### D-012 · 2026-11-18 · Kafka: `apache/kafka:4.3.2`, single KRaft node, no volume, `restart: unless-stopped`
- **Image:** the official Apache image (no Bitnami/Confluent licensing or availability changes). 4.3.2 is the latest patch of the previous minor line (4.4.0 was 4 weeks old). Spark's connector ships kafka-clients 3.4.x, which Kafka 4.x brokers accept (they require clients ≥ 2.1).
- **No volume:** the image runs as a non-root user and a named volume would need ownership fixes; more importantly Kafka is only the transport — the lake is the source of truth. `make down` drops queued messages; `make lake-reset` rebuilds consistently.
- **Restart policy:** WSL stops the distro (and Docker with it) when no Windows-launched process is attached — observed during development: Kafka exited with 143 between two `wsl` calls. `restart: unless-stopped` brings the same container (and its messages) back when the distro restarts. Running commands from an open WSL terminal avoids the stop entirely.

### D-013 · 2026-11-18 · Exactly-once into the lake: watermark dedup + `foreachBatch` commit protocol
- **Dedup:** `withWatermark("ts", "1 hour").dropDuplicatesWithinWatermark(source, scenario_id, sensor_type, sensor_id, ts)`. The producer is idempotent, so Kafka itself holds no retry duplicates; dedup covers a publisher that re-sends recent data (e.g. a restarted replayer). Data older than the watermark is dropped as late and counted (`late_rows_dropped_by_watermark`). Consequence: the stream must see event time move forward — replaying an *older* year into the same checkpoint drops it as late. Rebuild with `make lake-reset` instead.
- **Sink:** Spark's `foreachBatch` is at-least-once. Each micro-batch is staged, then its files are moved into the lake with a `b<batch_id>-` prefix, and a commit marker is written last; a re-run of a committed batch is skipped, and a re-run of a half-moved batch first deletes its old files (`stream/lake_writer.py`). Verified by unit tests and an end-to-end test with an injected duplicate, a late copy and a malformed message.
- **Alternatives:** Spark's file sink is exactly-once but hides committed files behind `_spark_metadata`, which DuckDB does not read — readers could see files from failed batches.

### D-014 · 2026-11-18 · Phase 1 library versions
- `pyspark==3.5.9` + Maven `spark-sql-kafka-0-10_2.12:3.5.9` (must match), `confluent-kafka==2.16.2` (librdkafka, fast C producer; pure-Python clients are slower), `orjson==3.13.0` (JSON encoding in the replayer hot loop), `duckdb==1.5.6` (last 1.x; 2.0.0 had just been released).
- Spark writes Parquet timestamps as `TIMESTAMP_MICROS` (UTC-adjusted) instead of the legacy INT96 default, so DuckDB and pyarrow read them as proper UTC timestamps.

### D-015 · 2026-11-18 · Exact rule for "normal" timestamps (baselines now, K-Means training in Phase 3)
- **Principle (D-010, user-confirmed):** small long-running leaks are background leakage and count as normal; time near leak onsets and during abrupt leaks does not.
- **Rule (per leak in the ground truth, `leakwatch.features.normal`):**
  - *background leak* = incipient **and** peak outflow ≤ `normal.background_max_peak_m3h` (10 m³/h): only its onset `[start, start + 7 days]` is excluded (`onset_exclusion_days`, = the 7-day detection horizon of PRD §7.4), capped at its repair;
  - every other leak (*significant*: abrupt, or peak > 10 m³/h, or peak unknown) is excluded from start to repair, or to the end of the data if never repaired;
  - a timestamp is normal iff it is in none of these intervals (bounds inclusive). A 1 h window is normal iff all its steps are (Phase 3).
- **Why 10 m³/h:** 2018 peak outflows split cleanly: 5.11, 5.49, 6.87, 6.91 m³/h (p427, p654, p257, p810 — exactly the four leaks never repaired, which carry into 2019) and then 16.21 … 35.38 m³/h. Any threshold in (6.91, 16.21) gives the same 2018 result. Abrupt leaks are always significant, also the three small abrupt 2019 leaks (5.3–5.6 m³/h).
- **Effect:** measured in `reports/metrics/baselines_hour_of_week.json` (normal days, stretches). The strict rule (every leak excluded while active, the D-010 ablation) is the same function with `rule="strict"`.
- **Known limitation:** the excluded periods are uneven over the year (most of Feb–Mar and Jul–Aug are excluded), so hour-of-week baselines lean on spring/autumn/winter demand. Hour-of-week medians ignore season by design (PRD §7.2).

### D-016 · 2026-11-18 · Cleaning semantics: causal stuck flag, interior forward fill, per-type exemptions
- **Stuck:** a reading is flagged once it and the previous 11 steps (`stuck_min_steps` = 12) were all *observed* with the same value. Causal (the first 11 steps of a run are not flagged, because a streaming job cannot know yet that the run will get long), a missing step breaks a run, filled readings are never flagged. Reference: `data.quality.stuck_flags`; the Spark version is tested against it.
- **Exempt types:** `cleaning.stuck_exempt_types: [flow, level, amr]`. Measured in Phase 0: PUMP_1 sits at exactly 0 or 44.07 m³/h for up to 233 steps (pump state), tank T1 holds its level for up to 83 steps (0.01 m rounding), AMR n25 repeats one value for up to 109 steps. For these types a constant value is normal operation, so the flag would be almost always wrong. Pressure keeps the rule; the PRV-regulated n215 (area B) will be flagged often, which is accurate (Finding 3).
- **Forward fill:** only *interior* gaps of 1–2 missing steps are filled (`filled = true`), once the reading after the gap has arrived. The pandas reference also fills a short gap at the very end of a frame; a stream cannot know a trailing gap is short, so we do not (the only difference, documented in `features.clean`).
- **Out of range:** flags from Phase 1 are carried; flagged values are left out of feature statistics and AMR sums (`features.exclude_out_of_range: true`). No BattLeDIM reading is out of range.
- Flags are data-quality columns, not model inputs. Each feature row carries counts (`n_filled`, `n_stuck`, `n_out_of_range`, `n_series_missing`) for Phase 3 alert explanations.

### D-017 · 2026-11-18 · Cleaned layer, baselines and feature-table layout
- **`lake/clean`** (not listed in PRD §6.2): cleaned readings with `filled`/`stuck` flags, same partitions as `lake/raw`, built in batch by `make features`. It feeds the baselines and makes the flags queryable (DuckDB view `clean_readings`). ~same size as raw.
- **Series:** one per pressure/flow/level sensor (37) plus the summed AMR consumption per area (PRD §7.2). All 82 AMRs are in area C, so there is one AMR series, `amr_sum__C` (L/h): 38 series.
- **Baselines** (`lake/baselines/hour_of_week`): exact median (`F.median`) per series and hour of week (0 = Monday 00:00 UTC) over the normal 2018 timestamps of D-015. 38 × 168 rows.
- **Statistics per series and window:** mean, population std, min, max, least-squares slope against time (units per hour), mean residual (value − baseline). 38 × 6 = 228 feature columns named `<series>__<stat>` (`pressure__n54__slope`), plus window keys and quality counts. One wide row per window (the "pivot" of PRD §7.2), built with conditional aggregation so the column set and order are fixed by `FeatureSpec` even when a series is missing.
- **Windows:** `F.window` 1 h, sliding 15 min, aligned to the epoch. A window belongs to the date/year of its start. The batch job keeps only windows fully covered by data: 35,040 for 2018 (the last three need 2019 readings) and 35,037 for 2019 (its last three would need 2020).
- **Same function:** `features.windows.window_features(readings, baselines, spec)` (cleaning included) is what the batch job applies to raw lake rows and what the streaming job applies to each micro-batch plus carried history. Batch reads the raw layer (not `lake/clean`) so both paths make literally the same call.

### D-018 · 2026-11-18 · Streaming features: foreachBatch with carried history, separate live table
- **Problem:** cleaning and 1 h windows need readings from earlier micro-batches. Spark's own stateful operators could do the windows, but not the forward fill or the stuck rule without a second, different implementation (`applyInPandasWithState`) — the opposite of batch/stream parity. Streaming `pivot` is also unsupported.
- **Decision:** the stream (Kafka → the raw ingest's parse + watermark dedup) hands each micro-batch to `stream.features_stream.FeatureBatchWriter`, which unions it with a carried *tail* (readings from `next window start − 55 min`: 11 steps of stuck history ≥ 3 steps of fill history), calls the batch function, and emits each window once, in order, as soon as data `features.stream.lateness_minutes` (15) past its last step has arrived. 15 min ≥ (2-step gap + closing reading), so the stream never misses a fill the batch makes. Feature files, tail and state are committed together with the lake_writer protocol (D-013): a retried batch restarts from the previous committed state.
- **Output:** `lake/features_stream/` with the batch schema, kept apart from `lake/features` so the training/evaluation tables stay immutable while the live path runs (online vs offline store). `make features-parity` compares the two window by window.
- **Consequences:** a window is available ~15 simulated minutes after it ends (2.5 s at the 360× demo speed); readings older than the next window to emit are counted as late and ignored. Event time is global across sources/scenarios, like the raw watermark (D-013); Phase 7's simulated leaks must keep event time moving forward.
- **Numerical fix found by the full-year parity run (2026-11-18):** the first run (`reports/metrics/features_parity_battledim_run1_epoch_time.json`, code 5a5afce) matched all 70,076 windows, but 1,539 of 16.3 M cells differed beyond rtol 1e-9, all in `amr_sum__C__slope` (max |diff| 3.4e-6). The slope used hours since the epoch (~4.3e5) as time; Spark's covariance then loses ~1e-10 of each time step, and the large AMR sum (~3e4 L/h) turns that into ~1e-6 slope noise that depends on aggregation order. Time is now measured from each window's start (0 … 11/12 h); a unit test checks the slope against numpy to 1e-9 on large values at 2029 timestamps.
