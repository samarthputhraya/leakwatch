# LeakWatch

Software-only leak detection for water distribution networks, built and evaluated on the open
[BattLeDIM 2020](https://github.com/KIOS-Research/BattLeDIM) benchmark network L-Town.

- **When** is something wrong? Streaming pipeline (Kafka → Spark) with K-Means anomaly detection.
  BDT title: *Applying K-Means Clustering for Anomaly Detection in Water Resource Management.*
- **Where** is the leak? A graph neural network ranks the 5 most likely leak locations.
- **What do we do?** An operator copilot turns alerts into work orders and answers questions with data.

Status: **Phase 0 (environment and data)**. See [PROGRESS.md](PROGRESS.md) for what works today and
[docs/PRD.md](docs/PRD.md) for the full specification. Results are reported only from
`reports/metrics/*.json`; none are published yet.

## Quick start (Windows + WSL2 Ubuntu)

From a PowerShell terminal in the repo folder:

```powershell
wsl make setup   # Python 3.11 venv via uv (pinned)
wsl make data    # download BattLeDIM from Zenodo, convert to Parquet, build the leak ground truth
wsl make test    # unit + contract tests
```

One-time toolchain inside WSL (already done on the development machine): `apt install openjdk-17-jdk-headless
make docker.io docker-compose-v2 build-essential`, then install `uv` from https://docs.astral.sh/uv/.
Copy `.env.example` to `.env` and set `LEAKWATCH_DATA_DIR` to a folder on the WSL disk.

## Data

BattLeDIM dataset: S. G. Vrachimis et al., Zenodo, doi:10.5281/zenodo.4017659. L-Town is a benchmark network,
not Bengaluru; no real utility data is used.
