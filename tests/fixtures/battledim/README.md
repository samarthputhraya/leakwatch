# BattLeDIM test fixtures

- `dataset_configuration.yaml`, `2018_Fixed_Leakages_Report.txt`: copied byte-for-byte from the
  BattLeDIM dataset (S. G. Vrachimis, D. G. Eliades, R. Taormina, A. Ostfeld, Z. Kapelan, S. Liu,
  M. S. Kyriakou, P. Pavlou, M. Qiu, M. Polycarpou), Zenodo record 4017659,
  doi:10.5281/zenodo.4017659, licensed CC-BY-4.0. Committed so the ground-truth tests run in CI
  without downloading data. MD5s match `leakwatch.data.battledim.FILES`.
- `sample_pressures.csv`: four hand-written rows in the BattLeDIM CSV format (`;` separator,
  `,` decimal) for parser tests. Not real data.
