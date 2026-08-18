# Contributing

Please keep changes reproducible and narrowly scoped.

1. Run `python reproduce.py --check` in the recorded environment.
2. Do not commit generated native libraries, caches unrelated to the paper,
   exploratory result trees, or machine-specific logs.
3. If an authoritative JSON/CSV/NPZ input changes, update `DATA_SHA256SUMS` and
   explain why the numerical record changed.
4. If a manuscript-facing PDF changes, update `FIGURE_SHA256SUMS` and document
   the plotting or presentation change.
5. Keep the full 100-million-step workflow out of routine CI.
