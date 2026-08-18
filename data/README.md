# Reference data

This directory contains the small, authoritative numerical records used to
validate and redraw the article figures. Files are grouped by scientific
mechanism rather than by file format:

- `reference/tanh/` — offset- and slope-collision traces;
- `reference/gaussian/` — center-collision, penalization, and profile caches;
- `reference/completion/` — one- and two-dimensional completion histories and
  matched controls;
- `reference/weak_pde/` — the matched weak-form coordinate comparison; and
- `reference/ingham/` — metadata for the deterministic Ingham illustration.

Every reference file is covered by [`DATA_SHA256SUMS`](../DATA_SHA256SUMS).
Cached validation and plotting require no external data download. Generated
figures and rerun controls are written to the ignored `outputs/` directory.

JSON files contain the complete machine-readable histories or analytic
metadata. The CSV is a compact endpoint summary, and the NPZ is a deterministic
Gaussian profile cache used by the long mixed-precision calculation.
