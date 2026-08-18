# Experiment sources

This directory contains only executable experiment, plotting, and numerical
helper sources. The drivers stay flat because several of them import
neighboring modules and are designed to run directly from the repository root.

## Source groups

- Matched weak-form PDE coordinates: `matched_elliptic_completion.py`
- Tanh collisions: `tanh_parameter_escape_loss.py`,
  `tanh_wb_trajectories.py`, and `tanh_slope_collision_snapshots.py`
- Gaussian collision: `gaussian_collision_mixed_precision.py`,
  its neighboring tanh numerical helpers, and
  `gaussian_rbf_trace_diagnostics.py`
- Penalization: `gaussian_instability_adaptive_weight_penalty.py` and its two
  neighboring helper modules
- Derivative completion: the one- and two-dimensional
  `boundary_completion*` drivers
- Analytic illustration: `ingham_illustration.py`

Shared repository locations are defined in `artifact_paths.py`. Authoritative
records live under [`../data/reference/`](../data/reference/), native source
under [`../native/`](../native/), and generated plots and rerun results under
the ignored `../outputs/` directory.

The preferred interface is `python reproduce.py ...`. Exact standalone
commands are listed in
[`../docs/figure-provenance.md`](../docs/figure-provenance.md).
