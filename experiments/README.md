# Experiment sources and authoritative caches

This directory intentionally keeps the standalone drivers and their imported
helpers together. It contains only the sources and cached results needed for
the paper's numerical section.

## Source groups

- Matched weak-PDE coordinates: `matched_elliptic_completion.py`
- Tanh collisions: `tanh_parameter_escape_loss.py`,
  `tanh_wb_trajectories.py`, `tanh_slope_collision_snapshots.py`
- Gaussian collision: `gaussian_rbf_instability_figure5_mixed_precision.py`,
  `tanh_instability_figure5_mixed_precision.py`,
  `tanh_instability_figure5.py`, `tanh_second_derivative_optimizer.c`, and
  `gaussian_rbf_trace_diagnostics.py`
- Penalization: `gaussian_instability_adaptive_weight_penalty.py` with its two
  neighboring helper modules
- Derivative completion: the one- and two-dimensional
  `boundary_completion*` scripts
- Analytic illustration: `ingham_illustration.py`
- Orchestration and validation: `reproduce.py`

JSON files are the authoritative histories or analytic metadata. The single
NPZ file is a deterministic Gaussian profile cache. Generated native libraries
and Python bytecode are ignored by Git.

Run commands from the repository root so relative output and figure paths stay
consistent. The preferred interface is `python reproduce.py ...`; exact
standalone commands are listed in `../docs/figure-provenance.md`.
