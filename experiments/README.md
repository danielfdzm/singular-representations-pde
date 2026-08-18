# Experiments and figure provenance

This directory contains the calculations used in the numerical section of
the paper. You do not need to run the long optimizations to inspect the
results: the saved numerical histories are in [`../data/`](../data/), the
eight publication figures are in [`../figures/paper/`](../figures/paper/),
and readable PNG previews are in
[`../figures/previews/`](../figures/previews/).

For a first reproduction, return to the repository root and run:

```bash
python reproduce.py --fast
```

This redraws every cache-backed figure under `outputs/`. The individual
commands below are useful when studying one experiment at a time. All of them
are run from the repository root.

> **Before rerunning an optimization:** commands labelled “Repeat” regenerate
> the corresponding stored record under `data/`. Use a clean clone if you want
> to compare a new run with the archived values.

## At a glance

| Figure | Mathematical point | Main saved record |
|---|---|---|
| [Fig. 1](../figures/paper/Fig1.pdf) | Tanh bias collision and parameter escape | [`data/tanh/`](../data/tanh/) |
| [Fig. 2](../figures/paper/Fig2.pdf) | Fixed-architecture tanh slope collision | [`data/tanh/`](../data/tanh/) |
| [Fig. 3](../figures/paper/Fig3.pdf) | Gaussian collision, Gramian degeneration, and precision loss | [`data/gaussian/`](../data/gaussian/) |
| [Fig. 4](../figures/paper/Fig4.pdf) | Stabilization by an adaptive weight penalty | [`data/gaussian/`](../data/gaussian/) |
| [Figs. 5–6](../figures/paper/Fig5.pdf) | One-dimensional derivative completion | [`data/completion/`](../data/completion/) |
| [Fig. 7](../figures/paper/Fig7.pdf) | Directional two-dimensional completion | [`data/completion/`](../data/completion/) |
| [Fig. 8](../figures/paper/Fig8.pdf) | Ingham lower-bound illustration | [`data/ingham/`](../data/ingham/) |
| [Additional comparison](../figures/previews/matched_elliptic_completion.png) | Translated versus completed weak-form coordinates | [`data/weak_pde/`](../data/weak_pde/) |

## Figure 1 — tanh bias collision

The two precision-dependent optimization histories are combined into the
parameter-plane figure by `tanh_wb_trajectories.py`.

Redraw from the saved histories:

```bash
python experiments/tanh_wb_trajectories.py
```

Repeat the two trajectories and then redraw:

```bash
python experiments/tanh_parameter_escape_loss.py --dtype float64
python experiments/tanh_parameter_escape_loss.py --dtype float32
python experiments/tanh_wb_trajectories.py
```

Saved files:
[`tanh_parameter_escape_loss.json`](../data/tanh/tanh_parameter_escape_loss.json)
and
[`tanh_parameter_escape_loss_float32.json`](../data/tanh/tanh_parameter_escape_loss_float32.json).
New plots are written to `outputs/tanh/`.

## Figure 2 — tanh slope collision

Redraw from the saved snapshots:

```bash
python experiments/tanh_slope_collision_snapshots.py --plot-from-cache
```

Repeat the calculation:

```bash
python experiments/tanh_slope_collision_snapshots.py
```

The saved snapshots are in
[`tanh_x_tanhprime_energy_minimization.json`](../data/tanh/tanh_x_tanhprime_energy_minimization.json),
and new plots are written to `outputs/tanh/`.

## Figure 3 — Gaussian center collision

Redraw the four diagnostics from the saved mixed-precision trace:

```bash
python experiments/gaussian_rbf_trace_diagnostics.py
```

Repeat the full trajectory used in the paper:

```bash
python experiments/gaussian_collision_mixed_precision.py \
  --orders 1 --float32-iter 100000000 --float64-iter 100000000 \
  --lr 1e-6 --q-min -16 --q-max 0.5 \
  --n-profile 6000 --n-quad 20001 --workers 1
python experiments/gaussian_rbf_trace_diagnostics.py
```

This is one of the very long runs. It uses the portable C source
[`../native/tanh_second_derivative_optimizer.c`](../native/tanh_second_derivative_optimizer.c).
The trace and profile cache are in [`../data/gaussian/`](../data/gaussian/),
and plots are written to `outputs/gaussian/`.

## Figure 4 — adaptive Gaussian weight penalty

Redraw from the saved continuation history:

```bash
python experiments/gaussian_instability_adaptive_weight_penalty.py \
  --plot-from-json \
  data/gaussian/gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json
```

Repeat the continuation:

```bash
python experiments/gaussian_instability_adaptive_weight_penalty.py \
  --start-exp 5 --end-exp 12 --max-iter 1000000 \
  --lr 0.5 --plateau-window 50000 --plateau-rtol 1e-3
```

The saved history is
[`gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json`](../data/gaussian/gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json),
and plots are written to `outputs/gaussian/`.

## Figures 5–6 — one-dimensional derivative completion

Redraw the collision tree and representation evolution:

```bash
python experiments/boundary_completion_experiment.py \
  --output-suffix lbfgs_plateau \
  --plot-from-cache \
  data/completion/boundary_completion_third_derivative_atoms_lbfgs_plateau.json
```

Repeat the completed-coordinate calculation:

```bash
python experiments/boundary_completion_experiment.py \
  --final-stage-optimizer lbfgs --output-suffix lbfgs_plateau
```

Repeat only the matched translated-coordinate endpoint control:

```bash
python experiments/boundary_completion_matched_control_lbfgs.py
```

The saved completion and control records are in
[`../data/completion/`](../data/completion/). New figures are written to
`outputs/completion/`, and the endpoint audit is written to
`outputs/matched-controls/`.

## Figure 7 — directional two-dimensional completion

Redraw from the saved trajectory:

```bash
python experiments/boundary_completion_2d_experiment.py \
  --plot-from-cache data/completion/boundary_completion_2d_atoms_adam.json \
  --output-directory outputs/completion --output-suffix adam
```

Repeat the completed-coordinate calculation:

```bash
python experiments/boundary_completion_2d_experiment.py \
  --seed 80209 --output-directory outputs/completion --output-suffix adam
```

Repeat only the matched translated-coordinate endpoint control:

```bash
python experiments/boundary_completion_2d_matched_control_lbfgs.py
```

The saved trajectory and control are in
[`../data/completion/`](../data/completion/). New figures are written to
`outputs/completion/`, and the endpoint audit is written to
`outputs/matched-controls/`.

## Figure 8 — Ingham illustration

Redraw from the saved grid metadata:

```bash
python experiments/ingham_illustration.py --plot-from-cache
```

Repeat the finite-sum calculation:

```bash
python experiments/ingham_illustration.py
```

This calculation has no optimizer or random input. Its record is
[`ingham_panel_T_2_5_10_an1_styled.json`](../data/ingham/ingham_panel_T_2_5_10_an1_styled.json),
and plots are written to `outputs/ingham/`.

## Additional weak-form elliptic comparison

Redraw the translated/completed comparison from the saved histories:

```bash
python experiments/matched_elliptic_completion.py --plot-from-cache
```

Repeat both linear optimization runs:

```bash
python experiments/matched_elliptic_completion.py
```

The JSON trace and CSV summary are in
[`../data/weak_pde/`](../data/weak_pde/), and plots are written to
`outputs/weak_pde/`.

## A note on the full runs

Several full calculations are deliberately long. For the environment,
deterministic-grid details, seeds, output policy, and integrity check, see
[`../REPRODUCING.md`](../REPRODUCING.md). The publication PDFs in
`figures/paper/` are reference files and are not overwritten by the commands
above.
