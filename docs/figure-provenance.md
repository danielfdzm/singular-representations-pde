# Figure provenance

All commands below are run from the repository root. Standalone scripts render
audit outputs beside the scripts in `experiments/` unless `--sync-figures` is
shown. Curated named outputs are in `figures/descriptive/`, and the exact final
numbered PDFs are in `figures/manuscript/`. The repository-level fast workflow
audits all cached plots while preserving the committed publication layouts.

## Figure 1: tanh offset collision

```bash
python experiments/tanh_parameter_escape_loss.py --dtype float64
python experiments/tanh_parameter_escape_loss.py --dtype float32
python experiments/tanh_wb_trajectories.py
```

Inputs/outputs: `tanh_parameter_escape_loss.json`,
`tanh_parameter_escape_loss_float32.json`, and `wb_trajectories.pdf`.
Initialization is explicit; `torch.manual_seed(0)` is recorded, but there is no
random sampling or minibatching.

## Figure 2: tanh slope collision

```bash
python experiments/tanh_slope_collision_snapshots.py --sync-figures
```

Cached redraw:

```bash
python experiments/tanh_slope_collision_snapshots.py \
  --plot-from-cache --sync-figures
```

Authoritative cache: `tanh_x_tanhprime_energy_minimization.json`.

## Figure 3: Gaussian center collision

```bash
python experiments/gaussian_rbf_instability_figure5_mixed_precision.py \
  --orders 1 --float32-iter 100000000 --float64-iter 100000000 \
  --lr 1e-6 --q-min -16 --q-max 0.5 \
  --n-profile 6000 --n-quad 20001 --workers 1

python experiments/gaussian_rbf_trace_diagnostics.py \
  --input experiments/gaussian_rbf_instability_unpenalized_mixed_precision.json \
  --output-directory experiments --sync-figures
```

The experiment uses the deterministic NPZ profile cache and compiles
`tanh_second_derivative_optimizer.c` into `experiments/.build/`. The native
library is never committed.

## Figure 4: adaptive Gaussian weight penalty

```bash
python experiments/gaussian_instability_adaptive_weight_penalty.py \
  --start-exp 5 --end-exp 12 --max-iter 1000000 \
  --lr 0.5 --plateau-window 50000 --plateau-rtol 1e-3
```

Cached redraw:

```bash
python experiments/gaussian_instability_adaptive_weight_penalty.py \
  --plot-from-json \
  experiments/gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json
```

## Figures 5--6: one-dimensional derivative completion

```bash
python experiments/boundary_completion_experiment.py \
  --final-stage-optimizer lbfgs --output-suffix lbfgs_plateau
```

Cached redraw:

```bash
python experiments/boundary_completion_experiment.py \
  --output-suffix lbfgs_plateau \
  --plot-from-cache \
  experiments/boundary_completion_third_derivative_atoms_lbfgs_plateau.json
```

Matched endpoint audit:

```bash
python experiments/boundary_completion_matched_control_lbfgs.py
```

## Figure 7: directional two-dimensional completion

```bash
python experiments/boundary_completion_2d_experiment.py \
  --seed 80209 --output-suffix adam
```

Cached redraw and matched endpoint audit:

```bash
python experiments/boundary_completion_2d_experiment.py \
  --plot-from-cache --output-directory experiments --output-suffix adam
python experiments/boundary_completion_2d_matched_control_lbfgs.py
```

## Figure 8: Ingham illustration

```bash
python experiments/ingham_illustration.py --sync-figures
```

This is an exact finite-sum calculation on a fixed grid; it has no optimizer
or random input.

## Additional matched weak-PDE comparison

The numerical section also reports a matched translated/completed coordinate
test that is not one of the eight numbered figures:

```bash
python experiments/matched_elliptic_completion.py --sync-figures
```

Its JSON contains the optimizer traces, validation errors, normalized Gram
condition numbers, coefficient norms, and wall times. The CSV is a compact
summary of the same endpoint comparison.
