# Figure provenance

All commands below run from the repository root. Authoritative numerical
records live under `data/reference/`; routine renderings go to the ignored
`outputs/` tree. The exact selected publication PDFs are immutable files under
`figures/manuscript/` and are never overwritten by these commands.

The simplest audit is `python reproduce.py --fast`, which executes every
deterministic or cached redraw listed here and then validates the complete
artifact.

## Figure 1: tanh offset collision

```bash
python experiments/tanh_parameter_escape_loss.py --dtype float64
python experiments/tanh_parameter_escape_loss.py --dtype float32
python experiments/tanh_wb_trajectories.py
```

Reference data:
`data/reference/tanh/tanh_parameter_escape_loss.json` and
`data/reference/tanh/tanh_parameter_escape_loss_float32.json`.
Rendered trajectory plots are written to `outputs/tanh/`. Initialization is
explicit; `torch.manual_seed(0)` is recorded, but there is no random sampling
or minibatching.

## Figure 2: tanh slope collision

Full calculation:

```bash
python experiments/tanh_slope_collision_snapshots.py
```

Cached redraw:

```bash
python experiments/tanh_slope_collision_snapshots.py --plot-from-cache
```

The authoritative cache is
`data/reference/tanh/tanh_x_tanhprime_energy_minimization.json`; renderings go
to `outputs/tanh/`.

## Figure 3: Gaussian center collision

Full trajectory:

```bash
python experiments/gaussian_collision_mixed_precision.py \
  --orders 1 --float32-iter 100000000 --float64-iter 100000000 \
  --lr 1e-6 --q-min -16 --q-max 0.5 \
  --n-profile 6000 --n-quad 20001 --workers 1
```

Cached diagnostic redraw:

```bash
python experiments/gaussian_rbf_trace_diagnostics.py
```

The driver reads the history and deterministic NPZ profile cache from
`data/reference/gaussian/`. It compiles
`native/tanh_second_derivative_optimizer.c` into `outputs/.build/`; the
platform-specific library is never committed. Plots go to `outputs/gaussian/`.

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
  data/reference/gaussian/gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json
```

The full calculation updates the reference JSON under
`data/reference/gaussian/`; plots are written to `outputs/gaussian/`.

## Figures 5–6: one-dimensional derivative completion

```bash
python experiments/boundary_completion_experiment.py \
  --final-stage-optimizer lbfgs --output-suffix lbfgs_plateau
```

Cached redraw:

```bash
python experiments/boundary_completion_experiment.py \
  --output-suffix lbfgs_plateau \
  --plot-from-cache \
  data/reference/completion/boundary_completion_third_derivative_atoms_lbfgs_plateau.json
```

Matched endpoint audit:

```bash
python experiments/boundary_completion_matched_control_lbfgs.py
```

Reference histories remain under `data/reference/completion/`; plots go to
`outputs/completion/` and the matched rerun goes to
`outputs/matched-controls/`.

## Figure 7: directional two-dimensional completion

```bash
python experiments/boundary_completion_2d_experiment.py \
  --seed 80209 --output-suffix adam
```

Cached redraw and matched endpoint audit:

```bash
python experiments/boundary_completion_2d_experiment.py \
  --plot-from-cache \
  data/reference/completion/boundary_completion_2d_atoms_adam.json \
  --output-suffix adam
python experiments/boundary_completion_2d_matched_control_lbfgs.py
```

The full run writes its JSON to `data/reference/completion/` by default. Use
`--cache-output PATH` to select a different destination. Plot output and
matched reruns remain under `outputs/`.

## Figure 8: Ingham illustration

```bash
python experiments/ingham_illustration.py
```

Cached redraw:

```bash
python experiments/ingham_illustration.py --plot-from-cache
```

This is an exact finite-sum calculation on a fixed grid; it has no optimizer
or random input. Metadata are written to `data/reference/ingham/` and plots to
`outputs/ingham/`.

## Additional matched weak-form PDE comparison

The numerical section also reports a matched translated/completed coordinate
test that is not one of the eight numbered manuscript figures:

```bash
python experiments/matched_elliptic_completion.py
```

Cached redraw:

```bash
python experiments/matched_elliptic_completion.py --plot-from-cache
```

Its JSON contains optimizer traces, validation errors, normalized Gram
condition numbers, coefficient norms, and wall times. The CSV is a compact
summary. Both reference files are under `data/reference/weak_pde/`; plots go to
`outputs/weak_pde/`.

## Updating gallery previews

The `--sync-figures` flag on the tanh-slope, Gaussian-diagnostic, completion,
Ingham, and matched weak-PDE plotters is a maintainer operation. It copies PNG
previews into `figures/gallery/`; the matched weak-PDE driver also updates its
supplementary PDF. It never changes `figures/manuscript/`.
