# Tanh collisions and controlled precision

This guide covers Section 4.1 and Appendices C.1 and C.6. The Gaussian
coefficient-cap experiment in Section 4.2 is documented in
[the one-dimensional guide](manufactured_1d.md).
Run the commands below from the repository root. Fresh runs write to `outputs/`
and leave the publication records in `data/` unchanged.

| Paper item | Experiment | Recorded input | Source |
| --- | --- | --- | --- |
| Figure 4 | Fixed-slope tanh bias collision, binary32 and binary64 | `data/tanh_bias/` | `experiments/tanh/run_bias.py`, `plot_bias.py` |
| Figure 5 | Tanh slope collision, binary64 | `data/tanh_slope/`; preserved `figures/paper/figure_05.pdf` | `experiments/tanh/historical_slope_source.py`, `run_slope.py` |
| Appendix C.6 (text) | Gaussian evaluation sweep, derivative orders 1 and 3 | `data/precision/` | `experiments/precision/run.py`, `gaussian_gram.py` |

## Tanh experiments

### Figure 4: colliding biases

The model is `w1*tanh(x-b1) + w2*tanh(x-b2)` and the target is `sech(x)^2`.
Both precision runs start from weights `(3.5, 1.5)` and biases `(1, -1)`;
the slopes remain fixed at one. The bulk energy is

\[
J(u)=20\int_{-10}^{10}\left[\tfrac12(u'(x)^2+u(x)^2)-f(x)u(x)\right]dx,
\qquad f=-u_*''+u_*.
\]

The runs use 8,000 Adam updates with learning rate `1e-3` and 1,001 trapezoidal
nodes, followed by strong-Wolfe LBFGS with `max_iter=200` and 2,001 nodes.
The recorded LBFGS history includes closure evaluations, so its entries are not
counts of accepted optimizer steps. Binary32 and binary64 are independently
optimized; comparing these endpoints alone does not isolate arithmetic error.

The original source and histories come from repository commit
`8b15f8714af3f87abcde9fa484eddd7d66343f06`. The original vector figure is retained
as a composition input. The current paper puts single precision on the left
and double precision on the right. The compositor verifies the original panel
endpoints against the two histories, exchanges the vector regions and changes
the panel letters; it does not redraw or interpolate trajectories.

Recreate the exact vector composition:

```sh
python experiments/tanh/plot_bias.py --pdf-only
```

This writes `outputs/figures/figure_04.pdf`. Omit `--pdf-only` to also create a
600-dpi PNG preview; that option requires Ghostscript (`gs`). It needs no
training or Python packages beyond the standard library.

Run the recorded training protocols again:

```sh
python experiments/tanh/run_bias.py --dtype float64
python experiments/tanh/run_bias.py --dtype float32
```

The original recorded software versions were Python 3.13.9, NumPy 2.4.2,
PyTorch 2.10.0 and Matplotlib 3.10.8 on an Apple M2 CPU. Environment details
and original hashes are in `data/tanh_bias/`. Optimizer results can vary across
platforms and library versions.

### Figure 5: colliding slopes

The target is `x*sech(x)^2`, with the same bulk energy and its corresponding
forcing. All six parameters of
`w1*tanh(a1*x+b1) + w2*tanh(a2*x+b2)` are trained in binary64. Initial weights
are `(3.5, 1.5)`, slopes `(2, 0.5)` and biases `(0, 0)`. The protocol is 5,000
Adam steps at learning rate `1e-3` on 1,001 trapezoidal nodes. The four plotted
columns are Adam steps 1,666 and 5,000, followed by **independent** LBFGS
continuations with `max_iter=200` and `max_iter=1000`, each starting from the
same Adam endpoint and using 2,001 nodes.

The exact historical notebook cell is retained as
`experiments/tanh/historical_slope_source.py`. Its old label “1/5 Adam” refers
to the actual selection condition `it == 5000 // 3`, hence step 1,666.
`data/tanh_slope/rounded_snapshots.json` transcribes the saved six-decimal
parameter printouts, with corrected stage descriptions.

**Exact historical checkpoints and the full historical library environment
were not saved.** Figure 5 is therefore preserved as the original PDF, with
its SHA-256 recorded in `data/tanh_slope/provenance.json`. The rounded values
are not sufficient to reconstruct its exact curves. The repository does not
substitute snapshots from a later run for this publication image.

To rerun the historical algorithm in a separate directory:

```sh
python experiments/tanh/run_slope.py
```

This executes the unchanged historical source, writes a new PDF under
`outputs/tanh_slope/`, and additionally saves full-precision rerun snapshots.
These new optimizer results may differ from Figure 5. The later, unplotted
companion experiment is excluded from this repository release.

## Precision sweep

### Appendix C.6: precision without optimization

The complete sweep uses the central Gaussian difference stencils of orders
one and three, with gaps `h=2**(-k)` for `k=0,...,38`, in binary32 and binary64.
Raw signed summation and removable-singularity formulas evaluate the same
finite-gap state. Direct evaluation of the completed derivative is a separate
control representing the limiting state. No parameters are trained.

Reported errors are relative `H1(-12,12)` errors, computed by composite
trapezoidal quadrature on 1,025 dyadic nodes. References are independent
100-digit direct Gaussian sums, stored internally as binary64 high/low pairs;
selected cases are checked at 160 digits. Whole-line Gaussian Gram formulas
with 140-digit interval arithmetic audit the finite-gap model error.
`sweep.csv` also retains the residual error diagnostics from these same states.

The complete sweep has 468 rows: two derivative orders, two arithmetic
formats, 39 gaps and three evaluation methods. The paper reports these
results in prose, without a precision figure. Printed Figure 6 is the
[Gaussian coefficient-cap study](manufactured_1d.md).

Recompute the sweep:

```sh
python experiments/precision/run.py --output outputs/precision
```

Recomputation needs NumPy and mpmath and can take several minutes. It resumes
an existing output sweep. Use a new empty output directory for a fresh run.
`gaussian_gram.py` contains only the interval/Gram helper functions used by
this experiment; their numerical definitions are unchanged from the original
implementation. Source hashes, record hashes and the recorded execution
environment are retained in `data/precision/`.
