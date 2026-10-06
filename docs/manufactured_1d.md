# Manufactured whole-line PDE: Figure 6

This study connects Section 4.2 and Appendix C.2 to the whole-line Sobolev setting of the theory. It uses `K(x)=exp(-x²/2)`, two multiplicity slots, and the Helmholtz equation `-g''+g=f` on the real line. The shifted quadratic energy is `J_g(v)=0.5*||v-g||²_H1(R)`.

Set `q=K'''/6+7K'/20` and `g_epsilon=K'-epsilon*q`. The prescribed forcing is obtained analytically as `f_epsilon=g_epsilon-g_epsilon''`. The symmetric-family sweep records `epsilon=0` and `epsilon=0.01`; the general pair search includes `epsilon=0.001` as well. These are the three target perturbations in the paper. The completed reference `u0=K'` uses a two-slot derivative block with coefficients `(0,1)`.

## What is computed

A finite symmetric pair is

```text
v_(b,h)(x) = b * [K(x+h)-K(x-h)] / (2h),
h = b/(sqrt(2)*R).
```

Its ordinary coefficient norm equals the imposed cap `R`; its stored derivative-coordinate coefficient is `b`. At each cap, `run.py` records both the prescribed path `b=1` and a stationary minimum near `b=1` in this symmetric, cap-active family. The latter solves the one-dimensional stationarity equation at 80 decimal digits and checks positive curvature.

`global_search.py` searches numerically for the quantity of Theorem 3.3,

```text
E_g(R) = inf { J_g(w1 K(.-z1) + w2 K(.-z2)) - J_g(u_g) : |w|_2 <= R },   u_g = K',
```

over centres and weights, for `epsilon = 0, 0.001, 0.01` and the same 29 caps between 2 and `1e5`. For fixed centres the weight problem is a two-dimensional quadratic trust-region subproblem. The search over centre and half-gap uses a grid, with a fine band near the cap-active gap, followed by 80-digit golden-section refinement of the best candidates. Off-centre and separated configurations are refined separately, and a scan of the completed one-block family checks the reference `u_g = K'`. In all 87 cases the best pair found is symmetric with its coefficient norm at the cap; it matches the stationary point of the symmetric family to relative `1e-12`. This numerical search does not establish global optimality.

For the perturbed targets, Theorem 3.3 predicts `R^2 E_g(R) -> C_g = epsilon ||q||^2 / 2`, the same constant as the prescribed path. The search gives `R^2 E_g(R)/C_g = 1 + 2.5e-9` for `epsilon = 0.01` and `1 + 2.5e-8` for `epsilon = 0.001` at `R = 1e5`. For the exact boundary target (`epsilon=0`), `g` lies in the completed class and the excess decreases like `R^-4`, while the derivative block represents the target exactly. The two laws cross over near `R = (4 epsilon)^(-1/2)`. The search is numerical, not an interval certificate, and the values of epsilon are not certified to lie in the theorem's robustness neighborhood.

## Reproduce

Run from the repository root; the global search takes about a minute with eight processes:

```bash
python experiments/manufactured_1d/run.py --output outputs/manufactured_1d_rerun
python experiments/manufactured_1d/global_search.py --output outputs/manufactured_1d_rerun --jobs 8
python experiments/manufactured_1d/plot.py --input outputs/manufactured_1d_rerun --output outputs/figures/figure_06_rerun
```

Redraw printed Figure 6 directly from supplied records (the manuscript's internal asset is `figure_11.pdf`):

```bash
python experiments/manufactured_1d/plot.py --input data/manufactured_1d --output outputs/figures/figure_06
```

The run uses analytical whole-line Gaussian Gram formulas evaluated with mpmath; the plotting grid is not used to compute energies. Four selected states receive an independent whole-line quadrature check. These are high-precision numerical checks, not interval certificates. `--digits` changes arithmetic precision, with a minimum of 80.

## Records

- `sweep.csv`: 116 records with case, family, cap, amplitude, gap, both coefficient norms, excess above the completed reference, H1 error, PDE residual, scaled rate diagnostics, stationarity residual and curvature.
- `states.csv`: the display grid for the unperturbed target, its forcing and a prescribed finite pair.
- `summary.json`: exact formulas, asymptotic constants, scope statements and independent integration checks.
- `provenance.json`: software versions, source hash, generation command and elapsed time.
- `global_search.csv`: 87 records with case, cap, best capped excess found, scaled excess over its predicted limit, best centre, half-gap and weights, comparison with the symmetric family, best off-centre and separated excesses, and centre probes.
- `global_summary.json` and `global_provenance.json`: limit constants, completed-family checks, scope, software versions and source hashes.

Source files are `experiments/manufactured_1d/run.py`, `global_search.py` and `plot.py`. The figure command in `reproduce.py` includes this study; recomputation uses the explicit run command above.
