# One-dimensional Gaussian derivative completion

Section 4.3 restores the historical one-dimensional experiment removed from the GitHub repository in commit `bf58b8fa2fb188681b24aeaae8e5fb25541acee9`. Its original sources and numerical records are recovered byte for byte from parent commit `8b15f87`; no optimization was repeated for this revision. Figure 7 is a new rendering of the saved stage endpoints.

## Target and training

The target is `u*(x) = K'''(x) + 0.6 K(x-1.4)`, with `K(x) = exp(-x²/2)`. The loss is the squared relative L2 fitting error on `[-8,8]`, evaluated by the composite trapezoidal rule at 2001 equally spaced points, plus a guard penalty `max(0,0.85-|c|)²` for the trainable singleton center. Displayed errors exclude the guard. This is a function-approximation experiment, not a PDE energy minimization.

Initial widths are 9, 10 and 12. Eight cluster centers are `h*(-7,-5,-3,-1,1,3,5,7)`, initially with `h=0.5`; the primary singleton starts at 2.25 and is trained. The additional singleton centers at larger widths are fixed at -3.2, -2.1 and 2.6 as applicable. Initial weights have scale `1e-3`. Full-batch float64 Adam uses learning rate `1.95e-4`, betas `(0.9,0.999)`, epsilon `1e-8` and gradient clipping at 100. Pair collapse is triggered when the active separation falls below fractions 0.35, 0.75 and 0.86 of its value at the beginning of the corresponding stage; Taylor transfer introduces the next kernel derivative.

Immediately after the third transfer, the final dictionary is polished with strong-Wolfe L-BFGS, learning rate 1, at most 200 iterations / 400 evaluations, gradient tolerance `1e-14`, and change tolerance `1e-18`. The cached timeline assigns this polish an artificial duration of 1000 units and repeats its final state out to 100000. These labels are plotting padding, not measured optimizer work. Figure 7 therefore uses stage snapshots without an iteration axis.

The first three dictionaries have P explicit basis columns, but their total multiplicities in the paper's definition are P, P and P+2. In the third dictionary, the K' and K'' terms at each of two centers require an order-0 slot even though that coefficient is zero. The final dictionary has total multiplicity P−4. Consequently this historical training heuristic is not a fixed-budget implementation of Algorithm 1. The target itself has total multiplicity five and is covered by the one-dimensional Gaussian closure theorem.

## Reproduce Figure 7 and verify the saved states

```bash
python3 experiments/plot_gaussian_1d_completion.py --output outputs/figures/figure_07
```

This command verifies the coefficient norm and total multiplicity of all twelve snapshots, recomputes their L2 errors on the original 2001-point grid and an independent 20001-point grid, and writes PDF/SVG/PNG plus a JSON audit. The figure displays width nine: weighted atoms above and their sum against the target below. The first three columns are saved pre-transfer stage endpoints; the last is the polished final state. Norm annotations refer to the coefficients of the displayed basis, whose coordinates change between stages.

| Initial width | Final recorded relative L2 error | Final coefficient norm |
|---|---:|---:|
| 9 | 3.4258463e-10 | 1.1661903841 |
| 10 | 4.8599435e-10 | 1.1661903836 |
| 12 | 9.5348938e-10 | 1.1661904666 |

The coefficient norm approaches `sqrt(1+0.6²)`, and the trained primary singleton center approaches 1.4. The fine-grid recheck confirms these numerical errors; it is not a validated continuum error bound.

## Historical controls and provenance

The archive includes finite-translate Adam controls and a separate L-BFGS polish with the same local solver settings. The completed and finite branches use different initial weight seeds (`80000+17P` versus `70000+P`) and reach the polish after different Adam trajectories. They are not a matched-start or matched-work benchmark. After the finite controls' polish, the recorded validation errors are approximately `2.0682e-3`, `1.8794e-3` and `1.5769e-2` for widths 9, 10 and 12. These historical controls are retained for provenance; Figure 7 makes no speedup or solver-ranking claim.

Restored files:

- `experiments/boundary_completion_experiment.py`: unchanged historical Adam/completion driver and old plotting routines.
- `experiments/boundary_completion_matched_control_lbfgs.py`: unchanged historical finite-control polish and fine-grid evaluator; its filename uses the historical term "matched" for the local L-BFGS settings.
- `experiments/artifact_paths.py`: unchanged historical path helper.
- `data/completion/boundary_completion_third_derivative_atoms_lbfgs_plateau.json`: saved Adam histories, finite controls and full stage snapshots.
- `data/completion/boundary_completion_matched_control_lbfgs.json`: saved finite-control polish and fine-grid endpoint checks.
- `data/completion/provenance.json`: source commit and recovered file hashes.

For fresh training, the original driver remains available, but its historical defaults write JSON into `data/completion/`; use a distinct `--output-suffix` to avoid replacing the supplied record. The current reproduction entry point only reads those records and writes new files under `outputs/`.

The superseded whole-line Matérn rerun is not part of the current paper. Its documentation is preserved in `docs/archive/whole_line_matern_rerun.md`; existing sources and records remain in `experiments/matern_swift_hohenberg_1d/` and `outputs/matern_1d/`. Historical figures and the publication assets before renumbering are preserved under `outputs/archive/` in the manuscript workspace.
