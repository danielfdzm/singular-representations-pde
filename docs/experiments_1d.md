# One-dimensional Gaussian completion at fixed multiplicity

Section 4.3 reruns the Gaussian target and initial centres recovered from [historical commit 8b15f87](https://github.com/danielfdzm/singular-representations-pde/tree/8b15f8714af3f87abcde9fa484eddd7d66343f06). The live GitHub default branch was checked at [bf58b8f](https://github.com/danielfdzm/singular-representations-pde/tree/bf58b8fa2fb188681b24aeaae8e5fb25541acee9), which removed that example. Its staged dictionary included an intermediate multiplicity of P+2. The current rerun replaces the whole eight-feature cluster at once, retaining P slots throughout, and uses the paper's variational setting and a validated transfer check.

## Target and energy

For `K(x) = exp(-x²/2)`, the target is `u*(x) = K'''(x) + 0.6 K(x-1.4)`. With `f = u* - (u*)''`, it uniquely minimizes

`J(u) = 0.5 ∫(u² + |u'|²) dx - ∫fu dx`

over H¹(R). The objective gap is `0.5 ||u-u*||²_H¹(R)`. The target has multiplicity five; the completed representation retains all allocated derivative slots, including coefficients close to zero.

## Protocol

Widths are 9, 10 and 12. Eight centres start at `h*(-7,-5,-3,-1,1,3,5,7)` with `h=0.5`. A further centre b starts at 2.25 and is optimized in `[0.85,3.5]`. Extra centres are fixed at -3.2 for width 10 and at -3.2, -2.1, 2.6 for width 12.

The displayed initial configuration takes one damped coefficient step from zero, `w = 0.8 w_LS`, at the starting centres. Its relative H¹ error is about 0.208 for width nine. Subsequent fits use the full least-squares solution. Each geometry evaluation solves for coefficients using column-normalized SVD (relative cutoff 1e-14) on an 801-node Gauss–Legendre discretization of the H¹ residual on `[-12,12]`. Bounded Nelder–Mead varies `(log(h),b)`, initially restricting h to `[0.175,0.5]`. If no transfer passes, a fresh bounded solve allows a further factor-0.75 contraction. The parameter/objective tolerances are 1e-9/1e-15 before transfer and 1e-11/1e-25 after transfer, with at most 1000 iterations per solve.

At each accepted geometry iterate with h at most 0.175, the eight translates are replaced by orders 0 through 7 at zero, with moment coefficients `alpha_r = sum_j w_j*(-b_j)^r/r!`. Other features are copied unchanged. Before any refit, the driver encloses the exact whole-line H¹ difference between the stored old and proposed states and requires it to be at most 3% of the incumbent norm. It stops at the first passing iterate. The recorded SciPy callback termination is intentional, not a claim of finite-geometry convergence. Rejected predictors are retained in the record.

After transfer, coefficients are refitted and b is optimized. An independent interval comparison also requires the refitted energy to decrease. These runs test one prescribed whole-cluster completion; they do not implement the general adaptive search or provide a matched-work optimizer comparison.

## Continuum checks and results

For `C(d)=sqrt(pi)*exp(-d²/4)`, exact Gaussian overlaps satisfy

`<K^(r)(.-a), K^(s)(.-b)>_H¹ = (-1)^r [C^(r+s)(a-b) - C^(r+s+2)(a-b)]`.

The driver evaluates signed Gram sums with 80-digit interval arithmetic, treating every stored binary64 centre and weight as an exact real input, and target constants as exact decimals. This includes rounding in moment formation. Reported errors and transfer bounds are whole-line enclosures, independent of the optimization quadrature. A 110-digit replay verifies the enclosures; a separate 1601-node quadrature check verifies the sampled evaluations.

| Width | Finite checkpoint relative H¹ error | Completed relative H¹ error | Transfer upper bound, rounded up | Total linear fits |
| --- | ---: | ---: | ---: | ---: |
| 9 | 1.1501e-4 | 6.2320e-16 | 1.227% | 99 |
| 10 | 9.3721e-5 | 1.6062e-15 | 1.228% | 99 |
| 12 | 5.8510e-6 | 8.6147e-16 | 0.403% | 113 |

Finite checkpoints occur at h = 0.175, 0.175 and 0.13125. Coefficient norms fall from 182.8, 184.2 and 476.9 to approximately `sqrt(1+0.6²) = 1.166190379` after completion and refitting. The final errors are at rounding scale. The transfer alone does not produce those errors.

## Reproduce

From the repository root:

```sh
# Fresh runs; preserves supplied records.
python experiments/gaussian_completion_consistent.py

# Replay the supplied states and transfer enclosures.
python experiments/gaussian_completion_consistent.py --audit

# Redraw Figure 5 from the supplied records.
python experiments/plot_gaussian_1d_completion.py --output outputs/figures/figure_05

# Plot fresh results instead.
python experiments/plot_gaussian_1d_completion.py \
  --input outputs/completion_consistent/results.json \
  --output outputs/figures/figure_05_fresh
```

The supplied [results](../data/completion_consistent/results.json) and [audit](../data/completion_consistent/audit.json) record source hashes, environment, actual solve counts, predictors and interval bounds. Fresh output goes under `outputs/completion_consistent/`. Figure 5 uses width nine uses a three-column format: initial damped fit, pre-transfer checkpoint, completed fit; weighted features and centre markers above, target/state and residual shading below. Upper vertical scales vary between panels. Plot output includes PDF/SVG/PNG and provenance JSON.

The historical sources and records remain unchanged under `experiments/boundary_completion_*.py` and `data/completion/`; their protocol is documented in [the archived guide](archive/gaussian_1d_historical.md). They do not supply the current numerical results. The superseded one-dimensional Matérn rerun likewise remains archived and is absent from the paper.
