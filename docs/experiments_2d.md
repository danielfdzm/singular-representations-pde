# Two-dimensional experiments

Sections 4.3–4.4 compare standard optimization and completion using ordinary finite-pair coordinates. Appendix C.6 reports stable finite-pair evaluation and four-method coordinate controls in prose. Run commands from the directory containing `reproduce.py` after installing its requirements.

## Primary ordinary-coordinate comparison

| Study | Main printed figures | Driver | Primary records |
| --- | --- | --- | --- |
| Gaussian Poisson–Neumann | 7: fields; 8: trajectories | `experiments/raw_coordinates/poisson_completion.py` | `data/raw_main/poisson/` (training order 32), `data/raw_main/poisson/quadrature64/` (order 64), seeds 10–19 |
| Matérn Swift–Hohenberg | 9: trajectories | `experiments/raw_coordinates/matern_completion.py` | `data/raw_main/matern/` (training order 64), `data/raw_main/matern/quadrature96/` (order 96), seeds 0–9 |
| Independent analysis and plotting | Main summaries, thresholds and figures | `experiments/raw_coordinates/analyse_main.py`, `plot_main.py` | All ten starts and both methods in each dataset |

Each primary dataset contains 20 runs, and all four datasets together contain 80. Initial/final states, accepted histories, transfer events, termination statuses and work counts are retained. Appendix C.6 adds 160 stable-coordinate runs across the two problems and two training orders.

Tables 1 and 2 report the Poisson and Matérn final results; Table 3 compares work to common error thresholds. Table 4 in Appendix C.3 reports higher-order training. `experiments/raw_coordinates/write_tables.py` generates the four numerical tables from independent replay results under `outputs/tables/raw_tex/`.

The two primary methods share initial geometries, prescribed pairings, coefficient fitting and optimizer settings. A positive-gap pair is evaluated as

```text
w_plus K(x-c-hd) + w_minus K(x-c+hd),  h > 0.
```

The first coefficient belongs to the centre `c+hd`, and the second to `c-hd`. Training evaluates individual kernels and spatial derivatives directly in float64. It uses neither pair averages nor divided-difference columns for finite pairs. Both methods refit coefficients at each geometry and optimize centres, directions and `log h` with L-BFGS-B; coefficients and geometry are not trained jointly by gradient descent.

The `ordinary` branch retains finite pairs. The `completion` branch converts a qualifying pair to

```text
a K(x-c) - b D_d K(x-c),  a=w_plus+w_minus, b=h*(w_plus-w_minus).
```

Converted pairs use direct kernel/negative-directional-derivative columns; remaining finite pairs continue to use ordinary weights. The coefficient vector keeps two slots per pair, followed by the singleton slots. For finite pairs these are `(w_plus,w_minus)`; for completed pairs they are `(a,b)`. The main coefficient panel shows the norm of this actual stored vector. Derived stable coefficients and reevaluation discrepancies are passive supplementary diagnostics.

## Shared optimization and transfer rules

Both methods use centres in `[-2,2]²`, unrestricted angles, `0<h<=1.5` without a positive floor, `ftol=gtol=0`, L-BFGS-B memory 12, at most 40 line-search steps, and an 8000-fit allowance. One work unit is one distinct coefficient fit, including line-search probes and post-transfer refits. Matérn fits are nonlinear Newton solves; Poisson fits are linear solves, so work counts should be compared within each problem.

Completion checks eligibility every 40 accepted iterates and on optimizer return. Full-gap thresholds are 0.05 for Matérn and 0.12 for Poisson. The measured H1 change must be at most 0.03, using order-64 tensor Gauss–Legendre quadrature. Each conversion preserves centre and direction, refits coefficients and restarts L-BFGS-B. Later candidates are rechecked after preceding refits. Transfer records retain before, converted predictor and refitted states, the measured change, and independent order-80/96 measurements. Failed refits restore the last accepted fitted state. These measured changes are not validated enclosures or a certified objective-descent gate.

Geometry is target-informed. Half-gaps start in `[0.02,0.05]`. Matérn pair centres receive offsets in `[-0.12,0.12]`, angles receive offsets in `[-0.2,0.2]`, and the singleton starts in `[-0.12,0.12]²`. Poisson centre offsets are in `[-0.10,0.10]` and angle offsets in `[-0.18,0.18]`. All Poisson full gaps are below the proximity threshold initially. Exact targets and anchors are in the sources. These are local manufactured comparisons, not discovery tests from arbitrary initial configurations.

## Inner coefficient solves and failures

The original coefficient solvers are reused with diagonal column scaling, their condition cutoff and residual tests. No ridge, rank truncation or stable-basis retry is used. A scaled quadratic condition number above `1e14` stops the run at its last accepted fitted state.

Each Matérn fit minimizes a strictly convex coefficient energy with at most 80 damped Newton iterations. After eight unsuccessful energy-test halvings it switches to an Armijo test on the residual norm. A fit is accepted at relative stationarity at most `1e-13`, or, in recovery mode, within four times its componentwise residual rounding floor. A fit meeting neither criterion stops with `inner_solve_failure`. Poisson linear solves require a residual within four times their rounding floor. Envelope gradients are evaluated only after acceptance. A floating-point acceptance criterion does not certify an exact reduced gradient; residuals in a poorly scaled raw basis can be less informative about derivative directions.

All statuses and returned states remain in the primary summaries. Optimizer success is distinct from attaining a small error or completing every pair. In the raw Matérn study, start 1 completes one pair at training order 64 and none at order 96; the higher-order outlier is retained. Higher quadrature can change optimization paths and transfer counts. Neither failed runs nor incomplete transfers are replaced by supplementary stable-coordinate results.

## Independent errors, figures and work summaries

```sh
python experiments/raw_coordinates/analyse_main.py --study matern --output outputs/tables/raw_matern.json
python experiments/raw_coordinates/analyse_main.py --study poisson --output outputs/tables/raw_poisson.json
python experiments/raw_coordinates/plot_main.py --analysis-directory outputs/tables --output outputs/figures
```

The independent NumPy evaluator constructs direct finite translates and completed kernel derivatives; it does not import a training solver or convert finite pairs into stable divided-difference columns. Main trajectory errors use saved histories and order-96 quadrature, with no retraining. Error panels show running minima; coefficient panels show current stored-vector norms. Abnormal termination is marked. The Poisson field plot uses start 16 and a 241×241 grid; marker offsets and connectors distinguish close atoms from their actual locations.

Final sensitivity checks use order 128, for both original and higher-order training:

```sh
python experiments/raw_coordinates/analyse_main.py --study matern --results data/raw_main/matern --quadrature-order 128 --final-only --output outputs/tables/raw_matern_eval128.json
python experiments/raw_coordinates/analyse_main.py --study matern --results data/raw_main/matern/quadrature96 --quadrature-order 128 --final-only --output outputs/tables/raw_matern_quadrature96.json
python experiments/raw_coordinates/analyse_main.py --study poisson --results data/raw_main/poisson --quadrature-order 128 --final-only --output outputs/tables/raw_poisson_eval128.json
python experiments/raw_coordinates/analyse_main.py --study poisson --results data/raw_main/poisson/quadrature64 --quadrature-order 128 --final-only --output outputs/tables/raw_poisson_quadrature64.json
```

Common-threshold work counts the first recorded initial, accepted, transfer or final state that reaches the threshold. Line-search probes count as work but are not candidate hits. Medians apply only to reaching starts, with misses and statuses retained. Fully completed starts can be identified as a secondary subset; primary summaries use every start. Matérn analysis also measures the quadratic-energy error `sqrt(||(1+Delta)v||²+||v||²)`; no equivalence with the full H2 norm on the free-boundary space is asserted.

## Repeat primary training

```sh
python experiments/raw_coordinates/matern_completion.py --output outputs/raw_matern_rerun --jobs 8
python experiments/raw_coordinates/matern_completion.py --output outputs/raw_matern_q96_rerun --quadrature-order 96 --jobs 8
python experiments/raw_coordinates/poisson_completion.py --method standard --output outputs/raw_poisson_ordinary_rerun --jobs 8
python experiments/raw_coordinates/poisson_completion.py --method completion --output outputs/raw_poisson_completion_rerun --jobs 8
python experiments/raw_coordinates/poisson_completion.py --method standard --output outputs/raw_poisson_ordinary_q64_rerun --quadrature-order 64 --jobs 8
python experiments/raw_coordinates/poisson_completion.py --method completion --output outputs/raw_poisson_completion_q64_rerun --quadrature-order 64 --jobs 8
python experiments/raw_coordinates/collect_runs.py --study poisson --ordinary outputs/raw_poisson_ordinary_rerun --completion outputs/raw_poisson_completion_rerun --output outputs/raw_poisson_rerun
python experiments/raw_coordinates/collect_runs.py --study poisson --ordinary outputs/raw_poisson_ordinary_q64_rerun --completion outputs/raw_poisson_completion_q64_rerun --output outputs/raw_poisson_q64_rerun
```

Matérn runs both `ordinary` and `completion` by default. Its `--reuse-ordinary` option can reuse a matching raw ordinary dataset with verified source hashes; this was used to combine the earlier raw order-64 baseline with new completion runs. Poisson selects one branch per command: CLI `standard` is stored as `ordinary`. `collect_runs.py` combines the branches and checks matched starts and generating-source hashes. Analyse the combined directory with `--results`; main plotting accepts it with `--poisson`.

The drivers validate physical feature equivalence, coefficient fits, geometry gradients and transfer conversion before running. Matérn validation covers zero, either one, and both completed pairs, plus failed-refit rollback. Seeds, quadrature and source hashes are recorded. Existing output directories are refused; new runs produce uncompressed JSON. `--jobs` runs independent single-threaded fits in parallel. Software versions are recorded per dataset, and platform/library changes can affect trajectories.

## Appendix stable-coordinate controls

The following numerical records reproduce Appendix C.6 and the exact-target reliability check in Appendix C.3:

| Appendix material | Code | Records |
| --- | --- | --- |
| Four-method Matérn comparisons | `experiments/matern_swift_hohenberg_2d/` | `data/matern/`, `data/matern/quadrature96/` |
| Four-method Poisson comparisons | `experiments/poisson_neumann_2d/` | `data/poisson/`, `data/poisson/quadrature64/` |
| Training quadrature at the exact target | `experiments/training_quadrature.py` | `data/audits/training_quadrature.json` |

Each appendix dataset contains four methods at ten starts. All four store stable `(a,b)` finite-pair coefficients and evaluate stable pair formulas. In these datasets, `ordinary` means standard optimization in stable coordinates. The primary datasets instead train using the ordinary kernel weights described above.

| Appendix method | Gap coordinate | Transfer or restart |
| --- | --- | --- |
| `ordinary` | `log h` | none |
| `restart` | `log h` | L-BFGS-B restart at the completion trigger, at most once per pair |
| `squared_gap` | `t=h²`, `1e-12<=t<=2.25` | none |
| `completion` | `log h` | derivative transfer and L-BFGS-B restart |

These controls examine optimizer restarts and the flat logarithmic-gap direction. For positive gaps, exact coefficient-refitted objectives agree with those of the raw basis, but floating-point evaluation and coefficient solves can differ. The current paper reports these controls numerically, without appendix figures.

Fresh appendix runs and replay commands are:

```sh
python experiments/matern_swift_hohenberg_2d/experiment.py --output outputs/stable_matern_rerun --jobs 8
python experiments/poisson_neumann_2d/experiment.py --output outputs/stable_poisson_rerun --jobs 8
python experiments/matern_swift_hohenberg_2d/analyse.py --output outputs/tables/matern.json
python experiments/poisson_neumann_2d/analyse.py --output outputs/tables/poisson.json
python experiments/training_quadrature.py
```

Appendix drivers default to all four methods; `--methods` selects a subset and higher quadrature uses `--quadrature-order 96` for Matérn or 64 for Poisson. To analyse higher-order records, pass `--results data/matern/quadrature96` or `--results data/poisson/quadrature64` to the respective `analyse.py` command. `reproduce.py --tables` replays the controls at both training orders; `--figures` reproduces the nine printed figures.

Stable Matérn records use the degree-six exponential-quotient continuation. No coefficient fit fails in those stable controls; that statement does not apply to raw main runs. Raw finite-pair training evaluates individual radial kernels directly. The generating solver sources are retained unchanged so their hashes continue to match the recorded provenance.

At the main training orders, the stable finite-pair and completion medians are respectively `5.3e-8` and `3.5e-8` for Matérn, and `9.2e-7` and `1.2e-7` for Poisson. The squared-gap medians are `3.6e-8` and `1.4e-7`. At Matérn order 96, the squared-gap and completion medians are `1.3e-8` and `2.1e-7`. These are the values reported in Appendix C.6; the retained records include every start and all four methods.

The exact-target coefficient-refit check gives a Matérn relative H1 error of `2.33e-8` at training order 64 and `5.27e-10` at 96, reproducing Appendix C.3. These checks are floating-point evidence, not continuum certifications.

## Data integrity

Readers accept uncompressed `results.json` and packaged deterministic `results.json.gz`. Provenance files record generating commands, source hashes and canonical per-run hashes; `reproduce.py --check` verifies the release inventory and all 240 retained runs. Primary raw and appendix stable records occupy separate directories. Source snapshots identify the training representation used for each dataset. Generated outputs go to `outputs/` and do not overwrite publication assets.
