# Two-dimensional experiments

These files reproduce Sections 4.3–4.4 and Appendices D.3–D.4 of *Singular Parameters and Missing Limits in Neural PDE Solvers*. Run the commands from the repository root after installing its requirements.

## What is included

| Paper item | Source | Stored records |
|---|---|---|
| Matérn Swift–Hohenberg comparison; Figures 7–8; Table 1 | `experiments/matern_swift_hohenberg_2d/` | `data/matern/`, seeds 0–9, all three methods |
| Gaussian Poisson–Neumann comparison; Figures 9–10; Table 2 | `experiments/poisson_neumann_2d/` | `data/poisson/`, seeds 10–19, all three methods |
| Matérn/Gaussian target quadrature and small-gap evaluation checks in D.3 | Matérn `audit.py` and `gaussian_target.py` | `data/matern/audits/replayed_checks.json` |
| Poisson target conditioning, compatibility, and transfer checks in D.4 | Poisson `audit.py` | `data/poisson/audits/replayed_checks.json` |
| Unit-scale stalled-optimization observation in D.4 | Poisson `unit_scale.py` | `data/poisson/audits/unit_scale/`, seeds 0–4, ordinary and completion methods |

Each main dataset contains 30 runs, including initial/final states, accepted histories, transfer states, optimizer termination statuses, work counts, independent quadrature checks, and gradient-validation records. Table 1 uses the nine starts at which completion transfers: 0, 2, 3, 4, 5, 6, 7, 8, 9. Start 1 is retained in the data and Figure 8. Table 2 uses all ten reported Poisson starts. Figure 7 uses Matérn start 0; Figure 9 uses Poisson start 16.

Only the unit-scale batch needed for the Appendix D.4 comparison is retained from the earlier Poisson configurations. Other tuning sweeps and unrelated PDE experiments are excluded.

## Redraw Figures 7–10

```bash
python experiments/matern_swift_hohenberg_2d/plot_fields.py --output outputs/figures
python experiments/matern_swift_hohenberg_2d/plot_results.py --output outputs/figures
python experiments/poisson_neumann_2d/plot_fields.py --output outputs/figures
python experiments/poisson_neumann_2d/plot_results.py --output outputs/figures
```

Each command reads the published compressed records directly and writes PDF, PNG, SVG, and plotting metadata. The field plots use a 241×241 grid; pair stars are visually separated along their true joining direction, and connectors indicate the true positions. The trajectory error panels show running minima; coefficient-size panels show the current iterate. Redrawing trajectories evaluates the saved histories at order 96 quadrature and can take several minutes. No optimization is repeated.

## Recompute Tables 1–2

```bash
python experiments/matern_swift_hohenberg_2d/analyse.py --output outputs/tables/matern.json
python experiments/poisson_neumann_2d/analyse.py --output outputs/tables/poisson.json
```

These commands evaluate only the stored final states. Output includes per-start relative H1 errors and the exact ranges behind the paper's rounded tables. Reference ranges are in `data/{matern,poisson}/paper_table.json`.

H1 errors use tensor Gauss–Legendre quadrature of order 96 against the manufactured solution. A finite pair has ordinary-kernel weights `a/2 ± b/(2h)`; after completion its two coefficients are `(a,b)`. The reported coefficient norm is the Euclidean norm in the applicable coordinates. For Matérn a work unit is a nonlinear coefficient fit; for Poisson it is a linear coefficient solve.

## Repeat the reported training

```bash
python experiments/matern_swift_hohenberg_2d/experiment.py --output outputs/matern_rerun
python experiments/poisson_neumann_2d/experiment.py --output outputs/poisson_rerun
```

The public defaults are the paper configurations. Matérn uses seeds 0–9 and quadrature order 64; Poisson uses seeds 10–19 and order 32. Both use 8000 work units, 40 optimizer iterations per block, `ftol=gtol=0`, initial half-gaps in [0.02, 0.05], transfer tolerance 0.03, and the matched finite-restart control. Full-gap proximity thresholds are 0.05 and 0.12 respectively. Existing output directories are refused to protect stored runs. Outputs are uncompressed JSON; the same plotting and analysis commands accept a rerun directory through `--input` or `--results`.

The main source was repackaged without changing solver mathematics. Unused Matérn forcing cases were removed; the auxiliary mixed-mode load remains solely for the original recorded envelope-gradient check. CLI defaults and descriptive metadata were corrected to the reported experiment. The original source and result hashes are recorded in each data directory's `provenance.json`; source hashes from the original run do not describe the repackaged public source. Original numerical run records are unchanged. The historical unit-scale solver functions are preserved; CLI defaults select its included batch.

Recorded versions were Python 3.14.0, NumPy 2.3.5, SciPy 1.17.0 and PyTorch 2.10.0. Numerical linear algebra, library versions and hardware can change optimization paths and last-bit diagnostics; the published records remain the reference for the figures and tables. The scripts use double precision and one PyTorch thread. NumPy, SciPy and PyTorch are needed for solvers/analysis; Matplotlib is also needed for figures.

## Appendix checks

```bash
python experiments/matern_swift_hohenberg_2d/audit.py
python experiments/poisson_neumann_2d/audit.py
```

These create fresh reports in `outputs/audits/` and do not train. They cover the exact-target quadrature comparison, cancellation at the smallest stored Matérn half-gap, coefficient conditioning and recovery, the Poisson mean/compatibility calculation, and the stored energy-audit/work/transfer bounds. `data/*/audits/replayed_checks.json` records the release replay. Near-machine-precision values depend on summation order and library versions. The original point grid/norm for the manuscript's brief Matérn cancellation diagnostic was not saved separately; the public replay explicitly defines both an order 64 quadrature L2 metric and a relative maximum norm. It reproduces the cancellation failure and stable evaluation, without claiming bit-for-bit agreement with that historical pointwise diagnostic.

The Matérn energy is strictly convex with its manufactured unique minimizer. No H2-coercivity claim is used. Transfers in both experiments use measured H1 changes, not validated enclosures, and the quadrature audits do not certify continuum errors.

For the historical unit-scale optimization comparison alone:

```bash
python experiments/poisson_neumann_2d/unit_scale.py \
  --output outputs/poisson_unit_scale_rerun --seeds 0 1 2 3 4 \
  --max-solves 8000 --quadrature-order 32 --block-iterations 40 \
  --proximity-threshold 0.12 --transfer-tolerance 0.03 --ftol 0 --gtol 0 \
  --initial-half-gap-min 0.02 --initial-half-gap-max 0.05
```

The unit-scale target/anchor centres and singleton coefficients also differ from the final scale 0.45 configuration. This historical comparison therefore does not isolate Gaussian scale as the only changed variable. The included records support the observed stalled optimization and its conditioning; they are not included in the main ten-start aggregate.

## Data format

`results.json.gz` files use deterministic gzip (`mtime=0`) to keep complete histories comfortably below GitHub's file-size limit. Readers prefer `results.json` when present and otherwise read `results.json.gz`. No unpacking is needed. `configuration.json`, `h1_errors.json`, `summary.csv`, `gradient_validation.json`, and `paper_table.json` provide compact entry points. Run failures and nontransferring cases have not been filtered out.
