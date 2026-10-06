# Singular Parameters and Missing Limits in Neural PDE Solvers

**Daniel Fernández**

[Experiments](#the-experiments) · [Reproduce the results](REPRODUCING.md) · [Paper figures](figures/paper/) · [Citation](CITATION.cff)

A neural PDE approximation can become more accurate while its parameters grow without bound. At fixed width, the limiting solution may be smooth and still lie outside the model's realization class.

For one-dimensional translated-feature models, the paper characterizes every weak Sobolev limit with unrestricted centres: recentered clusters converge strongly and escaping clusters vanish weakly. The resulting completed class admits variational minimizers under the stated assumptions. A complementary deep-tanh result identifies parameter escape or hidden-layer degeneracy as necessary routes to missing limits.

<p align="center"><img src="figures/previews/figure_03.png" alt="Three Gaussian features collide and approach a derivative block." width="920"></p>

## The experiments

The package follows the current paper: nine figures, four numerical studies, and the numerical evaluation and coordinate controls reported in Appendix C.

| Section | Question | Paper figures | Code and records |
| --- | --- | --- | --- |
| 4.1 | What happens to parameters as tanh neurons approach a missing limit? | [4](figures/paper/figure_04.pdf), [5](figures/paper/figure_05.pdf) | [Tanh experiments](docs/experiments_tanh_precision.md#tanh-experiments) |
| 4.2 | Does the coefficient-capped excess follow the law of Theorem 3.3? | [6](figures/paper/figure_06.pdf) | [One-dimensional study](docs/manufactured_1d.md) |
| 4.3 | How does completion affect a wider Poisson–Neumann model? | [7](figures/paper/figure_07.pdf), [8](figures/paper/figure_08.pdf) | [Poisson comparison](docs/experiments_2d.md) |
| 4.4 | How does completion affect a nonlinear fourth-order Matérn problem? | [9](figures/paper/figure_09.pdf) | [Matérn comparison](docs/experiments_2d.md) |
| C.6 | How do stable evaluation and coordinate controls affect accuracy? | Numerical results in the text | [Precision sweep](docs/experiments_tanh_precision.md#precision-sweep), [two-dimensional controls](docs/experiments_2d.md#appendix-stable-coordinate-controls) |

The main Matérn and Poisson comparisons each include all ten starts and two methods: standard optimization and completion. Both train finite pairs using ordinary translated-kernel weights and direct kernel evaluation. Completion converts eligible pairs to kernel/derivative blocks and continues optimization. Primary records are in `data/raw_main/`, including higher-order training repetitions. Four-method comparisons using stable finite-pair coordinates are in `data/matern/` and `data/poisson/` for Appendix C.6, including restart and squared-gap controls. Figures 1–3 are [analytical illustrations](docs/illustrations.md).

The four printed tables report Poisson final results (Table 1), Matérn final results (Table 2), work to common error thresholds (Table 3), and higher-order training quadrature (Table 4, Appendix C.3). The [reproduction guide](REPRODUCING.md) maps each table to its generated file. The appendix reports its evaluation and coordinate controls numerically and contains no figures.

## Quick start

```bash
git clone https://github.com/danielfdzm/singular-representations-pde.git
cd singular-representations-pde
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --check
python reproduce.py --figures
python reproduce.py --tables
```

New outputs go into `outputs/`. The nine publication PDFs use their printed figure numbers in `figures/paper/`; [the figure map](figures/manifest.json) lists their manuscript asset names and previews. The [reproduction guide](REPRODUCING.md) gives individual commands and instructions for fresh runs. The check command verifies the release inventory and all 240 recorded two-dimensional runs: 80 primary runs and 160 appendix controls.

Figure 5 preserves a historical run: its original script, publication PDF and rounded logged states are included, but exact unrounded optimizer snapshots are unavailable. Its reproduction command restores the preserved figure; a fresh run is a new realization of the same protocol.

## Reading the results

The two-dimensional comparisons use manufactured targets, prescribed pairings and target-informed initial anchors. The main solvers refit coefficients at every geometry and then optimize centres, directions and logarithmic half-gaps with L-BFGS-B. Finite-pair coefficient slots contain ordinary weights; transferred slots contain the coefficients of the kernel and its negative directional derivative. Figures 8 and 9 show the norm of these actual stored vectors. Independent NumPy analysis evaluates finite translates and completed derivative blocks directly. Every coefficient fit must meet its stationarity criterion before its envelope gradient is used; failed fits or ill-conditioned quadratic systems stop the run at its last accepted state.

Primary summaries retain all starts, termination statuses and missed accuracy thresholds. Higher-order training can change optimization paths and transfer counts; the Matérn start-1 outlier is retained in both training studies. Successful optimizer termination alone does not establish a small error or an exact reduced gradient. Work-to-error medians apply to starts that reach the common threshold, with misses reported separately. Matérn work counts nonlinear coefficient fits and Poisson work counts linear solves, so counts should be compared within each problem. Appendix C studies stable evaluation, restarts, squared-gap coordinates and training quadrature. These local comparisons do not establish a general performance advantage.

The whole-line one-dimensional study provides a direct bridge to the theoretical setting. A search over two-atom networks under a coefficient cap recovers the predicted constant for two perturbed targets and the faster law at the boundary target. The search does not establish global optimality or certify that the perturbations lie in the theorem's robustness neighbourhood; see [the study guide](docs/manufactured_1d.md). The bounded-domain two-dimensional experiments illustrate directional collisions; the whole-line closure theorem does not establish their closure. Floating-point quadrature and transfer checks are not validated continuum enclosures.

## Citation and license

Please cite **Singular Parameters and Missing Limits in Neural PDE Solvers**, by Daniel Fernández, and this repository when using these materials. Machine-readable metadata is in [CITATION.cff](CITATION.cff). The repository [license](LICENSE) applies.
