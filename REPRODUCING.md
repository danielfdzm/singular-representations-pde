# Reproducing the paper

This repository accompanies **Singular Parameters and Missing Limits in Neural PDE Solvers**. It contains the experiments in Sections 4.1–4.4, the checks reported in Appendices D.1–D.4, and the analytical illustrations in Figures 1–3. Older studies that are absent from the current paper are not part of this release.

## Install

Use Python 3.11 or later. The reproduction commands were checked on Python 3.14.0 with the versions pinned in `requirements.txt`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, use `conda env create -f environment.yml` and `conda activate singular-representations-pde`. The Matplotlib scripts use a headless backend. Latin Modern Roman gives the publication typography; an installed serif font is used otherwise. Ghostscript is optional and is used only when requesting a new PNG preview of Figure 4.

The original tanh runs used an older Python/NumPy/Matplotlib environment, recorded with their data. Fresh optimization is not guaranteed to reproduce the last digits across software versions or platforms. The supplied recorded states and exact publication figures provide the comparison baseline.

## Verify the supplied files

```sh
python reproduce.py --check
```

This uses only the Python standard library to verify the SHA-256 inventory. It also checks the figure map and the recorded runs for the two matched experiments. The inventory covers the scientific sources, numerical records, documentation, and publication figures. Generated outputs are excluded.

## Redraw the figures

```sh
python reproduce.py --figures
```

`--fast` is an alias for this command. New plots go into `outputs/figures/`; the publication assets remain untouched. Figures 1–3 are recalculated from their prescribed formulas, Figure 4 is recomposed from its recorded vector panels, and Figures 6–10 are redrawn from the stored numerical records. Figure 5 is copied from its preserved publication PDF because exact historical optimizer snapshots were not saved.

| Figure | Published PDF | Source |
| --- | --- | --- |
| 1 | [Tanh singular limit](figures/paper/figure_01.pdf) | `illustrations/plot_tanh_limit.py` |
| 2 | [Deep-network escape and degeneracy](figures/paper/figure_02.pdf) | `illustrations/plot_forced_singularity.py` |
| 3 | [Gaussian collision and completion](figures/paper/figure_03.pdf) | `illustrations/plot_completion_patterns.py` |
| 4 | [Tanh bias trajectories](figures/paper/figure_04.pdf) | `experiments/tanh/plot_bias.py` |
| 5 | [Tanh slope snapshots](figures/paper/figure_05.pdf) | Preserved historical PDF; fresh-run source in `experiments/tanh/` |
| 6 | [Gaussian precision sweep](figures/paper/figure_06.pdf) | `experiments/precision/plot.py` |
| 7 | [Matérn fields](figures/paper/figure_07.pdf) | `experiments/matern_swift_hohenberg_2d/plot_fields.py` |
| 8 | [Matérn trajectories](figures/paper/figure_08.pdf) | `experiments/matern_swift_hohenberg_2d/plot_results.py` |
| 9 | [Poisson–Neumann fields](figures/paper/figure_09.pdf) | `experiments/poisson_neumann_2d/plot_fields.py` |
| 10 | [Poisson–Neumann trajectories](figures/paper/figure_10.pdf) | `experiments/poisson_neumann_2d/plot_results.py` |

PDF metadata, font availability and plotting-library versions can change regenerated file bytes. Exact publication PDFs and their browser previews are supplied under `figures/`.

## Recompute the tables

```sh
python reproduce.py --tables
```

This reevaluates the final recorded states with order-96 quadrature and reports per-start relative `H¹` errors, coefficient norms, smallest half-gaps, coefficient-solve counts, and transfers. JSON summaries are written to `outputs/tables/`.

Table 1 uses the nine Matérn starts where completion transfers; all ten starts, including the no-transfer case, remain in the supplied records. Table 2 uses all ten Poisson–Neumann starts, numbered 10–19. Full optimizer histories are stored as compressed JSON; the scripts read them directly.

## Repeat the calculations

Individual commands, run settings, provenance, and the reported appendix checks are documented in:

- [Tanh experiments and precision sweep](docs/experiments_tanh_precision.md): fresh binary32/binary64 training, the historical slope-collision protocol, and the full high-precision sweep.
- [Matérn and Poisson–Neumann experiments](docs/experiments_2d.md): matched starts and controls, the two tables, quadrature and gradient audits, kernel evaluation checks, and the reported scale comparison.
- [Analytical illustrations](docs/illustrations.md): formulas and independent commands for Figures 1–3.

Fresh training is separate from the quick figure and table commands. It writes new results under `outputs/`, without modifying the supplied records. No external dataset is required.
