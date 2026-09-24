# Singular Parameters and Missing Limits in Neural PDE Solvers

**Daniel Fernández · Enrique Zuazua**

[Experiments](#the-experiments) · [Reproduce the results](REPRODUCING.md) · [All paper figures](figures/paper/) · [Citation](CITATION.cff)

A neural PDE approximation can become more accurate while its parameters grow without bound. At fixed width, the limiting solution may be smooth and still lie outside the neural model's realization class.

This paper connects that behaviour to the **closure of the neural parametrisation class**. For deep tanh networks, approaching a missing limit forces parameter escape or hidden-layer degeneracy. For one-dimensional translated-feature models, we characterise the closure through kernel derivatives, counted within the original width. Admitting these limits—**completion**—restores variational attainment under the stated assumptions and provides stable coordinates for computation.

<p align="center">
  <img src="figures/previews/figure_05.png" alt="Gaussian completion: initial configuration, finite checkpoint and completed derivative block." width="920">
</p>

*Figure 5. Section 4.3 completes an eight-feature Gaussian cluster within the original budget. The transfer is checked in the whole-line H¹ norm before refitting.*

## The experiments

The repository contains the five numerical studies in the paper, their reported appendix checks, and the two analytical illustrations. Every figure below is the current publication figure.

| Section | Question | Figures | Code and records |
| --- | --- | --- | --- |
| 4.1 | What happens to parameters as two tanh neurons approach a missing limit? | [3](figures/paper/figure_03.pdf), [4](figures/paper/figure_04.pdf) | [Tanh experiments](docs/experiments_tanh_precision.md#tanh-experiments) |
| 4.2 | Can stable evaluation recover the accuracy lost when Gaussian centres coalesce? | — | [Precision sweep](docs/experiments_tanh_precision.md#precision-sweep) |
| 4.3 | Can a whole Gaussian collision be completed within the original feature budget? | [5](figures/paper/figure_05.pdf) | [One-dimensional experiment](docs/experiments_1d.md) |
| 4.4 | Does completion work for Matérn features on a nonlinear fourth-order PDE? | [6](figures/paper/figure_06.pdf), [7](figures/paper/figure_07.pdf), Table 1 | [Matérn experiment](docs/experiments_2d.md) |
| 4.5 | Does completion improve accuracy and computational cost in a wider Poisson–Neumann model? | [8](figures/paper/figure_08.pdf), [9](figures/paper/figure_09.pdf), Table 2 | [Poisson–Neumann experiment](docs/experiments_2d.md) |

The Matérn comparison uses starts 0–9; the Poisson comparison uses starts 10–19. Both compare ordinary finite-gap optimization, completion, and a matched finite restart. The Gaussian study reruns the archived target with a whole-cluster transfer at fixed multiplicity, checked in the whole-line H¹ norm before refitting. Figures 1–2 are [analytical illustrations](docs/illustrations.md).

<p align="center">
  <img src="figures/previews/figure_08.png" alt="Poisson–Neumann target, finite and completed approximations, feature centres, and spatial errors." width="920">
</p>

*Figure 8. Finite and completed representations in the Poisson–Neumann experiment. Close pair markers in panel (b) are separated visually; connectors indicate the true centre locations.*

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

New outputs go into `outputs/`. The nine exact publication PDFs in [`figures/paper/`](figures/paper/) and their [browser previews](figures/previews/) remain available without running any code. The [reproduction guide](REPRODUCING.md) gives individual commands, the recorded environment, and instructions for fresh runs.

Figure 4 preserves a historical run: its original script, publication PDF and rounded logged states are included, but exact unrounded optimizer snapshots are unavailable. Its reproduction command restores the preserved figure; a fresh run is a new numerical realization of the same protocol.

## Reading the results

These are controlled experiments with prescribed targets and pairings. Stable evaluation, admitting limiting states, and finding them through optimization are separate issues. Completion keeps the Matérn coefficients bounded without uniformly improving accuracy; in the reported Poisson runs it improves accuracy on every start and reduces the median number of coefficient solves. The two-dimensional examples illustrate directional collisions, while the exact closure theorem is one-dimensional. The two-dimensional transfer checks use measured quadrature errors. The one-dimensional Gaussian experiment instead uses validated whole-line transfer and objective enclosures.

## Citation and license

Please cite **Singular Parameters and Missing Limits in Neural PDE Solvers**, by Daniel Fernández and Enrique Zuazua, and this repository when using these materials. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). The existing repository [license](LICENSE) applies.
