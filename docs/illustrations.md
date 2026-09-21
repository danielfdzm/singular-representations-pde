# Analytical illustrations: Figures 1–3

These scripts reconstruct the three illustrations accompanying the paper's
theory. They require NumPy 2.0 or later and Matplotlib; no training records,
PyTorch, or external datasets are needed.

| Paper figure | Illustration | Script | Published PDF |
| --- | --- | --- | --- |
| 1 | Two tanh neurons approach a derivative as their biases collide | [`plot_tanh_limit.py`](../illustrations/plot_tanh_limit.py) | [Figure 1](../figures/paper/figure_01.pdf) |
| 2 | Necessary escape or degeneracy in fixed-architecture deep tanh networks | [`plot_forced_singularity.py`](../illustrations/plot_forced_singularity.py) | [Figure 2](../figures/paper/figure_02.pdf) |
| 3 | Three Gaussian translates approach a completed derivative block | [`plot_completion_patterns.py`](../illustrations/plot_completion_patterns.py) | [Figure 3](../figures/paper/figure_03.pdf) |

Run these commands from the repository root:

```sh
python illustrations/plot_tanh_limit.py
python illustrations/plot_forced_singularity.py
python illustrations/plot_completion_patterns.py
```

Each script writes a PDF, SVG, 600-dpi PNG, and JSON metadata file under
`outputs/illustrations/`. The published assets in `figures/paper/` are preserved.
To choose another output stem, use `--output`, for example:

```sh
python illustrations/plot_tanh_limit.py --output outputs/custom/tanh_singular_limit
python illustrations/plot_forced_singularity.py --output outputs/custom/forced_singularity
python illustrations/plot_completion_patterns.py --output outputs/custom/completion_patterns
```

The scripts also work when invoked by absolute path from another working
directory. They share a local [`plot_style.py`](../illustrations/plot_style.py)
that uses Matplotlib's headless backend. Matching the published typography
requires the Latin Modern Roman font; Matplotlib uses its installed serif
fallback otherwise.

Figure 1 evaluates the prescribed tanh difference quotient directly. Figure 2
is a schematic: its paths and coordinates are illustrative, not network
trajectories. Figure 3 computes least-squares fits at fixed centres; it is not
an optimization trajectory. Its completed block has coefficients `(1, 1, 1)`
up to floating-point roundoff. The metadata records the formulas, plotted
settings, and calculated quantities.

The packaged scripts were checked with NumPy 2.3.5 and Matplotlib 3.10.7.
Their regenerated PNGs matched the corresponding publication-source PNGs
pixel for pixel, and their mathematical metadata was unchanged.
