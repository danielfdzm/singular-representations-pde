# Reproducing the numerical experiments

The repository includes the numerical records needed to redraw the figures,
so the quickest reproduction does not repeat the long optimization runs. All
commands below are run from the repository root.

## 1. Create the recorded environment

Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
or Miniconda, then run:

```bash
git clone https://github.com/danielfdzm/singular-representations-pde.git
cd singular-representations-pde
conda env create -f environment.yml
conda activate singular-representations-pde
```

The environment contains Python 3.13.9, NumPy 2.4.2, Matplotlib 3.10.8,
and PyTorch 2.10.0.

## 2. Choose a reproduction level

### Validate the repository without producing files

```bash
python reproduce.py --check
```

This checks the required sources, cached numerical records, figure
provenance, file integrity, and recorded package versions.

### Redraw the figures from cached records

```bash
python reproduce.py --fast
```

This is the recommended first run. It redraws the figures from the cached
records under `outputs/`, then performs the same validation as `--check`.

### Repeat the matched endpoint controls

```bash
python reproduce.py --matched-controls
```

This reruns the one- and two-dimensional LBFGS endpoint comparisons without
repeating the preceding Adam trajectories. Results are written to
`outputs/matched-controls/`.

### Repeat every optimization trajectory

```bash
python reproduce.py --full
```

> **Long-run warning.** The full workflow includes two 100-million-step
> Gaussian trajectories, a one-million-step penalty continuation, three
> 100,000-step one-dimensional completion stages, and a 40,000-step
> two-dimensional run. It is intended for a dedicated machine, not as a first
> check of the repository. It also regenerates authoritative records under
> `data/`, so use a clean clone when comparing files with the archive.

If the installed numerical-library versions differ from the recorded ones,
the structural checks can still be run with:

```bash
python reproduce.py --check --allow-version-drift
```

This is useful for inspection, but floating-point trajectories obtained with
other versions are not expected to be byte-for-byte identical.

## 3. Where to find the results

- `outputs/tanh/`, `outputs/gaussian/`, `outputs/completion/`,
  `outputs/weak_pde/`, and `outputs/ingham/` contain newly rendered results.
- `data/tanh/`, `data/gaussian/`, `data/completion/`, `data/weak_pde/`, and
  `data/ingham/` contain the recorded histories and cached calculations.
- `figures/paper/Fig1.pdf` through `Fig8.pdf` are the exact publication
  figures. Reproduction commands do not overwrite them.
- `figures/previews/` contains browser-friendly versions for readers.

For the command associated with each figure, see
[`experiments/README.md`](experiments/README.md).

## 4. Reference platform

The archived calculations were made on CPU under macOS arm64 on an Apple M2.
The Gaussian mixed-precision run also requires a C compiler; the portable
source is [`native/tanh_second_derivative_optimizer.c`](native/tanh_second_derivative_optimizer.c),
and the compiled library is placed under `outputs/.build/`.

The experiments use fixed full-batch quadrature grids, with no minibatching or
random resampling. Numerically equivalent output is expected on other
platforms, although PDF bytes and the final digits of floating-point results
may differ because of fonts, backends, compiler choices, and arithmetic.
Further machine-readable platform information is stored in
[`data/execution_environment.json`](data/execution_environment.json).

## 5. Seeds and deterministic inputs

- The tanh bias-collision calculation records seed 0, uses explicit initial
  values, and has no stochastic sample.
- The tanh slope, Gaussian collision, Gaussian penalty, matched weak-form PDE,
  and Ingham calculations have no random input.
- The one-dimensional completion initializes completed coordinates with seed
  `70000 + width` and fixed controls with seed
  `80000 + 17*width + stage`.
- The two-dimensional completion uses seed 80209 for both translated and
  completed coordinates.

These runs are reproducible case studies, not averages over random seeds.

## 6. Verify the archived files

On macOS, verify all cached records and publication figures with:

```bash
shasum -a 256 -c data/SHA256SUMS
```

On Linux, the equivalent command is:

```bash
sha256sum -c data/SHA256SUMS
```
