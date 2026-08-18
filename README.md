# Singular representations in neural PDE models

Reproducible experiment and plotting code for the article *Stable Solutions
and Singular Representations in Neural PDE Models* by Daniel Fernández and
Enrique Zuazua.

The experiments study how stable represented states can coexist with singular
parameter sequences in neural PDE trial classes. They cover feature collision,
parameter escape, loss of conditioning, weight penalization, and completion by
derivative atoms. The repository contains the exact scripts, authoritative raw
histories, and manuscript-facing figures used for the numerical results.

## Scope and interpretation

The completion experiments are controlled mechanism studies. Their collision
directions, trees, and derivative orders are prescribed in advance. They do
not implement automatic collision discovery, prove generic optimizer
superiority, or establish an optimizer-convergence theorem. The exact
confluent-closure result in the article is one-dimensional; the two-dimensional
experiment is a directional embedding of the same mechanism.

## Quick start

Create the recorded Conda environment:

```bash
conda env create -f environment.yml
conda activate coercivity-artifact
python reproduce.py --check
```

Or install the exact direct Python dependencies with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python reproduce.py --check
```

On Linux x86-64, install the CPU-only PyTorch wheel first to avoid pulling the
CUDA/NVIDIA runtime used by the default PyPI wheel:

```bash
python -m pip install "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
```

The Conda file records the macOS reference environment; the Linux commands
above are the lean CPU-only equivalent used by continuous integration.

The strict check enforces the recorded NumPy, Matplotlib, and PyTorch versions.
For a structural cache/provenance check in a different environment, use:

```bash
python reproduce.py --check --allow-version-drift
```

## Reproduction modes

```bash
# Read-only validation of scripts, caches, figures, hashes, and schemas
python reproduce.py --check

# Redraw deterministic and cached figures, then validate
python reproduce.py --fast

# Rerun the cached-state matched LBFGS endpoint controls
python reproduce.py --matched-controls

# Rerun every optimizer trajectory used in the numerical section
python reproduce.py --full
```

Matched-control reruns are written to the ignored `reproduced/` directory, so
the checksummed reference caches remain unchanged. The full mode deliberately
regenerates authoritative caches in place; run it in a clean clone or worktree.

`--full` is intentionally not a routine CI command. It includes two
100-million-step Gaussian trajectories, a one-million-step penalty
continuation, three 100,000-step one-dimensional completion runs, and a
40,000-step two-dimensional run. The Gaussian trajectory also compiles a small
C optimizer locally; set `CC` if compiler discovery does not find a suitable C
compiler.

Figures 1--7 retain the wide layouts selected for the manuscript. Frozen
publication artwork for all eight figures is kept under `figures/preserved/`.
A fast redraw audits the scripts and caches, then restores those exact PDFs
before synchronizing `figures/manuscript/Fig1.pdf` through `Fig8.pdf`. Audit
renderings remain as ignored files in `experiments/`, so this mode leaves the
tracked artifact unchanged.

## Repository layout

```text
.
├── experiments/             experiment, plotting, and validation sources
├── figures/
│   ├── manuscript/          exact Fig1.pdf--Fig8.pdf files used by the paper
│   ├── descriptive/         named PDF/PNG outputs for inspection
│   └── preserved/           exact publication artwork for Figures 1--8
├── docs/                    provenance, reproduction, and limitations
├── reproduce.py             repository-level entry point
├── environment.yml          recorded Conda environment
├── requirements-lock.txt    exact direct Python package versions
├── execution_environment.json
├── DATA_SHA256SUMS           authoritative cache checksums
└── FIGURE_SHA256SUMS         manuscript-figure checksums
```

The experiment directory is deliberately flat: several standalone drivers
import neighboring helper modules, and keeping them together preserves the
tested command-line entry points without an unnecessary package refactor.

## Figure provenance

| Figure | Mechanism | Primary source |
|---|---|---|
| Fig. 1 | tanh offset collision and parameter escape | `tanh_parameter_escape_loss.py`, `tanh_wb_trajectories.py` |
| Fig. 2 | tanh slope collision | `tanh_slope_collision_snapshots.py` |
| Fig. 3 | Gaussian center collision and Gram conditioning | `gaussian_rbf_instability_figure5_mixed_precision.py`, `gaussian_rbf_trace_diagnostics.py` |
| Fig. 4 | adaptive weight penalty | `gaussian_instability_adaptive_weight_penalty.py` |
| Figs. 5--6 | one-dimensional derivative completion | `boundary_completion_experiment.py` |
| Fig. 7 | directional two-dimensional completion | `boundary_completion_2d_experiment.py` |
| Fig. 8 | Ingham lower-bound illustration | `ingham_illustration.py` |

See [docs/figure-provenance.md](docs/figure-provenance.md) for exact commands,
data dependencies, and output names.

## Data and environment

All numerical inputs needed for cached validation and plotting are committed;
no external dataset download is required. JSON is the authoritative raw format.
The `.npz` file is a deterministic order-one Gaussian profile cache. Verify the
curated inputs with:

```bash
shasum -a 256 -c DATA_SHA256SUMS
shasum -a 256 -c FIGURE_SHA256SUMS
```

The reference run used Python 3.13.9, NumPy 2.4.2, Matplotlib 3.10.8, and
PyTorch 2.10.0 on CPU. Hardware, threading, and seed details are recorded in
`execution_environment.json` and summarized in
[docs/reproduction.md](docs/reproduction.md).

## Citation

Please cite the associated article and this software artifact. Machine-readable
metadata are provided in [CITATION.cff](CITATION.cff). No article or artifact
DOI is invented here; permanent identifiers can be added once assigned.

## License status

No open-source license has yet been selected by the authors. The current
[LICENSE](LICENSE) records the default copyright status and grants no reuse
license. This should be replaced with an author-approved license before a
permanent archival release intended for reuse.
