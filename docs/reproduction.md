# Reproduction guide

All commands are run from the repository root. The stable entry point is
`python reproduce.py`; it delegates to the versioned orchestrator in `tools/`.

## Installation

The recorded environment is available through Conda:

```bash
conda env create -f environment.yml
conda activate singular-representations-pde
```

Alternatively, install the exact direct Python dependencies with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Linux x86-64, installing the CPU-only PyTorch wheel first avoids pulling the
CUDA/NVIDIA runtime from the default PyPI wheel:

```bash
python -m pip install "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
```

The Conda file records the macOS reference environment. The Linux commands are
the lean CPU-only equivalent used by continuous integration.

## Validation tiers

`python reproduce.py --check` is read-only. It checks the curated source and
reference-data files, JSON schemas, source hashes, manuscript PDFs, PDF media
boxes, absence of Type 3 fonts, checksum manifests, and recorded direct package
versions.

`python reproduce.py --fast` redraws deterministic or cache-backed plots under
the ignored `outputs/` tree and finishes with the same validation. It does not
modify the immutable manuscript PDFs or the tracked gallery previews.

`python reproduce.py --matched-controls` reruns the one- and two-dimensional
matched LBFGS endpoint audits without repeating the Adam trajectories. Results
are written to `outputs/matched-controls/`, isolating nondeterministic measured
wall-clock fields from the checksummed reference records.

`python reproduce.py --full` reruns every optimizer trajectory. It is a manual,
long-running workflow and is not suitable for pull-request CI. Full
reproduction regenerates authoritative files under `data/reference/`, so run it
in a clean clone or dedicated worktree when byte-for-byte comparison matters.
All rendered audit products remain under `outputs/`.

To validate structure and provenance while using other dependency versions,
append `--allow-version-drift`. Version drift can change floating-point
trajectories and is not equivalent to reproducing the recorded environment.

## Reference environment

- Python 3.13.9
- NumPy 2.4.2
- Matplotlib 3.10.8
- PyTorch 2.10.0
- CPU execution
- macOS arm64 on an Apple M2 reference machine
- a C compiler for the mixed-precision Gaussian trajectory

The native source is `native/tanh_second_derivative_optimizer.c`; compiled
libraries are written below `outputs/` and are never committed. Set `CC` if
compiler discovery does not find a suitable C compiler.

The drivers use full-batch deterministic grids. The orchestrator selects the
noninteractive Matplotlib backend and a temporary writable Matplotlib cache.
Numerically equivalent output on another platform is expected, but PDF bytes
can differ through embedded timestamps, fonts, or backend metadata.

## Seeds

- Tanh offset: seed 0, with explicit initialization and no stochastic sample
- Tanh slope, matched PDE, Gaussian collision/penalty, and Ingham: no random input
- One-dimensional completion: `70000 + width` for completed initialization;
  `80000 + 17*width + stage` for fixed controls
- Two-dimensional completion: seed 80209 shared by translated and completed runs

## Integrity

The committed checksum files cover all 13 authoritative numerical records and
all eight manuscript PDFs. On macOS:

```bash
shasum -a 256 -c DATA_SHA256SUMS
shasum -a 256 -c FIGURE_SHA256SUMS
```

On Linux, replace `shasum -a 256` with `sha256sum`.
