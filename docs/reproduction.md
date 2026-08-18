# Reproduction notes

## Validation tiers

`python reproduce.py --check` is read-only. It checks the curated source and
cache files, JSON schemas, source hashes, exact figure aliases, PDF media boxes,
absence of Type 3 fonts, and recorded direct package versions.

`python reproduce.py --fast` redraws deterministic or cache-backed plots,
restores the selected publication PDFs, updates the numbered figure
aliases, and finishes with the same validation.

`python reproduce.py --matched-controls` reruns the one- and two-dimensional
matched LBFGS endpoint audits without repeating the Adam trajectories. Results
are written to the ignored `reproduced/` directory, then the committed caches
and their checksums are validated. This isolates the nondeterministic measured
wall-clock field from the reference artifact.

`python reproduce.py --full` reruns all optimizer trajectories. It is a manual,
long-running workflow and is not suitable for pull-request CI. It validates
regenerated contents without comparing them byte-for-byte with the committed
reference caches. Full reproduction replaces authoritative caches in place and
should therefore be run in a clean clone or dedicated worktree.

## Reference environment

- Python 3.13.9
- NumPy 2.4.2
- Matplotlib 3.10.8
- PyTorch 2.10.0
- CPU execution
- macOS arm64 on an Apple M2 reference machine
- a C compiler for the mixed-precision Gaussian trajectory

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

The committed checksum files cover every authoritative numerical cache and all
eight manuscript PDFs. On macOS:

```bash
shasum -a 256 -c DATA_SHA256SUMS
shasum -a 256 -c FIGURE_SHA256SUMS
```

On Linux, replace `shasum -a 256` with `sha256sum`.
