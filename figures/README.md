# Figures

The figure tree separates publication artifacts from browser previews and
supplementary results:

- `manuscript/` contains the exact, immutable `Fig1.pdf`–`Fig8.pdf` files used
  by the paper. These files are covered by `FIGURE_SHA256SUMS`.
- `gallery/` contains descriptive PNG previews for GitHub and visual
  inspection. Each preview links to its generating script and reference data
  in the repository README and provenance guide.
- `supplementary/` contains reported figures that are not among the eight
  numbered manuscript figures.

Generated audit plots are written to the ignored top-level `outputs/`
directory. This keeps a cached redraw from silently replacing the selected
publication PDFs. PDF bytes may vary across platforms because of fonts,
timestamps, and backend metadata even when the numerical content agrees.

Run `python reproduce.py --check` to validate the manuscript files, their
source/data provenance, expected page dimensions, and absence of Type 3 fonts.
