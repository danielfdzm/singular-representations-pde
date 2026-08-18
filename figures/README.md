# Figures

- `manuscript/` contains the exact numbered PDFs included in the paper.
- `descriptive/` contains named PDF and PNG copies that make provenance and
  visual inspection easier.
- `preserved/` contains the exact frozen publication PDFs for Figures 1--8.

The preserved files are intentional reproducibility inputs, not unexplained
duplicates. Cache redraws may use a newer layout or embed nondeterministic PDF
metadata; the orchestrator restores the selected manuscript artwork before it
updates the numbered aliases. Figure 8 is computed directly, but its selected
PDF is also preserved so embedded metadata cannot change publication bytes.
