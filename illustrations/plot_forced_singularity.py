"""Draw the fixed-depth tanh forced-singularity theorem as a schematic.

The two panels illustrate hidden-parameter behavior and strong convergence
of represented states. Coordinates and paths are chosen for readability;
they are not network parameters, training records, or a projection of data.
Run with ``python3 illustrations/plot_forced_singularity.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_style import (  # noqa: E402
    apply_style, save_figure, FULL_WIDTH, BLUE, RED, BLACK, PALE_GRAY,
)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as PlotPath  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "illustrations" / "forced_singularity"
FIGURE_HEIGHT = 3.0


def domain_path(start, segments) -> PlotPath:
    """A smooth illustrative domain with deliberately irregular geometry."""
    vertices = [start]
    codes = [PlotPath.MOVETO]
    for segment in segments:
        vertices.extend(segment)
        codes.extend([PlotPath.CURVE4] * 3)
    vertices.append(start)
    codes.append(PlotPath.CLOSEPOLY)
    return PlotPath(np.asarray(vertices, dtype=float), codes)


COMPACT_BOUNDARY = domain_path((0.056, 0.565), [
    [(0.012, 0.700), (0.030, 0.830), (0.160, 0.860)],
    [(0.270, 0.890), (0.300, 0.810), (0.405, 0.820)],
    [(0.510, 0.830), (0.540, 0.740), (0.508, 0.640)],
    [(0.476, 0.565), (0.556, 0.500), (0.502, 0.405)],
    [(0.461, 0.335), (0.364, 0.375), (0.285, 0.341)],
    [(0.162, 0.290), (0.105, 0.360), (0.092, 0.440)],
    [(0.080, 0.500), (0.079, 0.500), (0.056, 0.565)],
])

STATE_BOUNDARY = domain_path((0.860, 0.480), [
    [(0.880, 0.620), (0.820, 0.770), (0.690, 0.785)],
    [(0.590, 0.800), (0.540, 0.714), (0.450, 0.748)],
    [(0.320, 0.820), (0.250, 0.830), (0.140, 0.700)],
    [(0.070, 0.640), (0.055, 0.510), (0.100, 0.425)],
    [(0.155, 0.325), (0.120, 0.215), (0.260, 0.185)],
    [(0.390, 0.145), (0.460, 0.245), (0.570, 0.205)],
    [(0.720, 0.145), (0.820, 0.280), (0.860, 0.480)],
])


def cubic_points(control: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate a decorative cubic Bezier curve, without implying sampled data."""
    t = parameters[:, None]
    return ((1 - t) ** 3 * control[0] + 3 * (1 - t) ** 2 * t * control[1]
            + 3 * (1 - t) * t ** 2 * control[2] + t ** 3 * control[3])


def draw_route(axis, control: np.ndarray, color: str, fractions: list[float]) -> None:
    """Draw a schematic subsequence as small circles on an arrowed path."""
    path = PlotPath(control, [PlotPath.MOVETO, PlotPath.CURVE4,
                              PlotPath.CURVE4, PlotPath.CURVE4])
    axis.add_patch(FancyArrowPatch(path=path, arrowstyle="->", mutation_scale=8,
                                   color=color, lw=1.2, zorder=3))
    dots = cubic_points(control, np.array(fractions))
    axis.plot(dots[:, 0], dots[:, 1], "o", color=color, markersize=4.3,
              markeredgecolor="white", markeredgewidth=0.55, zorder=4)


def make_figure():
    fig = plt.figure(figsize=(FULL_WIDTH, FIGURE_HEIGHT))
    left = fig.add_axes([0.027, 0.17, 0.441, 0.705])
    right = fig.add_axes([0.555, 0.17, 0.424, 0.705])
    for axis in (left, right):
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.1, 1)
        axis.axis("off")

    fig.text(0.2475, 0.966, "(a) Hidden-parameter space", ha="center",
             va="top", fontsize=10, color=BLACK)
    fig.text(0.767, 0.966, "(b) State space", ha="center", va="top",
             fontsize=10, color=BLACK)

    # A compact nondegenerate set is eventually left by every missing-limit
    # sequence. The organic outline is schematic, not a claimed set geometry.
    left.add_patch(PathPatch(COMPACT_BOUNDARY, facecolor=PALE_GRAY,
                              edgecolor=BLACK, linestyle=(0, (3.5, 2.5)),
                              linewidth=0.95, zorder=0))
    left.text(0.27, 0.952, "Compact, nondegenerate", ha="center",
              va="center", fontsize=9)
    left.text(0.27, 0.695, r"$\Vert\mathbf{z}\Vert_2\leq R$", ha="center",
              va="center", fontsize=9, color=BLUE)
    left.text(0.27, 0.575, r"$\Delta_{\mathrm{deep}}(\mathbf{z})\geq s$",
              ha="center", va="center", fontsize=9, color=RED)

    left.plot([0.07, 0.93], [0.085, 0.085], color=RED, lw=1.3,
              solid_capstyle="round", zorder=1)
    left.text(0.5, -0.005, r"Degeneracy: $\Delta_{\mathrm{deep}}(\mathbf{z})=0$",
              ha="center", va="center", fontsize=9, color=BLACK)

    # The drawn routes are representative subsequences. They neither assert a
    # whole-sequence dichotomy nor exclude escape and degeneracy together.
    escape = np.array([[0.15, 0.418], [0.40, 0.400], [0.63, 0.665], [0.91, 0.817]])
    degenerate = np.array([[0.31, 0.431], [0.45, 0.388], [0.48, 0.180], [0.54, 0.099]])
    draw_route(left, escape, BLUE, [0.0, 0.23, 0.49, 0.70, 0.86])
    draw_route(left, degenerate, RED, [0.23, 0.48, 0.68, 0.85])
    left.text(0.745, 0.965, "Escape", ha="center", va="center", fontsize=9)
    left.text(0.745, 0.866, r"$\Vert\mathbf{z}_{n_k}\Vert_2\to\infty$",
              ha="center", va="center", fontsize=9)
    left.text(0.765, 0.420, "Bounded subsequence", ha="center", va="center", fontsize=9)
    left.text(0.765, 0.304, r"$\Delta_{\mathrm{deep}}(\mathbf{z}_{n_k})\to0$",
              ha="center", va="center", fontsize=9)
    left.text(0.765, 0.194, r"$\Vert c_{n_k}\Vert_2\to\infty$",
              ha="center", va="center", fontsize=9)

    # All blue circles lie strictly inside this illustrative realization class.
    # The excluded limit is on its drawn boundary, marked by an open circle.
    right.add_patch(PathPatch(STATE_BOUNDARY, facecolor=PALE_GRAY,
                              edgecolor=BLACK, linewidth=0.9, zorder=0))
    state_control = np.array([[0.23, 0.429], [0.40, 0.561],
                              [0.61, 0.524], [0.843, 0.482]])
    state_fractions = [0.0, 0.30, 0.57, 0.77, 0.88, 0.955]
    state_dots = cubic_points(state_control, np.array(state_fractions))
    assert np.all(STATE_BOUNDARY.contains_points(state_dots))
    assert np.all(STATE_BOUNDARY.contains_points(
        cubic_points(state_control, np.linspace(0, 1, 301))))
    draw_route(right, state_control, BLUE, state_fractions)
    right.plot(0.86, 0.48, "o", markersize=7.0, markerfacecolor="white",
               markeredgecolor=BLUE, markeredgewidth=1.1, zorder=6)
    right.text(0.901, 0.492, r"$u\notin\mathcal{M}$", ha="left", va="center",
               fontsize=11)
    right.text(0.47, 0.671, r"$u_{\theta_n}\longrightarrow u$", ha="center",
               va="center", fontsize=11)
    right.text(0.47, 0.286, r"$\mathcal{M}=T(\Theta)$", ha="center", va="center",
               fontsize=10)
    right.text(0.47, 0.927, r"Strong convergence in $H^m(\Omega)$",
               ha="center", va="center", fontsize=9)

    fig.add_artist(FancyArrowPatch((0.485, 0.536), (0.541, 0.536),
                                   transform=fig.transFigure, arrowstyle="->",
                                   mutation_scale=9, color=BLACK, lw=0.8))
    return fig, state_dots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output stem for PDF, SVG, PNG and metadata JSON")
    args = parser.parse_args()
    figure, state_dots = make_figure()
    outputs = save_figure(figure, args.output)
    metadata = {
        "type": "conceptual schematic, not experimental data",
        "result": "Fixed-architecture deep tanh forced singularity",
        "manuscript_sources": [
            "deep_tanh_theorem.tex", "deep_tanh_proofs.tex", "preliminaries.tex",
        ],
        "formulas": {
            "regular_region": "{z: ||z||_2 <= R, Delta_deep(z) >= s}",
            "degeneracy": "Delta_deep(z) = 0",
            "realization_class": "M = T(Theta)",
            "state_limit": "u_theta_n -> u strongly in H^m(Omega), u not in M",
            "bounded_subsequences": "Delta_deep(z_nk) -> 0 and ||c_nk||_2 -> infinity",
            "global_statement": "min{1/(1+||z_n||_2), Delta_deep(z_n)} -> 0",
        },
        "notes": [
            "Coordinates, set boundaries and routes are purely illustrative.",
            "Every compact nondegenerate region with fixed R and s is visited only finitely often.",
            "The two paths depict subsequence behaviors, not exclusive whole-sequence alternatives.",
            "An unbounded sequence can have bounded and escaping subsequences.",
            "Escape and degeneracy can also occur together.",
            "All blue state markers lie strictly inside the drawn realization class.",
            "The open circle is the excluded regular limit on its schematic boundary.",
            "No full closure description for deep tanh networks is asserted.",
        ],
        "style_source": "illustrations/plot_style.py",
        "figure_inches": [FULL_WIDTH, FIGURE_HEIGHT],
        "colors": {"ordinary_paths": BLUE, "degeneracy": RED,
                   "norm_bound_label": BLUE, "nondegeneracy_bound_label": RED,
                   "compact_domain_contour": BLACK,
                   "state_domain_contour": BLACK, "other_annotations": BLACK},
        "domain_design": {
            "compact_region": "Smooth arbitrary-domain outline; black dashed contour",
            "realization_class": "Smooth arbitrary-domain outline; black contour",
            "compact_boundary_vertices": COMPACT_BOUNDARY.vertices.tolist(),
            "state_boundary_vertices": STATE_BOUNDARY.vertices.tolist(),
            "boundary_path_codes": STATE_BOUNDARY.codes.tolist(),
            "removed_labels": ["K_{R,s}", "T(c,z)", "A regular state with no finite representation",
                               "Schematic: the two subsequence behaviors can coexist."],
            "state_marker_label": "u \\notin \\mathcal{M}",
            "state_markers_and_path_strictly_inside": True,
            "excluded_limit_on_boundary": [0.86, 0.48],
        },
        "base_font_size_pt": 9.0,
        "png_dpi": 600,
        "schematic_state_marker_coordinates": state_dots.tolist(),
        "outputs": outputs,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(figure)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
