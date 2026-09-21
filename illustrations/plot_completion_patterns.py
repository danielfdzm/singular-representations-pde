"""Illustrate a collision as it would occur during optimization, at P = 3.

Three Gaussian translates are fitted by least squares to the target
u* = K + K' + K'', a state that lies on the boundary of the width-three
translate model: it is approached as the centers coalesce but represented
exactly only by the completed block. Panels (a) and (b) are genuine
least-squares fits at fixed centers; panel (c) is the completed block, whose
coefficients are exactly (1, 1, 1). Run with
``python3 illustrations/plot_completion_patterns.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_style import (  # noqa: E402
    apply_style, save_figure, FULL_WIDTH, BLUE, RED, GREEN, BLACK, GRAY, LIGHT_GRAY,
)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "illustrations" / "completion_patterns"
ATOM_COLORS = (BLUE, RED, GREEN)
FIGURE_HEIGHT = 4.5
GRID = np.linspace(-6.0, 6.0, 4001)
UPPER_YLIM = (-12.0, 12.0)


def gaussian_derivative(x: np.ndarray, order: int) -> np.ndarray:
    """Return K^(order)(x) for the unit Gaussian K(x) = exp(-x^2/2)."""
    gaussian = np.exp(-0.5 * x**2)
    return (gaussian, -x * gaussian, (x**2 - 1.0) * gaussian)[order]


def target(x: np.ndarray) -> np.ndarray:
    """u* = K + K' + K'', the multiplicity-three block at the origin."""
    return sum(gaussian_derivative(x, order) for order in range(3))


def least_squares_fit(columns: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Fit the columns to the target in L2 and return coefficients and relative error."""
    matrix = np.stack(columns, axis=1)
    rhs = target(GRID)
    weight = np.sqrt(np.gradient(GRID))[:, None]
    coefficients, *_ = np.linalg.lstsq(matrix * weight, rhs * weight[:, 0], rcond=None)
    residual = matrix @ coefficients - rhs
    relative = np.sqrt(np.trapezoid(residual**2, GRID) / np.trapezoid(rhs**2, GRID))
    return coefficients, float(relative)


def panel_data() -> list[dict]:
    """Two fitted translate configurations and the completed block they approach."""
    panels = []
    for centers in ([-1.6, 0.0, 1.6], [-0.44, 0.0, 0.44]):
        columns = [gaussian_derivative(GRID - b, 0) for b in centers]
        coefficients, relative = least_squares_fit(columns)
        panels.append({"centers": centers, "orders": [0, 0, 0], "columns": columns,
                       "coefficients": coefficients, "relative": relative})
    columns = [gaussian_derivative(GRID, r) for r in range(3)]
    coefficients, relative = least_squares_fit(columns)
    panels.append({"centers": [0.0, 0.0, 0.0], "orders": [0, 1, 2], "columns": columns,
                   "coefficients": coefficients, "relative": relative})
    return panels


def make_figure(panels: list[dict]):
    fig = plt.figure(figsize=(FULL_WIDTH, FIGURE_HEIGHT))
    lefts = (0.082, 0.395, 0.708)
    width = 0.252
    titles = ("(a) Separated centers", "(b) Centers nearly collided",
              "(c) The completed block")
    for index, (left, panel) in enumerate(zip(lefts, panels)):
        middle = left + width / 2
        fig.text(middle, 0.985, titles[index], ha="center", va="top", fontsize=9.5)

        # Upper row: the weighted contributions the model actually carries.
        upper = fig.add_axes([left, 0.615, width, 0.22])
        for color, coefficient, column in zip(ATOM_COLORS, panel["coefficients"], panel["columns"]):
            upper.plot(GRID, coefficient * column, color=color, lw=1.3, alpha=0.9,
                       solid_capstyle="round")
        upper.axhline(0, color=LIGHT_GRAY, lw=0.6, zorder=0)
        for center in set(panel["centers"]):
            upper.plot(center, UPPER_YLIM[0] * 1.07, "^", color=GRAY, markersize=4,
                       clip_on=False, zorder=5)
        upper.set_xlim(-5.0, 5.0)
        upper.set_ylim(*UPPER_YLIM)
        upper.set_xticks([])
        if index == 0:
            upper.set_ylabel("weighted\nfeatures", labelpad=3, fontsize=8.5)
        style_axis(upper)

        # Lower row: what the model produces, against the target it is chasing.
        lower = fig.add_axes([left, 0.17, width, 0.22])
        total = sum(c * col for c, col in zip(panel["coefficients"], panel["columns"]))
        lower.fill_between(GRID, target(GRID), total, color=RED, alpha=0.18, lw=0)
        lower.plot(GRID, target(GRID), color=BLACK, lw=1.7, zorder=3,
                   solid_capstyle="round")
        lower.plot(GRID, total, color=RED, lw=1.4, ls="--", dashes=(4, 2.4), zorder=4)
        lower.axhline(0, color=LIGHT_GRAY, lw=0.6, zorder=0)
        lower.set_xlim(-5.0, 5.0)
        lower.set_ylim(-0.75, 1.62)
        lower.set_xticks([-4, 0, 4])
        lower.set_xlabel(r"$x$", labelpad=0.5)
        if index == 0:
            lower.set_ylabel("state", labelpad=3, fontsize=8.5)
        style_axis(lower)

    for start, end in ((0.290, 0.423), (0.603, 0.736)):
        fig.add_artist(FancyArrowPatch((start, 0.912), (end, 0.912),
                                       transform=fig.transFigure, arrowstyle="->",
                                       mutation_scale=8, lw=0.75, color=BLACK))
    fig.text(0.3565, 0.903, "centers approach", ha="center", va="top",
             fontsize=7.5, color=BLACK)
    fig.text(0.6695, 0.903, "collision limit", ha="center", va="top",
             fontsize=7.5, color=BLACK)

    feature_labels = [
        r"$w_1K(\cdot-b_1)$ / $K$",
        r"$w_2K(\cdot-b_2)$ / $K'$",
        r"$w_3K(\cdot-b_3)$ / $K''$",
    ]
    feature_handles = [
        Line2D([0], [0], color=color, lw=1.3, alpha=0.9, label=label)
        for color, label in zip(ATOM_COLORS, feature_labels)
    ]
    fig.legend(handles=feature_handles, loc="center",
               bbox_to_anchor=(0.52, 0.505), ncol=3, columnspacing=2.0,
               handlelength=2.4, fontsize=8)

    handles = [
        Line2D([0], [0], color=BLACK, lw=1.7, label=r"target $u^*=K+K'+K''$"),
        Line2D([0], [0], color=RED, lw=1.4, ls="--", label="model state"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.002),
               ncol=2, columnspacing=2.0, handlelength=2.4, fontsize=8)
    return fig


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(LIGHT_GRAY)
    axis.spines[["left", "bottom"]].set_linewidth(0.65)
    axis.tick_params(axis="both", color=LIGHT_GRAY, labelcolor=BLACK, width=0.65,
                     labelsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    panels = panel_data()
    figure = make_figure(panels)
    outputs = save_figure(figure, args.output)
    metadata = {
        "type": "least-squares illustration of a collision at fixed width",
        "width_budget": 3,
        "kernel": "K(x) = exp(-x^2/2)",
        "target": "u* = K + K' + K'', the multiplicity-three block at the origin",
        "fit": "L2 least squares on x in [-6, 6], 4001 samples, trapezoidal weights",
        "panels": [
            {"centers": p["centers"], "derivative_orders": p["orders"],
             "coefficients": [float(c) for c in p["coefficients"]],
             "max_abs_coefficient": float(np.abs(p["coefficients"]).max()),
             "relative_L2_error": p["relative"]}
            for p in panels
        ],
        "notes": [
            "Panels (a) and (b) are genuine least-squares fits at fixed centers.",
            "Panel (c) is the completed block; its coefficients are exactly (1, 1, 1).",
            "As the centers coalesce the coefficients grow with alternating signs "
            "while the represented state converges to the target.",
            "The target lies in the closure of the width-three translate model but "
            "not in the model itself.",
        ],
        "style_source": "illustrations/plot_style.py",
        "figure_inches": [FULL_WIDTH, FIGURE_HEIGHT],
        "upper_row_ylim": list(UPPER_YLIM),
        "legend_layout": {
            "upper_row": "Three colored entries in one shared horizontal legend beneath the upper row",
            "lower_row": "Shared target and model-state legend beneath the lower row",
            "completed_features": {"K": BLUE, "K'": RED, "K''": GREEN},
        },
        "outputs": outputs,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(figure)
    print(json.dumps(metadata["panels"], indent=2))


if __name__ == "__main__":
    main()
