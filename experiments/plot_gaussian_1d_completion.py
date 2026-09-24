#!/usr/bin/env python3
"""Plot the fixed-budget Gaussian experiment as publication Figure 5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plot_style import (apply_style, save_figure, FULL_WIDTH, BLUE, RED, GREEN,
                        BLACK, GRAY, LIGHT_GRAY)

apply_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/completion_consistent/results.json"
GRID = np.linspace(-6.0, 6.0, 4001)


def atom(x: np.ndarray, order: int, center: float) -> np.ndarray:
    z = x - center
    coefficients = np.zeros(order + 1)
    coefficients[-1] = 1.0
    return (-1.0)**order * np.polynomial.hermite_e.hermeval(z, coefficients) * np.exp(-z*z/2)


def target(x: np.ndarray) -> np.ndarray:
    return atom(x, 3, 0.0) + 0.6 * atom(x, 0, 1.4)


def multiplicity(snapshot: dict) -> int:
    orders = {}
    for order, center in snapshot["basis"]:
        orders[center] = max(orders.get(center, -1), int(order))
    return sum(order + 1 for order in orders.values())


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(LIGHT_GRAY)
    axis.spines[["left", "bottom"]].set_linewidth(.65)
    axis.tick_params(axis="both", color=LIGHT_GRAY, labelcolor=BLACK,
                     width=.65, labelsize=8)


def make_figure(snapshots: list[dict]):
    # Match the layout of the archived Gaussian collision illustration.
    fig = plt.figure(figsize=(FULL_WIDTH, 4.5))
    lefts, width = (.082, .395, .708), .252
    titles = ("(a) Initial configuration", "(b) Centers nearly collided", "(c) The completed block")
    for index, (left, snapshot) in enumerate(zip(lefts, snapshots, strict=True)):
        fig.text(left + width/2, .985, titles[index], ha="center", va="top", fontsize=9.5)
        upper = fig.add_axes([left, .615, width, .22])
        contributions = []
        for (order, center), weight in zip(snapshot["basis"], snapshot["weights"], strict=True):
            values = weight * atom(GRID, int(order), center)
            contributions.append(values)
            color = BLUE if order == 0 else RED if order <= 3 else GREEN
            upper.plot(GRID, values, color=color, lw=1.3, alpha=.9,
                       solid_capstyle="round")
        amplitude = max(float(np.max(np.abs(contributions))), 1.0)
        upper.set_ylim(-1.2*amplitude, 1.2*amplitude)
        upper.axhline(0, color=LIGHT_GRAY, lw=.6, zorder=0)
        for center in set(center for _, center in snapshot["basis"]):
            upper.plot(center, -1.2*amplitude*1.07, "^", color=GRAY,
                       markersize=4, clip_on=False, zorder=5)
        upper.set_xlim(-5, 5)
        upper.set_xticks([])
        if index == 0:
            upper.set_ylabel("weighted\nfeatures", labelpad=3, fontsize=8.5)
        style_axis(upper)

        lower = fig.add_axes([left, .17, width, .22])
        state = np.sum(contributions, axis=0)
        lower.fill_between(GRID, target(GRID), state, color=RED, alpha=.18, lw=0)
        lower.plot(GRID, target(GRID), color=BLACK, lw=1.7, zorder=3,
                   solid_capstyle="round")
        lower.plot(GRID, state, color=RED, lw=1.4, ls="--", dashes=(4, 2.4), zorder=4)
        lower.axhline(0, color=LIGHT_GRAY, lw=.6, zorder=0)
        lower.set_xlim(-5, 5)
        lower.set_ylim(-1.7, 2.15)
        lower.set_xticks([-4, 0, 4])
        lower.set_xlabel(r"$x$", labelpad=.5)
        if index == 0:
            lower.set_ylabel("state", labelpad=3, fontsize=8.5)
        style_axis(lower)

    for start, end in ((.290, .423), (.603, .736)):
        fig.add_artist(FancyArrowPatch((start, .912), (end, .912),
                       transform=fig.transFigure, arrowstyle="->",
                       mutation_scale=8, lw=.75, color=BLACK))
    fig.text(.3565, .903, "centers approach", ha="center", va="top", fontsize=7.5, color=BLACK)
    fig.text(.6695, .903, "completion and refit", ha="center", va="top", fontsize=7.5, color=BLACK)
    feature_handles = [
        Line2D([], [], color=BLUE, lw=1.3, label=r"$wK$"),
        Line2D([], [], color=RED, lw=1.3, label=r"$wK^{(r)}$, $1\leq r\leq3$"),
        Line2D([], [], color=GREEN, lw=1.3, label=r"$wK^{(r)}$, $4\leq r\leq7$"),
    ]
    fig.legend(handles=feature_handles, loc="center", bbox_to_anchor=(.52, .505),
               ncol=3, columnspacing=2, handlelength=2.4, fontsize=8)
    handles = [Line2D([], [], color=BLACK, lw=1.7,
                     label=r"target $u^*=K'''+0.6K(\cdot-1.4)$"),
               Line2D([], [], color=RED, lw=1.4, ls="--", label="model state")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .002),
               ncol=2, columnspacing=2, handlelength=2.4, fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures/figure_05")
    parser.add_argument("--width", type=int, choices=(9, 10, 12), default=9)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    run = next(item for item in payload["runs"] if item["initial_width"] == args.width)
    snapshots = run["snapshots"]
    if len(snapshots) != 3:
        raise ValueError("Expected initial, pre-transfer, and completed snapshots")
    if any(multiplicity(snapshot) != args.width for snapshot in snapshots):
        raise ValueError("A snapshot violates the fixed multiplicity budget")
    figure = make_figure(snapshots)
    outputs = save_figure(figure, args.output)
    metadata = {
        "source": str(args.input), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "initial_width": args.width, "snapshot_multiplicities": [multiplicity(s) for s in snapshots],
        "style_source": "illustrations/plot_completion_patterns.py (archived illustration)",
        "initial_coefficient_step_fraction": snapshots[0]["coefficient_step_fraction"],
        "upper_scales": "Separate symmetric vertical scales show each representation's features.",
        "outputs": outputs,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(figure)
    print(f"Wrote Figure 5 with three fixed-{args.width}-slot snapshots: {args.output}.pdf")


if __name__ == "__main__":
    main()
