"""Draw the prescribed tanh collision family from the motivating example.

This is an analytic illustration of the manuscript's exact approximation
family, not a training experiment. Run from any working directory; the
default outputs are in the repository's ``outputs/illustrations/`` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
from plot_style import (  # noqa: E402
    apply_style, save_figure, FULL_WIDTH, BLUE, RED, GREEN, GRAY, BLACK,
    LIGHT_GRAY,
)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

EPSILONS = (1.0, 0.5, 0.25)
COLORS = (BLUE, RED, GREEN)


def draw_collision(axis) -> None:
    """A compact diagram of additive biases, with tanh-shaped neuron icons."""
    # Match the physical panel box while keeping data lengths isotropic, so
    # the neuron circles stay round rather than shrinking the whole panel.
    box = axis.get_position()
    width, height = axis.figure.get_size_inches()
    y_span = 2.56 * (box.height * height) / (box.width * width)
    lower = 1.54 - y_span
    axis.set_xlim(-1.28, 1.28)
    axis.set_ylim(lower, 1.54)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.set_title("(a) Two colliding neurons", fontsize=9, pad=8)

    axis.text(0, 1.35, "output weights", ha="center", va="center",
              fontsize=8, color=BLACK)
    for location, color, coefficient in (
        (-0.80, RED, r"$-\frac{1}{2\varepsilon}$"),
        (0.80, BLUE, r"$+\frac{1}{2\varepsilon}$"),
    ):
        axis.text(location, 1.01, coefficient, ha="center", va="center",
                  color=color, fontsize=12)
        axis.plot([location, location], [0.81, 0.64], color=color, lw=1.1)

    # Bias locations are additive tanh arguments, not spatial transition points.
    baseline = lower + 0.10
    axis.annotate("", xy=(1.26, baseline), xytext=(-1.23, baseline),
                  arrowprops={"arrowstyle": "->", "color": LIGHT_GRAY,
                              "lw": 1.0}, zorder=0)
    axis.text(1.22, baseline-0.23, r"$b$", ha="center", va="center", color=BLACK)
    for location, color, bias in (
        (-0.80, RED, r"$-\varepsilon$"),
        (0.80, BLUE, r"$+\varepsilon$"),
    ):
        axis.plot([location, location], [baseline, -0.04], color=color,
                  lw=0.9, ls=(0, (2, 2)), alpha=0.65)
        axis.add_patch(Circle((location, 0.30), 0.34, facecolor=color,
                              edgecolor="none", alpha=0.065, zorder=2))
        axis.add_patch(Circle((location, 0.30), 0.34, facecolor="none",
                              edgecolor=color, lw=1.7, zorder=3))
        icon_x = np.linspace(-0.22, 0.22, 81)
        axis.plot(location + icon_x, 0.38 + 0.12*np.tanh(12*icon_x),
                  color=color, lw=1.7, solid_capstyle="round", zorder=4)
        axis.text(location, 0.15, r"$\tanh$", ha="center", va="center",
                  fontsize=8, color=BLACK, zorder=4)
        axis.plot([location, location], [baseline-0.05, baseline+0.05], color=color, lw=1.1)
        axis.text(location, baseline-0.23, bias, ha="center", va="center",
                  fontsize=9, color=BLACK)
        axis.annotate("", xy=(np.sign(location) * 0.07, 0.30),
                      xytext=(np.sign(location) * 0.42, 0.30),
                      arrowprops={"arrowstyle": "->", "color": color,
                                  "lw": 1.6, "mutation_scale": 8})

    axis.plot([0, 0], [baseline-0.05, baseline+0.05], color=BLACK, lw=0.8)
    axis.text(0, baseline-0.23, r"$0$", ha="center", va="center", fontsize=8)
    axis.text(0, 0.64, r"$\varepsilon\downarrow0$", ha="center", va="center",
              fontsize=9, color=BLACK)


def draw_contributions(axis) -> dict:
    """Show the two exact opposing terms on a common axis."""
    x = np.linspace(-3, 3, 1201)
    epsilon = 0.1
    positive = np.tanh(x+epsilon)/(2*epsilon)
    negative = -np.tanh(x-epsilon)/(2*epsilon)
    state = positive+negative
    exact_family = (np.tanh(x+epsilon)-np.tanh(x-epsilon))/(2*epsilon)
    assert np.max(np.abs(state-exact_family)) < 2e-15
    axis.axhline(0, color=LIGHT_GRAY, lw=0.65, zorder=0)
    axis.plot(x, positive, color=BLUE, lw=1.35, alpha=0.82, zorder=2)
    axis.plot(x, negative, color=RED, lw=1.35, alpha=0.82, zorder=2)
    axis.set_title("(b) Large terms cancel", fontsize=9, pad=8)
    axis.set(xlim=(-3, 3), ylim=(-5.65, 5.65), xlabel=r"$x_1$")
    axis.set_xticks((-3, 0, 3))
    axis.set_yticks((-5, 0, 5))
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(LIGHT_GRAY)
    axis.spines[["left", "bottom"]].set_linewidth(0.65)
    axis.tick_params(axis="both", color=LIGHT_GRAY, labelcolor=BLACK, width=0.65)
    axis.text(-2.75, 3.7, "Neuron 2", color=BLACK, fontsize=7.5,
              ha="left", va="center")
    axis.text(-2.75, -3.55, "Neuron 1", color=BLACK, fontsize=7.5,
              ha="left", va="center")
    axis.text(0.52, 0.055, r"$\varepsilon=0.1$", transform=axis.transAxes,
              ha="center", va="bottom", fontsize=8, color=BLACK)
    return {"epsilon": epsilon, "output_weights": [-5.0, 5.0],
            "sum_peak": float(np.max(state)), "common_y_range": [-5.65, 5.65],
            "sum_displayed": False,
            "design": "Open axes, thin opposing curves, direct neuron labels; no sum curve",
            "shading": "None",
            "neuron_labels": {"Neuron 1": "+5*tanh(x_1+0.1)",
                              "Neuron 2": "-5*tanh(x_1-0.1)"}}


def draw_profiles(axis) -> dict:
    """Plot the exact difference quotient and its analytic limit."""
    x = np.linspace(-3, 3, 1201)
    target = 1.0 / np.cosh(x) ** 2
    discrepancies = {}
    coarse = (np.tanh(x+EPSILONS[0])-np.tanh(x-EPSILONS[0]))/(2*EPSILONS[0])
    axis.fill_between(x, coarse, target, color=GREEN, alpha=0.10,
                      linewidth=0, zorder=1)
    for epsilon, color in zip(EPSILONS, COLORS):
        u = (np.tanh(x + epsilon) - np.tanh(x - epsilon)) / (2 * epsilon)
        axis.plot(x, u, color=color, lw=2.1 if epsilon == EPSILONS[-1] else 1.35,
                  alpha=1.0 if epsilon == EPSILONS[-1] else 0.82, zorder=3)
        discrepancies[str(epsilon)] = float(np.max(np.abs(u - target)))
    axis.plot(x, target, color=BLACK, lw=1.0, ls=(0, (3, 2.5)), zorder=4)
    axis.set_title("(c) A finite limit", fontsize=9, pad=8)
    axis.set_xlabel(r"$x_1$")
    axis.set_xlim(-3, 3)
    axis.set_ylim(0, 1.12)
    axis.set_xticks((-3, 0, 3))
    axis.set_yticks((0, 0.5, 1))
    axis.set_yticklabels(("0", "0.5", "1"))
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(LIGHT_GRAY)
    axis.spines[["left", "bottom"]].set_linewidth(0.65)
    axis.tick_params(axis="both", color=LIGHT_GRAY, labelcolor=BLACK, width=0.65)
    # Direct labels keep the shrinking profile gap visible without a legend box.
    for epsilon, color, anchor_x, label_y in (
        (EPSILONS[2], GREEN, -0.30, 1.04),
        (EPSILONS[1], RED, -0.20, 0.86),
        (EPSILONS[0], BLUE, -0.80, 0.68),
    ):
        anchor_y = (np.tanh(anchor_x + epsilon)
                    - np.tanh(anchor_x - epsilon)) / (2 * epsilon)
        axis.annotate(rf"$\varepsilon={epsilon}$", xy=(anchor_x, anchor_y),
                      xytext=(-2.75, label_y), color=BLACK, fontsize=7.5,
                      ha="left", va="center",
                      arrowprops={"arrowstyle": "-", "color": color,
                                  "lw": 0.65, "alpha": 0.7,
                                  "shrinkA": 3, "shrinkB": 2}, zorder=5)
    axis.annotate(r"$u^\ast$", xy=(0.20, 1.0 / np.cosh(0.20) ** 2),
                  xytext=(1.50, 1.04), color=BLACK, fontsize=9,
                  ha="left", va="center",
                  arrowprops={"arrowstyle": "-", "color": BLACK,
                              "lw": 0.65, "shrinkA": 3, "shrinkB": 2}, zorder=5)
    return discrepancies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_DIR.parent / "outputs" / "illustrations" / "tanh_singular_limit",
                        help="Output stem for PDF, SVG, PNG, and metadata JSON")
    args = parser.parse_args()

    # The panel band keeps its physical size; the figure ends below the axis
    # labels, with no caption strip beneath the panels.
    height = 2.45
    figure = plt.figure(figsize=(FULL_WIDTH, height))
    panel_width = 0.278
    panels = [figure.add_axes((x, 0.1388, panel_width, 0.6661))
              for x in (0.025, 0.3675, 0.710)]
    left, middle, right = panels
    draw_collision(left)
    contributions = draw_contributions(middle)
    discrepancies = draw_profiles(right)

    # Maintain the requested physical width despite the shared tight exporter.
    horizontal_pad = plt.rcParams["savefig.pad_inches"] / FULL_WIDTH
    vertical_pad = plt.rcParams["savefig.pad_inches"] / height
    figure.text(horizontal_pad, 1 - vertical_pad, " ", fontsize=1, va="top")
    figure.text(1 - horizontal_pad, vertical_pad, " ", fontsize=1,
                ha="right", va="bottom")
    figure.canvas.draw()
    panel_geometry = []
    for label, axis in zip(("a", "b", "c"), panels):
        bounds = axis.get_position().bounds
        physical_width, physical_height = bounds[2]*FULL_WIDTH, bounds[3]*height
        title_bounds = axis.title.get_window_extent()
        panel_geometry.append({"panel": label, "bounds_figure_fraction": list(bounds),
                               "width_inches": physical_width, "height_inches": physical_height,
                               "width_to_height": physical_width/physical_height,
                               "title_center_y_inches": float((title_bounds.y0+title_bounds.y1)/(2*figure.dpi))})
    assert np.ptp([p["width_inches"] for p in panel_geometry]) < 1e-12
    assert np.ptp([p["height_inches"] for p in panel_geometry]) < 1e-12
    assert np.ptp([p["title_center_y_inches"] for p in panel_geometry]) < 1e-12
    circle = left.patches[0].get_window_extent()
    assert abs(circle.width-circle.height) < 1e-10
    outputs = save_figure(figure, args.output)
    metadata = {
        "figure_kind": "analytic illustration, not training data",
        "family": "(tanh(x_1+epsilon)-tanh(x_1-epsilon))/(2*epsilon)",
        "target": "sech(x_1)^2",
        "epsilons": EPSILONS,
        "x_1_range": [-3.0, 3.0],
        "profile_samples": 1201,
        "coefficient_norm": "1/(sqrt(2)*epsilon)",
        "max_sampled_state_discrepancy": discrepancies,
        "cancellation_panel": contributions,
        "profile_shading": f"Deterministic gap between epsilon={EPSILONS[0]} profile and target; not uncertainty",
        "profile_panel_design": "Open light axes, thin finite profiles, emphasized green near-limit curve, direct labels and a fine dashed target",
        "figure_size_inches": list(figure.get_size_inches()),
        "panel_geometry": panel_geometry,
        "collision_panel_design": "Equal physical panel dimensions; isotropic neuron circles with larger icons and vertically spread labels and bias axis.",
        "neuron_circle_diameter_inches": [float(circle.width/figure.dpi), float(circle.height/figure.dpi)],
        "style_source": "illustrations/plot_style.py",
        "outputs": outputs,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(figure)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
