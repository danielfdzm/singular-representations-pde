#!/usr/bin/env python3
"""Spatial fields for the Poisson-Neumann study."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import math
import numpy as np
from records import load_results

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
from plot_style import apply_style, save_figure, FULL_WIDTH  # noqa: E402
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402

# The top row uses the same sequential ramp and limits as the companion
# fourth-order field figure, so the two are read the same way.
TOP_CMAP = "plasma"
BOTTOM_CMAP = "Blues"

# Cyan reads against every part of the plasma ramp, unlike the warm hues in it.
WEIGHT_CIRCLE = "#00E5FF"
PAIR_MARKER_OFFSET_PT = 9.0
# Stars are coloured by the highest kernel derivative order held at that centre.
# A pair has multiplicity two, so only orders 0 and 1 occur; order 2 is defined
# so a higher-multiplicity block would be drawn correctly.
ATOM_STYLE = {0: {"color": "#FFFFFF", "edge": "#101010", "label": r"$K$"},
              1: {"color": "#E8000B", "edge": "#FFFFFF", "label": r"$K,K'$ block"},
              2: {"color": "#111111", "edge": "#FFFFFF", "label": r"$K,K',K''$ block"}}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as RE  # noqa: E402


def exact_values(points):
    return RE.manufactured_jet_solution(points[:, 0], points[:, 1])


def row_colorbar(fig, row, mappable, label, pad=.014, width=.017, **kwargs):
    """Colour bar whose height matches the drawn row exactly, top and bottom.

    The panels use an equal aspect, so their drawn boxes are smaller than the
    slots allocated by subplots_adjust; positions are therefore read after a
    draw rather than computed from the grid.
    """
    fig.canvas.draw()
    first, last = row[0].get_position(), row[-1].get_position()
    rect = [last.x1+pad, first.y0, width, first.y1-first.y0]
    cax = fig.add_axes(rect)
    bar = fig.colorbar(mappable, cax=cax, **kwargs)
    bar.set_label(label, rotation=90, labelpad=6)
    bar.outline.set_linewidth(.5)
    # No "extend" arrow: it makes matplotlib shrink the axes through a locator,
    # and the bar would then stop short of the row. The clamped floor is stated
    # in the caption instead.
    cax.set_aspect("auto")
    cax.set_position(rect)
    return bar


def kernel_atoms(state):
    """Underlying kernel centres and the weight each one carries.

    A finite pair at half-gap h is two Gaussians at c-hd and c+hd with weights
    a/2 +- b/(2h); a completed pair is a single centre carrying the Gaussian and
    its directional derivative, whose weight is the norm of that coefficient
    pair. The singletons carry their own weights.
    """
    out, coefficients = [], state["coefficients"]
    for index, pair in enumerate(state["pairs"]):
        a, b = coefficients[2*index], coefficients[2*index+1]
        cx, cy = pair["center"]
        dx, dy = math.cos(pair["angle"]), math.sin(pair["angle"])
        if pair["completed"]:
            out.append(((cx, cy), math.hypot(a, b), 1))
        else:
            h = pair["half_gap"]
            out.append(((cx-h*dx, cy-h*dy), abs(a/2 + b/(2*h)), 0))
            out.append(((cx+h*dx, cy+h*dy), abs(a/2 - b/(2*h)), 0))
    for index, centre in enumerate(state["singletons"]):
        out.append((tuple(centre), abs(coefficients[2*RE.NPAIRS+index]), 0))
    return out


def overlay_atoms(ax, atoms, low, high, r_min=0.15, r_max=0.55, pairs=()):
    """Weight circles at true centres; enlarge finite-pair star separations.

    Each finite-pair star moves outward along its pair direction by a fixed
    display distance. A connector returns to a dot at its true position.
    Circles, fields and numerical states keep their original coordinates.
    """
    offsets = []
    for pair in pairs:
        if pair["completed"]:
            offsets.append((0.0, 0.0))
        else:
            dx, dy = math.cos(pair["angle"]), math.sin(pair["angle"])
            offsets.extend((sign*PAIR_MARKER_OFFSET_PT*dx,
                            sign*PAIR_MARKER_OFFSET_PT*dy) for sign in (-1, 1))
    offsets.extend([(0.0, 0.0)] * (len(atoms)-len(offsets)))
    drawn = []
    for ((cx, cy), weight, order), offset in zip(atoms, offsets):
        span = math.log10(high)-math.log10(low)
        frac = 0.0 if span <= 0 else (math.log10(max(weight, low))-math.log10(low))/span
        radius = r_min+(r_max-r_min)*frac
        style = ATOM_STYLE[order]
        ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor=WEIGHT_CIRCLE,
                            linestyle=(0, (2.4, 1.7)), linewidth=1.15, alpha=.95, zorder=4))
        transform = ax.transData
        if offset != (0.0, 0.0):
            connector = ax.annotate("", xy=(cx, cy), xytext=offset,
                                    textcoords="offset points",
                                    arrowprops={"arrowstyle": "-", "color": "#101010",
                                                "linewidth": .9, "shrinkA": 0, "shrinkB": 0},
                                    zorder=5)
            connector.arrow_patch.set_path_effects([
                path_effects.withStroke(linewidth=2.2, foreground="white")])
            ax.plot([cx], [cy], marker="o", markersize=2.8,
                    color="#101010", markeredgecolor="white", markeredgewidth=.5,
                    linestyle="none", zorder=7)
            transform += ScaledTranslation(offset[0]/72, offset[1]/72,
                                           ax.figure.dpi_scale_trans)
        ax.plot([cx], [cy], marker="*", markersize=9.0, color=style["color"],
                markeredgecolor=style["edge"], markeredgewidth=.7,
                linestyle="none", transform=transform, zorder=6)
        drawn.append({"centre": [cx, cy], "weight": weight,
                      "radius": radius, "derivative_order": order,
                      "marker_offset_points": list(offset)})
    return drawn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"data/poisson")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/poisson")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--grid", type=int, default=241)
    args = parser.parse_args()
    data = load_results(args.input)
    runs = {r["method"]: r for r in data["runs"] if r["seed"] == args.seed}

    axis = np.linspace(-1.0, 1.0, args.grid)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((X.ravel(), Y.ravel()))
    shape = X.shape
    exact = exact_values(points).reshape(shape)

    def field(state):
        return RE.evaluate_state(state, points)[0].reshape(shape)

    without = field(runs["ordinary"]["final_state"])
    withc = field(runs["completion"]["final_state"])
    start = field(runs["completion"]["initial_state"])

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 5.35))
    fig.subplots_adjust(left=.065, right=.90, bottom=.075, top=.945, wspace=.16, hspace=.50)

    norm = Normalize(vmin=float(exact.min()), vmax=float(exact.max()))
    tops = ((exact, "(a) Exact solution $u^*$"),
            (without, "(b) Without completion"),
            (withc, "(c) Basic completion"))
    atoms = {1: kernel_atoms(runs["ordinary"]["final_state"]),
             2: kernel_atoms(runs["completion"]["final_state"])}
    weights = [w for group in atoms.values() for _, w, _ in group]
    weight_low, weight_high = min(weights), max(weights)   # one scale for both panels
    overlays = {}
    for column, (ax, (value, title)) in enumerate(zip(axes[0], tops)):
        image = ax.pcolormesh(X, Y, value, cmap=TOP_CMAP, norm=norm, shading="auto", rasterized=True)
        if column in atoms:
            pairs = runs["ordinary"]["final_state"]["pairs"] if column == 1 else ()
            overlays[column] = overlay_atoms(ax, atoms[column], weight_low, weight_high,
                                             pairs=pairs)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal"); ax.set_xlabel("$x_1$")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        if column == 0:
            ax.set_ylabel("$x_2$")
        else:
            ax.set_yticklabels([])
    top_image = image

    errors = (np.abs(start-exact), np.abs(without-exact), np.abs(withc-exact))
    floor = 1e-16
    high = float(max(e.max() for e in errors))
    lognorm = LogNorm(vmin=floor, vmax=high)
    titles = ("(d) Initial error", "(e) Error without completion", "(f) Completion error")
    for column, (ax, value, title) in enumerate(zip(axes[1], errors, titles)):
        image = ax.pcolormesh(X, Y, np.maximum(value, floor), cmap=BOTTOM_CMAP,
                              norm=lognorm, shading="auto", rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal"); ax.set_xlabel("$x_1$")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        if column == 0:
            ax.set_ylabel("$x_2$")
        else:
            ax.set_yticklabels([])
    orders = sorted({o for group in atoms.values() for _, _, o in group})
    handles = [Line2D([], [], marker="*", linestyle="none", markersize=11,
                      color=ATOM_STYLE[o]["color"], markeredgecolor="#101010",
                      markeredgewidth=.7, label=ATOM_STYLE[o]["label"]) for o in orders]
    handles.append(Line2D([], [], color=WEIGHT_CIRCLE, linestyle=(0, (2.4, 1.7)),
                          linewidth=1.4, label="radius grows with the weight"))
    row_colorbar(fig, axes[0], top_image, "$u$")
    row_colorbar(fig, axes[1], image, "$|u-u^*|$")

    fig.canvas.draw()
    upper, lower = axes[0][0].get_position(), axes[1][0].get_position()
    fig.legend(handles=handles, loc="center",
               bbox_to_anchor=(0.5*(axes[0][0].get_position().x0
                                    + axes[0][2].get_position().x1),
                               0.5*(lower.y1+upper.y0)),
               ncol=len(handles), fontsize=8.4, frameon=False,
               handlelength=1.9, columnspacing=1.9, handletextpad=.55)

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, args.output/"poisson_neumann_2d_fields")
    summary = {"seed": args.seed, "grid": args.grid,
               "max_abs_error": {"initial": float(errors[0].max()),
                                 "without_completion": float(errors[1].max()),
                                 "completion": float(errors[2].max())},
               "atom_overlay": {"weight_range": [weight_low, weight_high],
                                "pair_marker_offset_points": PAIR_MARKER_OFFSET_PT,
                                "pair_display": "Finite-pair stars move outward along their true joining direction; connectors end at true-centre dots. Separations are exaggerated; fields and weight circles are unchanged.",
                                "radius_range": [0.15, 0.55],
                                "radius_rule": "r_min+(r_max-r_min)*log10(w/w_min)/log10(w_max/w_min), shared by both panels",
                                "without_completion": overlays.get(1),
                                "completion": overlays.get(2)},
               "outputs": outputs}
    (args.output/"poisson_neumann_2d_fields.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary["max_abs_error"]))


if __name__ == "__main__":
    main()
