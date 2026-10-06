#!/usr/bin/env python3
"""Draw main Figures 7--9 from genuine ordinary-weight comparisons only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_style import apply_style, save_figure, FULL_WIDTH, BLUE, RED, LIGHT_GRAY
from analyse_main import CODE, METHODS, analyse, evaluate_state, fingerprint, read_results, target_state
apply_style(font_size=8.5)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.transforms import ScaledTranslation

COLORS = {"ordinary": BLUE, "completion": RED}
LABELS = {"ordinary": "Standard optimization", "completion": "Completion"}
STYLES = {"ordinary": "-", "completion": "--"}


def load(study, directory, cache=None):
    payload = read_results(directory)
    result = json.loads(cache.read_text()) if cache else analyse(study, directory)
    if result["run_sha256"] != {f"{r['method']}:{r['seed']}": fingerprint(r) for r in payload["runs"]}:
        raise ValueError("Analysis cache does not match these runs")
    if result["quadrature_order"] != 96 or not result["histories"]:
        raise ValueError("Main trajectories require full order-96 analysis")
    return payload, result


def trajectories(study, payload, analysis, output, number):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))
    fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.80, wspace=.32)
    for ax, metric, title, ylabel in zip(axes,
            ("running_min_h1_error", "coefficient_norm"),
            ("(a) Best recorded error", "(b) Coefficients used by the solver"),
            ("Relative $H^1$ error", "Coefficient norm")):
        for method in METHODS:
            for run in (r for r in payload["runs"] if r["method"] == method):
                trace = analysis["history_metrics"][f"{method}:{run['seed']}"]
                ax.plot(trace["work"], trace[metric], color=COLORS[method], ls=STYLES[method], lw=.85, alpha=.65)
                if run["status"] != "optimizer_converged":
                    ax.plot(trace["work"][-1], trace[metric][-1], color=COLORS[method], marker="x", ms=4,
                            mew=.85, linestyle="none")
        ax.set(xscale="log", yscale="log", xlabel="Coefficient fits" if study == "matern" else "Coefficient solves", ylabel=ylabel)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=LIGHT_GRAY, lw=.45, alpha=.6)
    axes[1].axhline(analysis["target_coefficient_norm"], color="black", ls=":", lw=.85)
    handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], label=LABELS[m]) for m in METHODS]
    handles.extend((Line2D([], [], color="black", marker="x", ls="none", label="Abnormal termination"),
                    Line2D([], [], color="black", ls=":", label="Completed target norm")))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1.02), ncol=2, columnspacing=2.6)
    save_figure(fig, output / f"figure_{number:02d}")
    plt.close(fig)


def atoms(state):
    result = []
    c = state["coefficients"]
    for i, pair in enumerate(state["pairs"]):
        center = np.asarray(pair["center"])
        direction = np.array((math.cos(pair["angle"]), math.sin(pair["angle"])))
        first, second = c[2*i:2*i+2]
        if pair["completed"]:
            result.append((center, math.hypot(first, second), True, np.zeros(2)))
        else:
            for sign, weight in ((1, first), (-1, second)):
                result.append((center+sign*pair["half_gap"]*direction, abs(weight), False, sign*9*direction))
    for i, center in enumerate(state["singletons"]):
        result.append((np.asarray(center), abs(c[2*len(state["pairs"])+i]), False, np.zeros(2)))
    return result


def fields(payload, output, seed=16, grid=241):
    runs = {r["method"]: r for r in payload["runs"] if r["seed"] == seed}
    coordinate = np.linspace(-1, 1, grid)
    x, y = np.meshgrid(coordinate, coordinate)
    points = np.column_stack((x.ravel(), y.ravel()))
    exact = evaluate_state("poisson", target_state("poisson"), points)[0].reshape(x.shape)
    values = {m: evaluate_state("poisson", r["final_state"], points)[0].reshape(x.shape) for m, r in runs.items()}
    initial = evaluate_state("poisson", runs["ordinary"]["initial_state"], points)[0].reshape(x.shape)
    errors = [np.abs(initial-exact), np.abs(values["ordinary"]-exact), np.abs(values["completion"]-exact)]
    top_norm = Normalize(float(exact.min()), float(exact.max()))
    floor, ceiling = 1e-12, float(max(e.max() for e in errors))
    bottom_norm = LogNorm(floor, ceiling)
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH, 5.35))
    fig.subplots_adjust(left=.065, right=.90, bottom=.075, top=.945, wspace=.16, hspace=.50)
    groups = {1: atoms(runs["ordinary"]["final_state"]), 2: atoms(runs["completion"]["final_state"])}
    weights = [w for group in groups.values() for _, w, _, _ in group]
    low, high = max(min(weights), 1e-300), max(weights)
    for row in range(2):
        images = (exact, values["ordinary"], values["completion"]) if row == 0 else errors
        titles = ("(a) Exact target", "(b) Standard optimization", "(c) Completion") if row == 0 else (
                  "(d) Initial error", "(e) Standard error", "(f) Completion error")
        for col, (ax, value, title) in enumerate(zip(axes[row], images, titles)):
            image = ax.pcolormesh(x, y, value if row == 0 else np.maximum(value, floor),
                cmap="plasma" if row == 0 else "Blues", norm=top_norm if row == 0 else bottom_norm,
                shading="auto", rasterized=True)
            ax.set_title(title)
            ax.set(aspect="equal", xlim=(-1, 1), ylim=(-1, 1), xlabel="$x_1$", xticks=(-1, 0, 1), yticks=(-1, 0, 1))
            if col == 0:
                ax.set_ylabel("$x_2$")
            else:
                ax.set_yticklabels([])
            if row == 0 and col in groups:
                for center, weight, completed, offset in groups[col]:
                    fraction = 0 if high <= low else math.log10(max(weight, low)/low)/math.log10(high/low)
                    ax.add_patch(Circle(center, .12+.28*fraction, fill=False, edgecolor="#00E5FF",
                                        linestyle=(0, (2.4, 1.7)), linewidth=1.05, zorder=4))
                    transform = ax.transData
                    if np.any(offset):
                        ax.annotate("", xy=center, xytext=offset, textcoords="offset points",
                            arrowprops={"arrowstyle": "-", "color": "black", "linewidth": .7}, zorder=5)
                        ax.plot(*center, marker="o", ms=2.5, color="black", mec="white", mew=.5, zorder=6)
                        transform += ScaledTranslation(offset[0]/72, offset[1]/72, fig.dpi_scale_trans)
                    ax.plot(*center, marker="*", ms=8.5, color="#E8000B" if completed else "white",
                        mec="white" if completed else "black", mew=.6, ls="none", transform=transform, zorder=7)
        fig.canvas.draw()
        first, last = axes[row, 0].get_position(), axes[row, -1].get_position()
        bar_axis = fig.add_axes((last.x1+.014, first.y0, .017, first.height))
        fig.colorbar(image, cax=bar_axis).set_label("$u$" if row == 0 else "$|u-u^*|$")
    handles = [Line2D([], [], marker="*", ls="none", color="white", mec="black", ms=10, label="Kernel centres"),
        Line2D([], [], marker="*", ls="none", color="#E8000B", ms=10, label="Derivative blocks"),
        Line2D([], [], color="#00E5FF", ls="--", label="Radius grows with coefficient size")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.48, .51), ncol=3, fontsize=7.5)
    save_figure(fig, output / "figure_07")
    plt.close(fig)
    return {"seed": seed, "grid": grid, "error_display_floor": floor,
            "max_absolute_error": dict(zip(("initial", "ordinary", "completion"), (float(e.max()) for e in errors)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matern", type=Path, default=CODE / "data/raw_main/matern")
    parser.add_argument("--poisson", type=Path, default=CODE / "data/raw_main/poisson")
    parser.add_argument("--analysis-directory", type=Path)
    parser.add_argument("--output", type=Path, default=CODE / "outputs/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for study, directory, number in (("poisson", args.poisson, 8), ("matern", args.matern, 9)):
        cache = args.analysis_directory / f"raw_{study}.json" if args.analysis_directory else None
        payload, analysis = load(study, directory, cache)
        trajectories(study, payload, analysis, args.output, number)
        if study == "poisson":
            info = fields(payload, args.output)
            (args.output / "figure_07.json").write_text(json.dumps(info, indent=2)+"\n")
        (args.output / f"figure_{number:02d}.json").write_text(json.dumps(analysis, indent=2)+"\n")
    print("Main Figures 7--9 use ordinary weights until completion; two methods only.")


if __name__ == "__main__":
    main()
