"""Two-panel parameter-plane plot for the tanh offset-collision example."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
TRACE64_PATH = ROOT / "tanh_parameter_escape_loss.json"
TRACE32_PATH = ROOT / "tanh_parameter_escape_loss_float32.json"
OUT_PDF = ROOT / "wb_trajectories.pdf"
OUT_PNG = ROOT / "wb_trajectories.png"

BLUE = "#0000FF"
RED = "#FF0000"


def load_trace(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text())
    history = data["history"]
    return {
        "w1": np.asarray([row["w1"] for row in history], dtype=float),
        "b1": np.asarray([row["b1"] for row in history], dtype=float),
        "w2": np.asarray([row["w2"] for row in history], dtype=float),
        "b2": np.asarray([row["b2"] for row in history], dtype=float),
    }


def add_trajectory_arrows(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, n_arrows: int = 3) -> None:
    if len(x) < 2:
        return
    candidates = np.linspace(0, len(x) - 2, n_arrows + 2, dtype=int)[1:-1]
    for idx in candidates:
        dx = x[idx + 1] - x[idx]
        dy = y[idx + 1] - y[idx]
        if np.hypot(dx, dy) < 1e-12:
            continue
        ax.annotate(
            "",
            xy=(x[idx + 1], y[idx + 1]),
            xytext=(x[idx], y[idx]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.35, mutation_scale=11),
            zorder=4,
        )


def plot_panel(ax: plt.Axes, trace: dict[str, np.ndarray], title: str) -> None:
    ax.plot(trace["w1"], trace["b1"], color=BLUE, lw=1.7, ls="-", zorder=2)
    ax.plot(trace["w2"], trace["b2"], color=RED, lw=1.7, ls="--", zorder=2)
    add_trajectory_arrows(ax, trace["w1"], trace["b1"], BLUE)
    add_trajectory_arrows(ax, trace["w2"], trace["b2"], RED)

    for key_w, key_b, color, final_marker in [
        ("w1", "b1", BLUE, ">"),
        ("w2", "b2", RED, "<"),
    ]:
        ax.plot(
            trace[key_w][0],
            trace[key_b][0],
            "o",
            color=color,
            markersize=4.6,
            markeredgecolor="k",
            markeredgewidth=0.65,
            zorder=5,
        )
        ax.plot(
            trace[key_w][-1],
            trace[key_b][-1],
            final_marker,
            color=color,
            markersize=5.0,
            markeredgecolor="k",
            markeredgewidth=0.65,
            zorder=5,
        )

    ax.axhline(0, color="black", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.axvline(0, color="black", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel(r"$w$")
    ax.set_ylabel(r"$b$")
    ax.set_title(title)
    ax.tick_params(direction="out", length=3.2)


def main() -> None:
    trace64 = load_trace(TRACE64_PATH)
    trace32 = load_trace(TRACE32_PATH)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.0,
        }
    )

    all_w = np.concatenate([trace64["w1"], trace64["w2"], trace32["w1"], trace32["w2"]])
    all_b = np.concatenate([trace64["b1"], trace64["b2"], trace32["b1"], trace32["b2"]])
    x_pad = 0.07 * (all_w.max() - all_w.min())
    y_pad = 0.08 * (all_b.max() - all_b.min())

    fig, axes = plt.subplots(1, 2, figsize=(4.685, 2.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.88, wspace=0.18)

    plot_panel(axes[0], trace64, r"(a) double precision")
    plot_panel(axes[1], trace32, r"(b) single precision")

    for ax in axes:
        ax.set_xlim(all_w.min() - x_pad, all_w.max() + x_pad)
        ax.set_ylim(all_b.min() - y_pad, all_b.max() + y_pad)

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.7, ls="-", label=r"$(w_1,b_1)$"),
        Line2D([0], [0], color=RED, lw=1.7, ls="--", label=r"$(w_2,b_2)$"),
        Line2D([0], [0], marker="o", color="k", lw=0, markersize=4.6, label="start"),
        Line2D([0], [0], marker=">", color="k", lw=0, markersize=5.0, label="end"),
    ]
    axes[1].legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=240)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
