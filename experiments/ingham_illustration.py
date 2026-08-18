"""Reproduce the Ingham lower-bound illustration used in the manuscript.

For equally spaced frequencies lambda_n=n*delta, |n|<=10, and a_n=1,
the squared-exponential integral is evaluated by its exact finite sum.  The
displayed lower bound is the explicit one-dimensional Ingham constant used in
the accompanying argument.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
FIGURE_OUTPUT = WORKSPACE / "figures" / "descriptive"
STEM = "ingham_panel_T_2_5_10_an1_styled"
N = 10
WINDOWS = (2.0, 5.0, 10.0)


def exact_integral(delta: np.ndarray, window: float) -> np.ndarray:
    """Return integral_{-T}^T |sum_{n=-N}^N exp(i*n*delta*t)|^2 dt."""

    number_of_terms = 2 * N + 1
    result = np.full_like(delta, 2.0 * window * number_of_terms, dtype=float)
    for gap in range(1, number_of_terms):
        result += (
            4.0
            * (number_of_terms - gap)
            * np.sin(gap * delta * window)
            / (gap * delta)
        )
    return result


def lower_bound(delta: np.ndarray, window: float) -> np.ndarray:
    """Explicit Ingham bound, shown only in its admissible regime T>pi/delta."""

    number_of_terms = 2 * N + 1
    values = (
        4.0
        * window
        / np.pi
        * (1.0 - np.pi**2 / (window**2 * delta**2))
        * number_of_terms
    )
    return np.where(delta > np.pi / window, values, np.nan)


def make_figure(delta: np.ndarray, output_directory: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(4.685, 2.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.24, top=0.70, wspace=0.14)
    for axis, window in zip(axes, WINDOWS):
        integral = exact_integral(delta, window)
        bound = lower_bound(delta, window)
        markevery = max(1, delta.size // 14)
        axis.plot(
            delta,
            integral,
            color="#0000FF",
            lw=1.55,
            ls="-",
            marker="o",
            ms=2.5,
            markevery=markevery,
            label="Integral",
        )
        axis.plot(
            delta,
            bound,
            color="#D62728",
            lw=1.55,
            ls="--",
            marker="s",
            ms=2.4,
            markevery=markevery,
            label="Lower bound",
        )
        axis.set_xlim(0.0, 15.0)
        axis.set_ylim(-35.0, 900.0)
        axis.set_xticks((0.0, 5.0, 10.0, 15.0))
        axis.set_yticks((0.0, 300.0, 600.0, 900.0))
        axis.set_xlabel(r"$\delta$")
        axis.set_title(rf"$T={window:g}$")
        axis.grid(False)
        axis.tick_params(direction="out", length=3.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        handlelength=2.8,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_directory / f"{STEM}.pdf")
    fig.savefig(output_directory / f"{STEM}.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1500)
    parser.add_argument(
        "--sync-figures",
        "--sync-paper",
        dest="sync_figures",
        action="store_true",
        help="copy the rendered PDF and PNG to figures/descriptive",
    )
    args = parser.parse_args()
    if args.points < 50:
        raise ValueError("--points must be at least 50")

    delta = np.linspace(0.15, 15.0, args.points)
    make_figure(delta, HERE)
    payload = {
        "frequencies": f"lambda_n=n*delta, |n|<={N}",
        "coefficients": "a_n=1",
        "windows": list(WINDOWS),
        "delta_min": float(delta[0]),
        "delta_max": float(delta[-1]),
        "delta_points": int(delta.size),
        "exact_integral_formula": (
            "2*T*M + 4*sum_{k=1}^{M-1}(M-k)*sin(k*delta*T)/(k*delta), M=2*N+1"
        ),
        "lower_bound_formula": (
            "(4*T/pi)*(1-pi^2/(T^2*delta^2))*sum_n|a_n|^2, valid for T>pi/delta"
        ),
    }
    (HERE / f"{STEM}.json").write_text(json.dumps(payload, indent=2) + "\n")

    if args.sync_figures:
        FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
        for suffix in ("pdf", "png"):
            source = HERE / f"{STEM}.{suffix}"
            shutil.copy2(source, FIGURE_OUTPUT / source.name)

    print(f"Wrote {HERE / f'{STEM}.pdf'}")
    print(f"Wrote {HERE / f'{STEM}.png'}")
    print(f"Wrote {HERE / f'{STEM}.json'}")


if __name__ == "__main__":
    main()
