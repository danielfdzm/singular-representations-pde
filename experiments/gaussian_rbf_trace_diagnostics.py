"""Plot Gaussian collision and direct H1 Gramian diagnostics from cached traces.

This postprocessing script does not retrain the model. It reads the JSON trace
produced by gaussian_collision_mixed_precision.py and evaluates
the exact two-center H1 Gramian formulas along the recorded separations.
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

from artifact_paths import (
    GALLERY_DIRECTORY,
    GAUSSIAN_DATA_DIRECTORY,
    experiment_output_directory,
)


STEM = "gaussian_rbf_instability_unpenalized_mixed_precision_diagnostics"
DEFAULT_INPUT = (
    GAUSSIAN_DATA_DIRECTORY
    / "gaussian_rbf_instability_unpenalized_mixed_precision.json"
)
DEFAULT_OUTPUT_DIRECTORY = experiment_output_directory("gaussian")

COLORS = {"float32": "#FF0000", "float64": "#0000FF"}
PRECISION_LABELS = {"float32": "single precision", "float64": "double precision"}
PRECISION_STYLES = {
    "float32": {"linestyle": "--", "marker": "s"},
    "float64": {"linestyle": "-", "marker": "o"},
}


def normalized_gramian_diagnostics(separation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lambda_min/g(0) and the condition number for the H1 Gramian.

    Apart from a common positive normalization, the Gramian correlation is

        g(r) = sqrt(pi) exp(-r^2/4) (3/2 - r^2/4).

    The stable expression below avoids subtracting nearly equal exponentials.
    """

    h2 = np.square(np.asarray(separation, dtype=np.float64))
    exponential = np.exp(-h2 / 4.0)
    normalized_lambda_min = -np.expm1(-h2 / 4.0) + exponential * h2 / 6.0
    normalized_lambda_min = np.maximum(normalized_lambda_min, np.finfo(float).tiny)
    condition_number = (2.0 - normalized_lambda_min) / normalized_lambda_min
    return normalized_lambda_min, condition_number


def load_order_one(path: Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(path.read_text())
    records = [item for item in raw["orders"] if int(item["order"]) == 1]
    if len(records) != 1:
        raise ValueError(f"expected one order-one record in {path}")
    order = records[0]
    result: dict[str, dict[str, np.ndarray]] = {}
    for precision in ("float32", "float64"):
        record = order[precision]
        trace = record["trace"]
        iteration = np.asarray(trace["iteration"], dtype=np.float64)
        separation = np.asarray(trace["separation"], dtype=np.float64)
        relative_error = np.asarray(record["realized_relative_error"], dtype=np.float64)
        if not (iteration.size == separation.size == relative_error.size):
            raise ValueError(f"inconsistent {precision} trace lengths in {path}")
        normalized_lambda_min, condition_number = normalized_gramian_diagnostics(separation)
        result[precision] = {
            "iteration": iteration,
            "relative_error": relative_error,
            "separation": separation,
            "normalized_lambda_min": normalized_lambda_min,
            "condition_number": condition_number,
        }
    return result


def make_figure(
    data: dict[str, dict[str, np.ndarray]], output_directory: Path
) -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(4.685, 4.6), sharex=True)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.10, top=0.92, wspace=0.34, hspace=0.34)

    panels = (
        ("relative_error", r"relative $L^2$ error", "(a) realized state error"),
        ("separation", r"center separation $h$", "(b) collision scale"),
        (
            "normalized_lambda_min",
            r"$\lambda_{\min}(G_{H^1})/g(0)$",
            r"(c) normalized smallest eigenvalue",
        ),
        ("condition_number", r"$\kappa(G_{H^1})$", "(d) condition number"),
    )

    for ax, (key, ylabel, title) in zip(axes.ravel(), panels):
        for precision in ("float64", "float32"):
            values = data[precision]
            plot_iteration = np.maximum(values["iteration"], 1.0)
            ax.plot(
                plot_iteration,
                values[key],
                color=COLORS[precision],
                lw=1.55,
                ls=PRECISION_STYLES[precision]["linestyle"],
                marker=PRECISION_STYLES[precision]["marker"],
                ms=2.8,
                markevery=max(1, len(plot_iteration) // 12),
                label=PRECISION_LABELS[precision],
            )
            ax.plot(
                plot_iteration[-1],
                values[key][-1],
                marker=PRECISION_STYLES[precision]["marker"],
                ms=3.4,
                color=COLORS[precision],
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(False)
        ax.tick_params(direction="out", length=3.0)

    axes[0, 0].legend(frameon=False, loc="lower left")
    axes[1, 0].set_xlabel("Adam iteration")
    axes[1, 1].set_xlabel("Adam iteration")

    output_directory.mkdir(parents=True, exist_ok=True)
    out_pdf = output_directory / f"{STEM}.pdf"
    out_png = output_directory / f"{STEM}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    for precision in ("float32", "float64"):
        record = data[precision]
        print(
            f"{precision}: final error={record['relative_error'][-1]:.6e}, "
            f"h={record['separation'][-1]:.6e}, "
            f"lambda_min/g0={record['normalized_lambda_min'][-1]:.6e}, "
            f"condition={record['condition_number'][-1]:.6e}"
        )
    return out_pdf, out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--sync-figures",
        action="store_true",
        help="copy the rendered PNG preview to figures/gallery",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_order_one(args.input)
    _, output_png = make_figure(data, args.output_directory)
    if args.sync_figures:
        GALLERY_DIRECTORY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_png, GALLERY_DIRECTORY / output_png.name)


if __name__ == "__main__":
    main()
