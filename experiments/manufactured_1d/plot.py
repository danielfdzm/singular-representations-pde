"""Plot the whole-line manufactured example from its recorded experiment data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style, save_figure, FULL_WIDTH, BLUE, RED, GREEN, BLACK, GRAY

apply_style(8.5)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def plot(input_directory, output):
    with (input_directory / "sweep.csv").open() as stream:
        data = list(csv.DictReader(stream))
    with (input_directory / "states.csv").open() as stream:
        states = list(csv.DictReader(stream))
    summary = json.loads((input_directory / "summary.json").read_text())
    cases = {case["epsilon"]: case for case in summary["cases"]}
    # Global search over all two-atom networks under the cap (global_search.py).
    with (input_directory / "global_search.csv").open() as stream:
        global_rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.8))
    fig.subplots_adjust(left=.075, right=.985, bottom=.29, top=.87, wspace=.40)
    x = np.array([float(row["x"]) for row in states])
    exact = np.array([float(row["g0"]) for row in states])
    finite = np.array([float(row["finite_R2"]) for row in states])
    axes[0].plot(x, exact, color=BLACK, lw=1.9, label="Exact")
    axes[0].plot(x, exact, color=GREEN, ls="--", lw=1.4, label="Completed")
    axes[0].plot(x, finite, color=BLUE, lw=1.1, label=r"Finite, $R=2$")
    axes[0].set(xlabel=r"$x$", ylabel="Solution", xlim=(-4.5, 4.5), ylim=(-.68, .68))
    axes[0].set_xticks([-4, -2, 0, 2, 4])
    axes[0].set_title("(a) Manufactured solution")
    axes[0].legend(loc="upper right", fontsize=6.3, handlelength=1.3)

    for epsilon, color in (("0", BLUE), ("0.01", RED)):
        for method, linestyle in (("prescribed", "-"), ("symmetric_optimized", "--")):
            rows = sorted((row for row in data if row["epsilon"] == epsilon and row["method"] == method),
                          key=lambda row: float(row["R"]))
            radius = np.array([float(row["R"]) for row in rows])
            errors = np.array([float(row["relative_h1_error"]) for row in rows])
            axes[1].loglog(radius, errors, color=color, ls=linestyle, lw=1.25)
            if method == "prescribed":
                case = cases[epsilon]
                if epsilon == "0":
                    ratio = (np.array([float(row["R4_excess"]) for row in rows])
                             / float(case["boundary_prescribed_path_limit_R4"]))
                else:
                    ratio = np.array([float(row["R2_excess"]) for row in rows]) / float(case["positive_residual_path_limit_R2"])
                axes[2].semilogx(radius, ratio, color=color, ls=linestyle, lw=1.25)
    # The best pair found coincides with the symmetric branch; it is drawn
    # from the independent numerical search, including the smaller perturbation.
    for epsilon, color in (("0", BLUE), ("0.001", GREEN), ("0.01", RED)):
        rows = sorted((row for row in global_rows if row["epsilon"] == epsilon), key=lambda row: float(row["R"]))
        axes[2].semilogx([float(row["R"]) for row in rows], [float(row["scaled_over_limit"]) for row in rows],
                         color=color, ls="--", lw=1.25, marker="o", ms=2.1, mew=0, markevery=4)

    completed_error = float(cases["0.01"]["completed_relative_h1_error"])
    axes[1].axhline(completed_error, color=GREEN, ls=":", lw=1.35)
    axes[1].set(xlabel=r"Ordinary coefficient cap $R$", ylabel=r"Relative $H^1(\mathbb{R})$ error",
                xlim=(2, 1e5), ylim=(5e-12, .15))
    axes[1].set_title("(b) Error against exact state")
    axes[1].text(.72, .88, r"$\varepsilon=0.01$", transform=axes[1].transAxes,
                 fontsize=7, color=RED, ha="center")
    axes[1].text(.67, .49, r"$\varepsilon=0$", transform=axes[1].transAxes,
                 fontsize=7, color=BLUE, ha="center")
    axes[2].axhline(1, color=GRAY, lw=.8, ls=":", zorder=0)
    axes[2].set(xlabel=r"Ordinary coefficient cap $R$", ylabel="Scaled excess / limiting value",
                xlim=(2, 1e5), yscale="log", ylim=(.7, 110))
    axes[2].set_title("(c) Two different energy laws")
    axes[2].text(.60, .80, r"$\varepsilon=0.001$: $R^2 E$", transform=axes[2].transAxes,
                 color=GREEN, fontsize=7, ha="center")
    axes[2].text(.60, .66, r"$\varepsilon=0.01$: $R^2 E$", transform=axes[2].transAxes,
                 color=RED, fontsize=7, ha="center")
    axes[2].text(.53, .30, r"$\varepsilon=0$: $R^4 E$", transform=axes[2].transAxes,
                 color=BLUE, fontsize=7, ha="center")
    for axis in axes[1:]:
        axis.set_xticks([10, 1000, 100000])
        axis.minorticks_off()
    axes[2].set_yticks([1, 10, 100], ["1", "10", "100"])
    paths = save_figure(fig, output)
    plt.close(fig)
    metadata = {
        "input": str(input_directory), "outputs": paths,
        "energy_scope": "Panel (c): prescribed pairs (solid) and the best capped two-atom pair found by the numerical search (dashed), which is symmetric and cap-active; differences from J(u_*), u_* = K'. The search does not establish global optimality.",
        "normalizations": {epsilon: {key: value for key, value in case.items() if "limit_R" in key}
                           for epsilon, case in cases.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/manufactured_1d")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures/figure_06")
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
