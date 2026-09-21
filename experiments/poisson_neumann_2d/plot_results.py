#!/usr/bin/env python3
"""Figure for the Poisson-Neumann study: accuracy and parameter size."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from records import load_results

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, save_figure, FULL_WIDTH, BLUE, RED, LIGHT_GRAY  # noqa: E402
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import experiment as SH  # noqa: E402


def kernel_weight_norm(state):
    """Norm of the weights the underlying kernel atoms must carry.

    A finite pair at half gap h is two kernels with weights a/2 +- b/(2h); a
    completed pair is one centre carrying the kernel and its directional
    derivative, with the finite coefficient pair (a, b). The singleton carries
    its own weight. This is the quantity that escapes when a pair collides and
    that completion keeps bounded.
    """
    weights, coefficients = [], state["coefficients"]
    for index, pair in enumerate(state["pairs"]):
        a, b = coefficients[2*index], coefficients[2*index+1]
        if pair["completed"]:
            weights.append(math.hypot(a, b))
        else:
            h = pair["half_gap"]
            weights.extend((abs(a/2+b/(2*h)), abs(a/2-b/(2*h))))
    weights.extend(abs(x) for x in coefficients[2*SH.NPAIRS:])
    return float(np.linalg.norm(weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"data/poisson")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/poisson")
    parser.add_argument("--quadrature-order", type=int, default=96)
    args = parser.parse_args()
    data = load_results(args.input)
    points, weights = SH.quadrature(args.quadrature_order)
    exact = SH.manufactured_jet_solution(points[:, 0], points[:, 1])
    exact_gradient = SH.manufactured_jet_gradient(points[:, 0], points[:, 1])
    norm = np.dot(weights, exact*exact)+np.dot(weights, np.sum(exact_gradient**2, axis=1))

    def error(state):
        u, g = SH.evaluate_state(state, points)
        du, dg = u-exact, g-exact_gradient
        return float(np.sqrt((np.dot(weights, du*du)
                              + np.dot(weights, np.sum(dg*dg, axis=1)))/norm))

    runs = {}
    for record in data["runs"]:
        runs.setdefault(record["method"], {})[record["seed"]] = record
    METHODS = ("ordinary", "completion")
    colors = {"ordinary": BLUE, "completion": RED}
    labels = {"ordinary": "Without completion", "completion": "Basic completion"}
    styles = {"ordinary": "-", "completion": "--"}

    # The error is a running minimum; the weight norm is the value actually
    # carried at each accepted iterate, so it is shown as it evolves.
    traces = {}
    for method in METHODS:
        for seed, record in runs[method].items():
            work, errs, sizes = [], [], []
            for entry in record["history"]:
                work.append(entry["solve_count"])
                errs.append(error(entry["state"]))
                sizes.append(kernel_weight_norm(entry["state"]))
            traces[(method, seed)] = {"work": np.asarray(work),
                                      "error": np.minimum.accumulate(np.asarray(errs)),
                                      "weight": np.asarray(sizes)}

    # Panel (a) emphasizes, per method, the start reaching the lowest error.
    # Panel (b) emphasizes one shared start -- the one whose finite branch
    # escapes furthest -- so the two curves there are a matched pair.
    best = {}
    escaping = max(sorted(runs["ordinary"]),
                   key=lambda s: traces[("ordinary", s)]["weight"].max())
    for method in METHODS:
        seeds = sorted(runs[method])
        best[("error", method)] = min(seeds, key=lambda s: traces[(method, s)]["error"][-1])
        best[("weight", method)] = escaping

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.4))
    fig.subplots_adjust(left=.075, right=.988, bottom=.165, top=.9, wspace=.265)
    panels = (("error", "Relative $H^1$ error"), ("weight", "Kernel weight norm"))
    for axis, (metric, ylabel) in zip(axes, panels):
        for method in METHODS:
            for seed in sorted(runs[method]):
                trace = traces[(method, seed)]
                bold = seed == best[(metric, method)]
                axis.plot(trace["work"], trace[metric], color=colors[method],
                          ls=styles[method], lw=1.7 if bold else .7,
                          alpha=1 if bold else .3, zorder=5 if bold else 2)
        axis.set(yscale="log", xscale="log", xlabel="Coefficient solves",
                 ylabel=ylabel)
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", color=LIGHT_GRAY, lw=.45, alpha=.55)
    handles = [Line2D([], [], color=colors[m], ls=styles[m], lw=1.7, label=labels[m])
               for m in METHODS]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8, borderaxespad=.3)
    axes[0].set_title("(a) State error", fontsize=9)
    axes[1].set_title("(b) Parameter size", fontsize=9)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, args.output/"poisson_neumann_2d")
    record = {"outputs": outputs, "quadrature_order": args.quadrature_order,
              "y_limits": {metric: [float(v) for v in ax.get_ylim()]
                           for (metric, _), ax in zip(panels, axes)},
              "emphasized_start": {f"{metric}:{method}": int(best[(metric, method)])
                                   for metric, _ in panels for method in METHODS},
              "final": {f"{metric}:{method}": {str(s): float(traces[(method, s)][metric][-1])
                                              for s in sorted(runs[method])}
                        for metric, _ in panels for method in METHODS},
              "shared_weight_start": int(escaping),
              "peak_weight_norm": {method: {str(s): float(traces[(method, s)]["weight"].max())
                                            for s in sorted(runs[method])} for method in METHODS}}
    (args.output/"poisson_neumann_2d.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"emphasized_start": record["emphasized_start"]}))


if __name__ == "__main__":
    main()
