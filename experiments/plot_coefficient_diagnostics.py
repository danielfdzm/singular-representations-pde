#!/usr/bin/env python3
"""Redraw Figures 7 and 9 with separate stable and ordinary weight norms.

Adapted from the manuscript's coefficient_diagnostics.py. Uses the original
state evaluators and source-hash-checked order-96 H1 error cache; no training.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "experiments"))
from plot_style import apply_style, save_figure, FULL_WIDTH, BLUE, RED, LIGHT_GRAY

apply_style(8)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ("ordinary", "completion")
COLORS = dict(zip(METHODS, (BLUE, RED)))
STYLES = dict(zip(METHODS, ("-", "--")))
LABELS = dict(zip(METHODS, ("Without completion", "Basic completion")))
CASES = {
    "matern": ("matern_swift_hohenberg_2d", "figure_07", "Nonlinear coefficient fits"),
    "poisson": ("poisson_neumann_2d", "figure_09", "Coefficient solves"),
}


def stable_coefficient_norm(state):
    """Euclidean norm of the same coordinates used by every solver branch."""
    return float(np.linalg.norm(state["coefficients"]))


def ordinary_weight_norm(state):
    """Raw translate weights, defined only while the whole state is finite.

    A completed atom has no ordinary translate coefficients. Return None once
    any pair has completed; do not splice derivative coefficients into this norm.
    """
    if any(pair["completed"] for pair in state["pairs"]):
        return None
    coefficients = state["coefficients"]
    weights = []
    for index, pair in enumerate(state["pairs"]):
        a, b = coefficients[2 * index:2 * index + 2]
        h = pair["half_gap"]
        assert h > 0
        weights.extend((a / 2 + b / (2 * h), a / 2 - b / (2 * h)))
    weights.extend(coefficients[2 * len(state["pairs"]):])
    return float(np.linalg.norm(weights))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, default=REPOSITORY / "outputs/figures")
    args = parser.parse_args()
    directory, figure_name, work_label = CASES[args.case]
    experiment_path = REPOSITORY / "experiments" / directory / "experiment.py"
    spec = importlib.util.spec_from_file_location("experiment_reused", experiment_path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)

    record_path = REPOSITORY / "data" / args.case / "results.json.gz"
    with gzip.open(record_path, "rt") as stream:
        data = json.load(stream)
    original_table = json.loads((record_path.parent / "paper_table.json").read_text())
    published_errors = json.loads((record_path.parent / "h1_errors.json").read_text())
    paper_seeds = original_table["paper_seeds"]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    points, weights = experiment.quadrature(96)
    exact = experiment.manufactured_jet_solution(points[:, 0], points[:, 1])
    gradient = experiment.manufactured_jet_gradient(points[:, 0], points[:, 1])
    norm = np.dot(weights, exact * exact) + np.dot(weights, np.sum(gradient * gradient, axis=1))
    error_cache_path = output / f"{args.case}_h1_error_cache.json.gz"
    if not error_cache_path.exists():
        shutil.copy2(record_path.parent / "history_h1_error_cache.json.gz", error_cache_path)
    error_cache = {}
    if error_cache_path.exists():
        with gzip.open(error_cache_path, "rt") as stream:
            cached = json.load(stream)
        if cached["source_sha256"] == sha256(record_path):
            error_cache = cached["errors"]

    def error(state):
        key = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        if key not in error_cache:
            u, g = experiment.evaluate_state(state, points)
            du, dg = u - exact, g - gradient
            error_cache[key] = float(np.sqrt((np.dot(weights, du * du)
                                    + np.dot(weights, np.sum(dg * dg, axis=1))) / norm))
        return error_cache[key]

    traces, final = {}, {}
    started = time.monotonic()
    for record in data["runs"]:
        method, seed = record["method"], record["seed"]
        if method not in METHODS:
            continue
        work, errors, stable, raw = [], [], [], []
        for entry in record["history"]:
            state = entry["state"]
            stable_norm = stable_coefficient_norm(state)
            assert math.isclose(stable_norm, entry["stable_coefficient_norm"], rel_tol=1e-12)
            ordinary_norm = ordinary_weight_norm(state)
            work.append(entry["solve_count"])
            errors.append(error(state))
            stable.append(stable_norm)
            raw.append(np.nan if ordinary_norm is None else ordinary_norm)
        traces[(method, seed)] = {"work": np.asarray(work),
                                  "error": np.minimum.accumulate(errors),
                                  "stable": np.asarray(stable), "raw": np.asarray(raw)}
        state = record["final_state"]
        final[(method, seed)] = {
            "stable_coefficient_norm": stable_coefficient_norm(state),
            "ordinary_kernel_weight_norm": ordinary_weight_norm(state),
            "relative_h1_error": error(state),
            "completed_pairs": sum(pair["completed"] for pair in state["pairs"]),
        }
        # The published reference used the same order-96 quadrature. Allow only
        # rounding differences from current NumPy/SciPy and platform versions.
        assert np.isclose(error(state), published_errors[str(seed)][method]["error"],
                          rtol=2e-4, atol=2e-12), (method, seed, error(state))
        print(f"{args.case}: {method}, seed {seed}; {len(work)} states; "
              f"elapsed {time.monotonic()-started:.1f}s", flush=True)

    with gzip.open(error_cache_path, "wt") as stream:
        json.dump({"source_sha256": sha256(record_path), "errors": error_cache}, stream)

    seeds = sorted(seed for method, seed in traces if method == "ordinary")
    escaping = max(seeds, key=lambda seed: np.nanmax(traces[("ordinary", seed)]["raw"]))
    best = {(metric, method): (min(seeds, key=lambda seed: traces[(method, seed)]["error"][-1])
                              if metric == "error" else escaping)
            for metric in ("error", "stable", "raw") for method in METHODS}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3.0))
    fig.subplots_adjust(left=.07, right=.99, bottom=.29, top=.88, wspace=.46)
    panels = (("error", "Relative $H^1$ error", "(a) State error"),
              ("stable", "Stable coefficient norm", "(b) Solver coordinates"),
              ("raw", "Ordinary kernel-weight norm", "(c) Finite representation"))
    for axis, (metric, ylabel, title) in zip(axes, panels):
        for method in METHODS:
            for seed in seeds:
                trace = traces[(method, seed)]
                bold = seed == best[(metric, method)]
                axis.plot(trace["work"], trace[metric], color=COLORS[method],
                          ls=STYLES[method], lw=1.45 if bold else .65,
                          alpha=1 if bold else .25, zorder=5 if bold else 2)
        axis.set(yscale="log", xscale="log", xlabel=work_label, ylabel=ylabel, title=title)
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", color=LIGHT_GRAY, lw=.45, alpha=.55)
    handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1.5, label=LABELS[m])
               for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .045), ncol=2,
               fontsize=7.5, columnspacing=1.25)
    outputs = save_figure(fig, output / figure_name)

    ranges = {}
    for method in METHODS:
        ranges[method] = {}
        for metric in ("stable_coefficient_norm", "ordinary_kernel_weight_norm"):
            values = [final[(method, seed)][metric] for seed in paper_seeds]
            assert all(value is None for value in values) or all(value is not None for value in values)
            ranges[method][metric] = None if values[0] is None else [min(values), max(values)]
        # Confirm the former mixed metric's finite branches are reproduced.
        if method != "completion":
            assert np.allclose(ranges[method]["ordinary_kernel_weight_norm"],
                               original_table["ranges"][method]["weight_norm"], rtol=1e-12)
    summary = {
        "source_repository": "https://github.com/danielfdzm/singular-representations-pde",
        "source_commit": "bf58b8fa2fb188681b24aeaae8e5fb25541acee9",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": experiment.scipy.__version__,
                     "torch": experiment.torch.__version__,
                     "matplotlib": sys.modules["matplotlib"].__version__},
        "source_record": str(record_path.relative_to(REPOSITORY)),
        "source_sha256": sha256(record_path),
        "evaluator_sha256": sha256(experiment_path),
        "quadrature_order": 96,
        "all_history_states_used": True,
        "paper_seeds": paper_seeds,
        "stable_coefficient_definition": "Euclidean norm of stored solver coefficients (a,b and singleton weights)",
        "ordinary_weight_definition": "Euclidean norm of a/2 +/- b/(2h) and singleton weights; undefined after any pair completes",
        "ranges": ranges,
        "per_start": {f"{method}:{seed}": values for (method, seed), values in final.items()},
        "stable_peak_norm": {f"{method}:{seed}": float(np.max(trace["stable"]))
                             for (method, seed), trace in traces.items()},
        "outputs": outputs,
        "validation": "Stored stable norms and published final H1 errors verified; finite raw ranges match original tables.",
    }
    (output / f"{args.case}_coefficient_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"ranges": ranges, "outputs": outputs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
