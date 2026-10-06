#!/usr/bin/env python3
"""Replay the main ordinary-weight/derivative-completion comparisons.

This NumPy evaluator is independent of the training drivers. Finite columns
are individual translates; completed columns are direct kernel derivatives.
It never converts a finite pair into a divided-difference basis.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[2]
METHODS = ("ordinary", "completion")
FOLDERS = {"matern": "matern_swift_hohenberg_2d", "poisson": "poisson_neumann_2d"}
spec = importlib.util.spec_from_file_location("direct_matern_radial", CODE / "experiments" / FOLDERS["matern"] / "matern.py")
matern = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matern)


@lru_cache(maxsize=None)
def literal(name, study):
    tree = ast.parse((CODE / "experiments" / FOLDERS[study] / "experiment.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def target_state(study):
    target = literal("MANUFACTURED_JET", study)
    result = {"pairs": [{"center": c, "angle": angle, "half_gap": 0.0, "completed": True}
                        for c, angle, a, b in target["pairs"]],
              "coefficients": [v for c, angle, a, b in target["pairs"] for v in (a, b)]}
    if study == "matern":
        result["background_center"], coefficient = target["background"]
        result["coefficients"].append(coefficient)
    else:
        result["singletons"] = [c for c, a in target["singletons"]]
        result["coefficients"].extend(a for c, a in target["singletons"])
    return result


def quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel())), (weights[:, None]*weights[None, :]).ravel()


def basis_fields(study, state, points):
    points = np.asarray(points)
    values, gradients, laps = [], [], []
    q = 1.0/literal("SCALE", study)**2 if study == "poisson" else None

    def add(center, direction=None):
        r = points-np.asarray(center)
        r2 = np.sum(r*r, axis=1)
        if study == "matern":
            radius = np.sqrt(r2)
            k, c1 = matern.radial("f0", radius), matern.radial("c1", radius)
            gradient, lap = r*c1[:, None], matern.radial("lap", radius)
        else:
            k = np.exp(-0.5*q*r2)
            gradient, lap = -q*r*k[:, None], (q*q*r2-2*q)*k
        values.append(k); gradients.append(gradient); laps.append(lap)
        if direction is not None:
            t = r@direction
            if study == "matern":
                psi = -c1*t
                grad_psi = -(matern.radial("c1p", radius)[:, None]*t[:, None]*r+c1[:, None]*direction)
                lap_psi = -matern.radial("c3", radius)*t
            else:
                psi = q*t*k
                grad_psi = k[:, None]*(q*direction-q*q*t[:, None]*r)
                lap_psi = q*t*(q*q*r2-4*q)*k
            values.append(psi); gradients.append(grad_psi); laps.append(lap_psi)

    for pair in state["pairs"]:
        center = np.asarray(pair["center"])
        direction = np.array((math.cos(pair["angle"]), math.sin(pair["angle"])))
        if pair["completed"]:
            if pair["half_gap"] != 0:
                raise ValueError("A completed pair must have zero gap")
            add(center, direction)
        else:
            if pair["half_gap"] <= 0:
                raise ValueError("A finite pair must have positive gap")
            add(center+pair["half_gap"]*direction)
            add(center-pair["half_gap"]*direction)
    for center in ([state["background_center"]] if study == "matern" else state["singletons"]):
        add(center)
    return np.stack(values, axis=1), np.stack(gradients, axis=1), np.stack(laps, axis=1)


def evaluate_state(study, state, points):
    values, gradients, laps = basis_fields(study, state, points)
    c = np.asarray(state["coefficients"])
    return values@c, np.einsum("npd,p->nd", gradients, c), laps@c


def read_results(directory):
    path = directory / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(directory / "results.json.gz", "rt") as stream:
        return json.load(stream)


def fingerprint(run):
    return hashlib.sha256(json.dumps(run, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def analyse(study, directory, order=96, histories=True):
    payload = read_results(directory)
    runs = payload["runs"]
    seeds = list(range(10)) if study == "matern" else list(range(10, 20))
    expected = {(seed, method) for seed in seeds for method in METHODS}
    if len(runs) != len(expected) or {(r["seed"], r["method"]) for r in runs} != expected:
        raise ValueError("Expected ten matched starts and exactly two methods")
    by = {(r["seed"], r["method"]): r for r in runs}
    for seed in seeds:
        left, right = (by[seed, m]["initial_state"] for m in METHODS)
        for key in ("background_center", "singletons"):
            if key in left and left[key] != right[key]:
                raise ValueError(f"Unmatched initial geometry: {study} {seed} {key}")
        for first, second in zip(left["pairs"], right["pairs"]):
            for key in ("center", "angle", "completed"):
                if first[key] != second[key]:
                    raise ValueError(f"Unmatched initial geometry: {study} {seed} {key}")
            if not np.isclose(first["half_gap"], second["half_gap"], rtol=2e-15, atol=0):
                raise ValueError("Unmatched initial half-gap")
    points, weights = quadrature(order)
    exact, grad_exact, lap_exact = evaluate_state(study, target_state(study), points)
    target_h1_sq = float(weights@(exact*exact+np.sum(grad_exact*grad_exact, axis=1)))
    target_quadratic_sq = float(weights@((exact+lap_exact)**2+exact*exact))

    def measure(state):
        value, gradient, lap = evaluate_state(study, state, points)
        error, grad_error = value-exact, gradient-grad_exact
        h1 = float(np.sqrt((weights@(error*error+np.sum(grad_error*grad_error, axis=1)))/target_h1_sq))
        eq = float(np.sqrt((weights@((error+lap-lap_exact)**2+error*error))/target_quadratic_sq)) if study == "matern" else None
        return h1, eq

    traces, final = {}, {method: [] for method in METHODS}
    for run in runs:
        history = run["history"] if histories else [{"state": run["final_state"], "solve_count": run["coefficient_solves"], "phase": "final"}]
        if not history or history[-1]["state"] != run["final_state"]:
            raise ValueError("History does not end at the accepted returned state")
        trace = {k: [] for k in ("work", "h1_error", "coefficient_norm", "phase")}
        cached, last_eq = {}, None
        for row in history:
            state = row["state"]
            signature = json.dumps(state, sort_keys=True)
            if signature not in cached:
                cached[signature] = measure(state)
            h1, last_eq = cached[signature]
            trace["work"].append(row["solve_count"])
            trace["h1_error"].append(h1)
            trace["coefficient_norm"].append(float(np.linalg.norm(state["coefficients"])))
            trace["phase"].append(row["phase"])
        if any(b < a for a, b in zip(trace["work"], trace["work"][1:])):
            raise ValueError("Decreasing work counter")
        trace["running_min_h1_error"] = np.minimum.accumulate(trace["h1_error"]).tolist()
        traces[f"{run['method']}:{run['seed']}"] = trace
        final[run["method"]].append({"seed": run["seed"], "h1_error": trace["h1_error"][-1],
            "quadratic_error": last_eq, "coefficient_norm": trace["coefficient_norm"][-1],
            "minimum_half_gap": min(p["half_gap"] for p in run["final_state"]["pairs"]),
            "solves": run["coefficient_solves"], "status": run["status"], "error": run.get("error"),
            "transfers": len(run["transfers"]), "completed_pairs": sum(p["completed"] for p in run["final_state"]["pairs"])})
    summary = {}
    for method, rows in final.items():
        def span(key):
            return [min(r[key] for r in rows), max(r[key] for r in rows)]
        summary[method] = {"error_range": span("h1_error"), "median_h1_error": statistics.median(r["h1_error"] for r in rows),
            "coefficient_norm_range": span("coefficient_norm"), "half_gap_range": span("minimum_half_gap"),
            "solves_range": span("solves"), "median_solves": statistics.median(r["solves"] for r in rows),
            "transfers_range": span("transfers"), "completed_pairs_range": span("completed_pairs"),
            "status_counts": dict(Counter(r["status"] for r in rows))}
        if study == "matern":
            summary[method]["quadratic_error_range"] = span("quadratic_error")
    thresholds = []
    if histories:
        for tau in (1e-3, 1e-5, 1e-7):
            threshold = {"threshold": tau, "methods": {}}
            for method in METHODS:
                hits, per_seed = [], {}
                for seed in seeds:
                    trace = traces[f"{method}:{seed}"]
                    idx = next((i for i, e in enumerate(trace["h1_error"]) if e <= tau), None)
                    work = None if idx is None else trace["work"][idx]
                    per_seed[str(seed)] = work
                    if work is not None:
                        hits.append(work)
                threshold["methods"][method] = {"reached": len(hits), "median_solves": statistics.median(hits) if hits else None,
                                                "first_hit_solves": per_seed}
            thresholds.append(threshold)
    return {"study": study, "quadrature_order": order, "histories": histories, "seeds": seeds,
        "methods": list(METHODS), "coefficient_semantics": "actual weights of direct finite translates or completed kernel/derivative blocks",
        "run_sha256": {f"{r['method']}:{r['seed']}": fingerprint(r) for r in runs}, "summary": summary,
        "final": final, "history_metrics": traces, "threshold_work": thresholds,
        "target_coefficient_norm": float(np.linalg.norm(target_state(study)["coefficients"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(FOLDERS), required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.results or CODE / "data/raw_main" / args.study
    result = analyse(args.study, directory, args.quadrature_order, not args.final_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
