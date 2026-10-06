"""Matched-start diagnostics from the stored two-dimensional trajectories.

The records contain the finite branch, completion and, in the revised runs,
the restart and squared-gap controls. Every method present is analysed at
every start. The training histories contain state norms, not errors against
the exact solution, so every error below is reevaluated with independent
Gauss--Legendre quadrature. Threshold work counts the first stored state
meeting the threshold; it is not the cost to a prescribed stopping rule.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import statistics
import numpy as np

METHODS = ("ordinary", "restart", "squared_gap", "completion")
DEFAULT_THRESHOLDS = (1e-3, 1e-5, 1e-7)


def methods_in(runs):
    """Methods present in the records, in the canonical order."""
    present = {run["method"] for run in runs}
    return tuple(method for method in METHODS if method in present)


def run_fingerprint(run):
    return hashlib.sha256(json.dumps(run, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def error_measure(module, order=96):
    points, weights = module.quadrature(order)
    exact = module.manufactured_jet_solution(points[:, 0], points[:, 1])
    gradient = module.manufactured_jet_gradient(points[:, 0], points[:, 1])
    squared_norm = float(weights @ (exact**2 + np.sum(gradient**2, axis=1)))

    def measure(state):
        value, derivative = module.evaluate_state(state, points)
        difference = value - exact
        return float(np.sqrt((weights @ (difference**2
                           + np.sum((derivative-gradient)**2, axis=1))) / squared_norm))

    return measure


def coefficient_norms(state):
    coefficients = state["coefficients"]
    atoms = []
    for index, pair in enumerate(state["pairs"]):
        a, b = coefficients[2*index:2*index+2]
        if pair["completed"]:
            atoms.extend((a, b))
        else:
            h = pair["half_gap"]
            atoms.extend((a/2+b/(2*h), a/2-b/(2*h)))
    atoms.extend(coefficients[2*len(state["pairs"]):])
    return float(np.linalg.norm(coefficients)), float(np.linalg.norm(atoms))


def history_trace(run, measure):
    """Retain current errors and coefficients separately from running minima."""
    history = run["history"]
    if not history:
        raise ValueError(f"Empty history for {run['method']} start {run['seed']}")
    errors, stored, atoms, work, phases = [], [], [], [], []
    previous_state, previous_error = None, None
    for entry in history:
        state = entry["state"]
        if state != previous_state:
            previous_error = measure(state)
            previous_state = state
        stable_norm, atom_norm = coefficient_norms(state)
        errors.append(previous_error)
        stored.append(stable_norm)
        atoms.append(atom_norm)
        work.append(int(entry["solve_count"]))
        phases.append(entry["phase"])
    if any(b < a for a, b in zip(work, work[1:])):
        raise ValueError("History solve counts must be nondecreasing")
    if history[-1]["state"] != run["final_state"]:
        raise ValueError("Last stored state does not match the returned state")
    return {"work": work, "h1_error": errors,
            "running_min_h1_error": np.minimum.accumulate(errors).tolist(),
            "stable_coefficient_norm": stored, "atom_jet_norm": atoms,
            "phase": phases}


def threshold_work(runs, traces, thresholds=DEFAULT_THRESHOLDS):
    """Report all starts, including misses and abnormal optimizer returns."""
    seeds = sorted({run["seed"] for run in runs})
    methods = methods_in(runs)
    by = {(run["seed"], run["method"]): run for run in runs}
    result = []
    for threshold in thresholds:
        summary, hits = {}, {}
        for method in methods:
            per_start = {}
            for seed in seeds:
                run = by[(seed, method)]
                trace = traces[f"{method}:{seed}"]
                first = next((i for i, value in enumerate(trace["h1_error"])
                              if value <= threshold), None)
                count = None if first is None else trace["work"][first]
                hits[(seed, method)] = count
                per_start[str(seed)] = {
                    "reached": first is not None, "first_recorded_hit_solves": count,
                    "first_recorded_hit_phase": None if first is None else trace["phase"][first],
                    "first_recorded_hit_error": None if first is None else trace["h1_error"][first],
                    "returned_error": trace["h1_error"][-1],
                    "returned_state_meets_threshold": trace["h1_error"][-1] <= threshold,
                    "optimizer_status": run["status"]}
            successful = [row["first_recorded_hit_solves"] for row in per_start.values() if row["reached"]]
            summary[method] = {
                "total_starts": len(seeds), "reached_count": len(successful),
                "not_reached_count": len(seeds)-len(successful),
                "returned_state_meets_threshold_count": sum(row["returned_state_meets_threshold"] for row in per_start.values()),
                "median_solves_among_reaching_starts": statistics.median(successful) if successful else None,
                "optimizer_status_counts_all_starts": dict(Counter(row["optimizer_status"] for row in per_start.values())),
                "optimizer_status_counts_reaching_starts": dict(Counter(row["optimizer_status"] for row in per_start.values() if row["reached"])),
                "per_start": per_start}
        paired = [seed for seed in seeds if all(hits[(seed, m)] is not None for m in ("ordinary", "completion"))]
        ratios = [hits[(seed, "ordinary")]/hits[(seed, "completion")] for seed in paired]
        result.append({"relative_h1_threshold": threshold, "methods": summary,
                       "paired_reaching_count": len(paired), "paired_reaching_seeds": paired,
                       "median_ordinary_to_completion_solves_on_paired_reaching_starts": statistics.median(ratios) if ratios else None})
    return result


def final_trace(run, measure):
    """Returned-state metrics only, without replaying the history."""
    stable_norm, atom_norm = coefficient_norms(run["final_state"])
    error = measure(run["final_state"])
    return {"h1_error": [error], "running_min_h1_error": [error],
            "stable_coefficient_norm": [stable_norm], "atom_jet_norm": [atom_norm]}


def analyse(payload, module, order=96, thresholds=DEFAULT_THRESHOLDS, histories=True):
    """With histories=False only returned states are evaluated: no threshold work."""
    runs = [run for run in payload["runs"] if run["method"] in METHODS]
    methods = methods_in(runs)
    seeds = sorted({run["seed"] for run in runs})
    expected = {(seed, method) for seed in seeds for method in methods}
    if {(run["seed"], run["method"]) for run in runs} != expected or len(runs) != len(expected):
        raise ValueError("Expected exactly one run of every method at every start")
    measure = error_measure(module, order)
    table, traces = {}, {}
    for run in runs:
        seed, method = run["seed"], run["method"]
        trace = history_trace(run, measure) if histories else final_trace(run, measure)
        traces[f"{method}:{seed}"] = trace
        state = run["final_state"]
        table.setdefault(seed, {})[method] = {
            "error": trace["h1_error"][-1],
            "best_recorded_error": trace["running_min_h1_error"][-1],
            "stable_coefficient_norm": trace["stable_coefficient_norm"][-1],
            "weight_norm": trace["atom_jet_norm"][-1],
            "status": run["status"], "solves": run["coefficient_solves"],
            "max_scaled_coefficient_quadratic_condition": run["maximum_gradient_basis_condition"],
            "max_relative_coefficient_stationarity": run["maximum_relative_coefficient_stationarity"],
            "transfers": len(run["transfers"]), "restarts": len(run.get("restart_events", [])),
            "energy": run["independent_quadrature"][-1]["energy"],
            "min_half_gap": min((p["half_gap"] for p in state["pairs"] if not p["completed"]), default=0.0)}
        print(f"start={seed} {method}: H1(returned)={trace['h1_error'][-1]:.4e}; {run['status']}", flush=True)
    keys = ("error", "stable_coefficient_norm", "weight_norm", "min_half_gap", "solves", "transfers",
            "restarts", "max_scaled_coefficient_quadratic_condition", "max_relative_coefficient_stationarity")

    def ranges(selected):
        return {method: {key: [min(table[s][method][key] for s in selected),
                                      max(table[s][method][key] for s in selected)]
                         for key in keys} for method in methods}

    def medians(selected):
        return {method: {key: statistics.median(table[s][method][key] for s in selected)
                         for key in ("error", "solves")} for method in methods}

    pairs = len(payload["runs"][0]["final_state"]["pairs"])
    transferring = [seed for seed in seeds if "completion" in methods
                    and table[seed]["completion"]["transfers"] == pairs]
    return {"protocol": {
                "quadrature_order": order, "error": "relative H1 against the manufactured solution",
                "threshold_cost": "First recorded initial, accepted, transfer, or final state with error <= threshold; line-search coefficient fits are included in its solve count.",
                "threshold_scope": "All starts and optimizer statuses are retained. Threshold attainment and successful optimizer termination are distinct. Medians exclude unreached starts and are labelled accordingly.",
                "history_note": "Training histories store H1 state norms, not target errors. Errors here are independently reevaluated; running minima are separate from returned-state errors.",
                "conditioning_note": "Recorded maximum condition number of the column-scaled quadratic coefficient matrix; this is neither the nonlinear coefficient Hessian nor the geometry Hessian."},
            "methods": list(methods), "training_quadrature_order": payload["configuration"]["quadrature_order"],
            "paper_seeds": seeds, "per_start": table, "paper_table": ranges(seeds), "medians": medians(seeds),
            "status_counts": {method: dict(Counter(table[s][method]["status"] for s in seeds)) for method in methods},
            "transferring_subgroup": {"definition": "starts on which completion transfers every pair",
                                      "seeds": transferring, "ranges": ranges(transferring) if transferring else {},
                                      "medians": medians(transferring) if transferring else {}},
            "run_sha256": {f"{run['method']}:{run['seed']}": run_fingerprint(run) for run in runs},
            "shared_threshold_work": threshold_work(runs, traces, thresholds) if histories else None,
            "history_metrics": traces if histories else None}


def reference_summaries(result):
    """The compact summaries stored beside the records in data/<study>/."""
    return {
        "h1_errors.json": result["per_start"],
        "paper_table.json": {"paper_seeds": result["paper_seeds"], "methods": result["methods"],
                             "training_quadrature_order": result["training_quadrature_order"],
                             "evaluation_quadrature_order": result["protocol"]["quadrature_order"],
                             "scope": "All ten starts, including partial-transfer and abnormal optimizer returns.",
                             "ranges": result["paper_table"], "medians": result["medians"],
                             "status_counts": result["status_counts"],
                             "transferring_subgroup": result["transferring_subgroup"]},
        "shared_threshold_work.json": {"protocol": result["protocol"], "thresholds": result["shared_threshold_work"]}}


def print_summary(result):
    print("\nReturned-state ranges over all starts:")
    print(json.dumps(result["paper_table"], indent=2))
    if result["shared_threshold_work"] is None:
        return
    print("\nFirst recorded threshold hit: reached/all; median fits among reaching starts")
    for row in result["shared_threshold_work"]:
        values = [f"{m} {v['reached_count']}/{v['total_starts']}, median {v['median_solves_among_reaching_starts']}"
                  for m, v in row["methods"].items()]
        print(f"H1 <= {row['relative_h1_threshold']:.0e}: " + "; ".join(values))
