"""Fixed-multiplicity rerun of the archived Gaussian completion example.

The target and initial centres are inherited from the historical experiment;
the dictionaries are complete derivative blocks and the objective is the
whole-line H1 distance. Numerical optimization uses sampled least squares;
independent interval Gaussian overlaps enclose errors of exact stored states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import mpmath
from mpmath import iv, mp
import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.special import eval_hermitenorm, roots_legendre
import math


EXTRA_CENTRES = (-3.2, -2.1, 2.6)
OFFSETS = ((-7, -5, -3, -1, 1, 3, 5, 7), (-6, -2, 2, 6), (-4, 4), (0,))
MAX_ORDERS = (0, 1, 3, 7)
COLLAPSE_RATIOS = (0.35, 0.75, 0.86)
TARGET_BASIS = [[3, "0"], [0, "1.4"]]
TARGET_WEIGHTS = ["1", "0.6"]


def gaussian_derivative(x, order):
    return (-1.0) ** order * eval_hermitenorm(order, x) * np.exp(-0.5 * x * x)


def basis_for_stage(width, stage, h, b):
    basis = [[order, float(offset * h)]
             for offset in OFFSETS[stage]
             for order in range(MAX_ORDERS[stage] + 1)]
    return basis + [[0, float(b)]] + [[0, c] for c in EXTRA_CENTRES[:width - 9]]


def total_multiplicity(basis):
    blocks = {}
    for order, center in basis:
        blocks[center] = max(blocks.get(center, 0), order + 1)
    return sum(blocks.values())


def exact_interval(value):
    if isinstance(value, str):
        return iv.mpf(value)
    numerator, denominator = float(value).as_integer_ratio()
    return iv.mpf(numerator) / iv.mpf(denominator)


def outward_endpoints(value):
    lower = float(mp.mpf(value._mpi_[0]))
    upper = float(mp.mpf(value._mpi_[1]))
    return [float(np.nextafter(lower, -np.inf)), float(np.nextafter(upper, np.inf))]


def interval_h1_norm(basis, weights, digits=80):
    """Enclose a whole-line H1 norm, treating stored floats as exact reals."""
    iv.dps = digits
    mp.dps = digits + 15
    total = iv.mpf(0)
    interval_centers = [exact_interval(atom[1]) for atom in basis]
    interval_weights = [exact_interval(weight) for weight in weights]
    for j, ((r, _), a, wj) in enumerate(zip(basis, interval_centers, interval_weights)):
        for k in range(j + 1):
            s = basis[k][0]
            d = a - interval_centers[k]
            q = [iv.mpf(1), -d / 2]
            for n in range(1, r + s + 2):
                q.append(-d * q[-1] / 2 - iv.mpf(n) * q[-2] / 2)
            gaussian_overlap = iv.sqrt(iv.pi) * iv.exp(-d * d / 4)
            entry = (-1) ** r * gaussian_overlap * (q[r + s] - q[r + s + 2])
            total += (1 if j == k else 2) * wj * interval_weights[k] * entry
    lo, hi = outward_endpoints(total)
    if hi < 0:
        raise ArithmeticError("Negative upper bound for a squared H1 norm")
    norm_interval = iv.sqrt(iv.mpf([max(0.0, lo), max(0.0, hi)]))
    norm_lo, norm_hi = outward_endpoints(norm_interval)
    return {"squared_norm": [max(0.0, lo), hi],
            "norm": [max(0.0, norm_lo), norm_hi], "interval_decimal_digits": digits}


def audit_state(basis, weights):
    target_norm = interval_h1_norm(TARGET_BASIS, TARGET_WEIGHTS)
    residual_basis = basis + TARGET_BASIS
    residual_weights = list(weights) + ["-1", "-0.6"]
    error = interval_h1_norm(residual_basis, residual_weights)
    state = interval_h1_norm(basis, weights)
    relative = [float(np.nextafter(error["norm"][0] / target_norm["norm"][1], -np.inf)),
                float(np.nextafter(error["norm"][1] / target_norm["norm"][0], np.inf))]
    return {"relative_h1_error_interval": [max(0.0, relative[0]), relative[1]],
            "h1_error_interval": error["norm"], "h1_norm_interval": state["norm"],
            "objective_interval": [float(np.nextafter(0.5 * error["squared_norm"][0], -np.inf)),
                                   float(np.nextafter(0.5 * error["squared_norm"][1], np.inf))],
            "interval_decimal_digits": 80}


def moment_predictor(width, h, b, old_basis, old_weights):
    """Merge all eight ordinary cluster atoms into their full order-0..7 block."""
    next_basis = basis_for_stage(width, 3, 0.0, b)
    next_weights = np.zeros(width, dtype=float)
    for old_index in range(8):
        r, a = old_basis[old_index]
        for order in range(r, 8):
            next_weights[order] += (old_weights[old_index] * (-a) ** (order - r)
                                    / float(math.factorial(order - r)))
    next_weights[8:] = old_weights[8:]
    return next_basis, next_weights


class SampledFitter:
    def __init__(self, nodes=801):
        z, w = roots_legendre(nodes)
        self.x = 12.0 * z
        self.sqrt_weight = np.sqrt(12.0 * w)
        target = gaussian_derivative(self.x, 3) + 0.6 * gaussian_derivative(self.x - 1.4, 0)
        target_derivative = gaussian_derivative(self.x, 4) + 0.6 * gaussian_derivative(self.x - 1.4, 1)
        self.target = np.r_[target * self.sqrt_weight, target_derivative * self.sqrt_weight]
        self.target_norm = np.linalg.norm(self.target)
        self.fit_count = 0
        self.objective_evaluations = 0

    def fit(self, basis):
        self.fit_count += 1
        matrix = np.column_stack([
            np.r_[gaussian_derivative(self.x - c, r) * self.sqrt_weight,
                  gaussian_derivative(self.x - c, r + 1) * self.sqrt_weight]
            for r, c in basis])
        scales = np.linalg.norm(matrix, axis=0)
        normalized = matrix / scales
        solution, _, rank, singular_values = lstsq(normalized, self.target,
                                                   cond=1e-14, lapack_driver="gelsd")
        weights = solution / scales
        relative_error = np.linalg.norm(matrix @ weights - self.target) / self.target_norm
        return weights, float(relative_error), int(rank), float(singular_values[0] / singular_values[-1])

    def refine(self, width, stage, h, b, lower_h=None):
        if stage == 3:
            initial = [b]
            bounds = [(0.85, 3.5)]
        else:
            initial = [np.log(h), b]
            bounds = [(np.log(lower_h), np.log(h)), (0.85, 3.5)]
        def objective(parameters):
            self.objective_evaluations += 1
            trial_h = 0.0 if stage == 3 else float(np.exp(parameters[0]))
            trial_b = float(parameters[-1])
            _, error, _, _ = self.fit(basis_for_stage(width, stage, trial_h, trial_b))
            return error * error
        probes = []
        optimizer_iterations = 0
        def callback(parameters):
            nonlocal optimizer_iterations
            optimizer_iterations += 1
            if stage == 3:
                return
            trial_h = float(np.exp(parameters[0]))
            if trial_h > 0.175 * (1 + 1e-12):
                return
            trial_b = float(parameters[-1])
            trial_basis = basis_for_stage(width, 0, trial_h, trial_b)
            trial_weights, _, _, _ = self.fit(trial_basis)
            predictor_basis, predictor_weights = moment_predictor(
                width, trial_h, trial_b, trial_basis, trial_weights)
            state_norm = interval_h1_norm(trial_basis, trial_weights)
            change = interval_h1_norm(trial_basis + predictor_basis,
                                      trial_weights.tolist() + (-predictor_weights).tolist())
            allowed = float(np.nextafter(0.03 * state_norm["norm"][0], -np.inf))
            passed = change["norm"][1] <= allowed
            probes.append({"optimizer_iteration": optimizer_iterations,
                           "cumulative_linear_fits": self.fit_count,
                           "cumulative_objective_evaluations": self.objective_evaluations,
                           "basis": trial_basis, "weights": trial_weights.tolist(),
                           "predictor_basis": predictor_basis,
                           "predictor_weights": predictor_weights.tolist(),
                           "h1_change_interval": change["norm"],
                           "allowed_change_lower": allowed, "passed": passed})
            if passed:
                raise StopIteration
        result = minimize(objective, initial, method="Nelder-Mead", bounds=bounds,
                          callback=callback,
                          options={"xatol": 1e-11 if stage == 3 else 1e-9,
                                   "fatol": 1e-25 if stage == 3 else 1e-15, "maxiter": 1000})
        final_h = 0.0 if stage == 3 else float(np.exp(result.x[0]))
        final_b = float(result.x[-1])
        basis = basis_for_stage(width, stage, final_h, final_b)
        weights, error, rank, condition = self.fit(basis)
        return {"stage": stage, "basis": basis, "weights": weights.tolist(),
                "h": final_h, "b": final_b, "sampled_relative_h1_error": error,
                "multiplicity": total_multiplicity(basis), "coefficient_norm": float(np.linalg.norm(weights)),
                "sampled_design_rank": rank, "sampled_normalized_design_condition": condition,
                "cumulative_linear_fits": self.fit_count,
                "cumulative_objective_evaluations": self.objective_evaluations,
                "optimizer": {"method": "Nelder-Mead with variable projection", "success": bool(result.success),
                              "message": str(result.message), "iterations": int(result.nit),
                              "function_evaluations": int(result.nfev),
                              "termination_reason": ("first eligible iterate passing the whole-cluster transfer gate"
                                                     if probes and probes[-1]["passed"]
                                                     else "geometry optimizer termination"),
                              "transfer_gate_probes": probes},
                **audit_state(basis, weights)}


def run_width(width, nodes):
    fitter = SampledFitter(nodes)
    h, b = 0.5, 2.25
    initial_basis = basis_for_stage(width, 0, h, b)
    initial_weights, initial_error, rank, condition = fitter.fit(initial_basis)
    # One damped Newton step from zero in the quadratic coefficient problem.
    # Later variable-projection evaluations use the full least-squares solve.
    initial_full_fit_error = initial_error
    initial_weights *= 0.8
    initial_error = float(np.sqrt(0.2**2 + (1.0 - 0.2**2) * initial_full_fit_error**2))
    initial_state = {"label": "initial", "stage": 0, "basis": initial_basis,
                     "weights": initial_weights.tolist(), "h": h, "b": b,
                     "multiplicity": total_multiplicity(initial_basis),
                     "coefficient_norm": float(np.linalg.norm(initial_weights)),
                     "coefficient_step_fraction": 0.8,
                     "undamped_sampled_relative_h1_error": initial_full_fit_error,
                     "cumulative_linear_fits": fitter.fit_count,
                     "cumulative_objective_evaluations": fitter.objective_evaluations,
                     "sampled_design_rank": rank, "sampled_normalized_design_condition": condition,
                     "sampled_relative_h1_error": initial_error, **audit_state(initial_basis, initial_weights)}
    proposals = []
    for attempt in range(12):
        # A genuine bounded geometry minimization precedes each tested moment
        # conversion. A rejection retains the finite dictionary and state;
        # the next attempt starts a fresh bounded geometry solve at smaller h.
        lower_h = h * (0.35 if attempt == 0 else 0.75)
        finite = fitter.refine(width, 0, h, b, lower_h)
        finite["label"] = "finite"
        h, b = finite["h"], finite["b"]
        new_basis, predictor = moment_predictor(width, h, b, finite["basis"], finite["weights"])
        change = interval_h1_norm(finite["basis"] + new_basis,
                                  finite["weights"] + (-predictor).tolist())
        allowed = float(np.nextafter(0.03 * finite["h1_norm_interval"][0], -np.inf))
        passes = change["norm"][1] <= allowed and h <= 0.175 * (1 + 1e-12)
        proposal = {"before": finite, "predictor_basis": new_basis,
                    "predictor_weights": predictor.tolist(),
                    "predictor_multiplicity": total_multiplicity(new_basis),
                    "unrefitted_h1_change_interval": change["norm"],
                    "relative_unrefitted_h1_change_upper": float(np.nextafter(change["norm"][1] / finite["h1_norm_interval"][0], np.inf)),
                    "relative_transfer_tolerance": 0.03, "allowed_change_lower": allowed,
                    "transfer_gate_passed": passes, "cluster_diameter": 14 * h,
                    "eligibility_max_h": 0.175}
        proposals.append(proposal)
        if passes:
            break
    else:
        raise RuntimeError(f"Width {width}: no admissible moment transfer")
    completed = fitter.refine(width, 3, 0.0, b)
    completed["label"] = "completed"
    transfer = proposals[-1]
    transfer["optimized_candidate_objective_interval"] = completed["objective_interval"]
    transfer["objective_gate_passed"] = completed["objective_interval"][1] < finite["objective_interval"][0]
    transfer["objective_decrease_lower"] = float(np.nextafter(
        finite["objective_interval"][0] - completed["objective_interval"][1], -np.inf))
    transfer["accepted"] = transfer["transfer_gate_passed"] and transfer["objective_gate_passed"]
    if not transfer["accepted"]:
        raise RuntimeError(f"Width {width}: completed candidate failed objective descent")
    snapshots = [initial_state, finite, completed]
    assert all(state["multiplicity"] == width for state in snapshots)
    return {"initial_width": width, "snapshots": snapshots, "proposals": proposals,
            "transfer": transfer, "final": completed,
            "linear_fits": fitter.fit_count, "objective_evaluations": fitter.objective_evaluations}


def audit_results(results_path, audit_output):
    """Replay saved states without rerunning or changing their optimization."""
    payload = json.loads(results_path.read_text())
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if payload["provenance"]["source_sha256"] != source_hash:
        raise ValueError("The saved results do not match this source revision")
    validator = SampledFitter(1601)
    records = []
    for run in payload["runs"]:
        for snapshot in run["snapshots"]:
            basis, weights = snapshot["basis"], np.asarray(snapshot["weights"])
            assert total_multiplicity(basis) == run["initial_width"]
            values = sum(w * gaussian_derivative(validator.x - c, r)
                         for (r, c), w in zip(basis, weights))
            derivatives = sum(w * gaussian_derivative(validator.x - c, r + 1)
                              for (r, c), w in zip(basis, weights))
            error = np.linalg.norm(np.r_[values * validator.sqrt_weight,
                                        derivatives * validator.sqrt_weight]
                                   - validator.target) / validator.target_norm
            fine = interval_h1_norm(basis + TARGET_BASIS,
                                    snapshot["weights"] + ["-1", "-0.6"], digits=110)
            coarse = snapshot["h1_error_interval"]
            assert coarse[0] <= fine["norm"][0] and fine["norm"][1] <= coarse[1]
            records.append({"width": run["initial_width"], "snapshot": snapshot["label"],
                            "sampled_relative_h1_error_1601": float(error),
                            "wholeline_h1_error_interval_110digits": fine["norm"],
                            "stored_80digit_interval_encloses_110digit_check": True})
        for proposal in run["proposals"]:
            fine = interval_h1_norm(
                proposal["before"]["basis"] + proposal["predictor_basis"],
                proposal["before"]["weights"] + [-w for w in proposal["predictor_weights"]], digits=110)
            coarse = proposal["unrefitted_h1_change_interval"]
            assert coarse[0] <= fine["norm"][0] and fine["norm"][1] <= coarse[1]
    result = {"source_sha256": source_hash,
              "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
              "method": "Independent 1601-node Gauss-Legendre state evaluation and 110-digit whole-line interval replay",
              "records": records}
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(records)} stored snapshots; wrote {audit_output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[9, 10, 12])
    parser.add_argument("--nodes", type=int, default=801)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON (defaults under outputs/completion_consistent)")
    parser.add_argument("--audit", action="store_true", help="Replay saved results without fresh optimization")
    parser.add_argument("--input", type=Path, default=root / "data/completion_consistent/results.json",
                        help="Saved results to replay with --audit")
    args = parser.parse_args()
    if args.output is None:
        args.output = root / "outputs/completion_consistent" / ("audit.json" if args.audit else "results.json")
    if args.audit:
        audit_results(args.input, args.output)
        return
    started = time.perf_counter()
    runs = []
    for width in args.widths:
        result = run_width(width, args.nodes)
        runs.append(result)
        print(json.dumps({"width": width, "errors": [s["relative_h1_error_interval"] for s in result["snapshots"]],
                          "transfer_gate": result["transfer"]["transfer_gate_passed"],
                          "linear_fits": result["linear_fits"]}), flush=True)
    payload = {"experiment": "Fixed-multiplicity whole-line H1 Gaussian completion rerun",
               "objective": "0.5 ||u-u*||_{H1(R)}^2", "target_basis": TARGET_BASIS,
               "target_weights": TARGET_WEIGHTS,
               "configuration": {"initial_h": 0.5, "initial_primary_center": 2.25,
                                 "initial_coefficient_step_fraction": 0.8,
                                 "extra_centers": list(EXTRA_CENTRES), "derivative_orders_completed_block": list(range(8)),
                                 "first_contraction_ratio": 0.35, "subsequent_contraction_ratio": 0.75, "geometry_method": "bounded Nelder-Mead",
                                 "coefficient_solver": "column-normalized sampled SVD, relative cutoff 1e-14",
                                 "quadrature": "Gauss-Legendre on [-12,12]", "quadrature_nodes": args.nodes,
                                 "transfer_relative_tolerance": 0.03, "interval_decimal_digits": 80,
                                 "geometry_bounds_primary_center": [0.85, 3.5]},
               "provenance": {"historical_source_commit": "8b15f8714af3f87abcde9fa484eddd7d66343f06",
                              "verified_github_main_commit": "bf58b8fa2fb188681b24aeaae8e5fb25541acee9",
                              "historical_elements_retained": "Gaussian target, widths, initial centres and initial contraction ratio",
                              "new_elements": "one whole-cluster order-0..7 block, H1 objective, variable projection, interval audits",
                              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "mpmath": mpmath.__version__},
               "runs": runs, "elapsed_seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
