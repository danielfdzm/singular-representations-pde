#!/usr/bin/env python3
"""Poisson paper starts trained with ordinary Gaussian weights and raw columns.

Only the linear basis changes: geometry, log gaps, variable projection,
quadrature, coefficient acceptance and L-BFGS-B settings match the paper's
standard branch. All training arithmetic evaluates translated Gaussians
directly. Stable coefficients and evaluations below are passive diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
PAPER_SOURCE = ROOT / "experiments/poisson_neumann_2d/experiment.py"
spec = importlib.util.spec_from_file_location("paper_poisson_for_raw_coordinates", PAPER_SOURCE)
E = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = E
spec.loader.exec_module(E)
np, torch = E.np, E.torch

quadrature = E.quadrature
manufactured_jet_solution = E.manufactured_jet_solution
manufactured_jet_gradient = E.manufactured_jet_gradient


def annotate_state(state):
    """The authoritative coefficient vector always contains raw weights."""
    state["raw_coefficients"] = list(state["coefficients"])
    stable = []
    for index, pair in enumerate(state["pairs"]):
        plus, minus = state["coefficients"][2*index:2*index+2]
        stable.extend((plus+minus, pair["half_gap"]*(plus-minus)))
    stable.extend(state["coefficients"][2*len(state["pairs"]):])
    state["derived_stable_coefficients"] = stable
    state["coefficient_coordinates"] = "ordinary_translate_weights"
    state["pair_column_order"] = "center + h*d, then center - h*d"
    return state


def to_stable_state(state):
    result = copy.deepcopy(state)
    annotate_state(result)
    result["coefficients"] = result["derived_stable_coefficients"]
    result["coefficient_coordinates"] = "derived_pair_coefficients_for_audit_only"
    return result


def basis_fields(state, points):
    """Raw K(x-c-hd), K(x-c+hd), singleton columns and spatial derivatives."""
    points = np.asarray(points, dtype=float)
    q = 1.0/E.SCALE**2
    values, gradients, laplacians = [], [], []
    centres = []
    for pair in state["pairs"]:
        if pair["completed"]:
            raise ValueError("The raw baseline contains no completed pairs")
        direction = np.array((math.cos(pair["angle"]), math.sin(pair["angle"])))
        centre = np.asarray(pair["center"])
        centres.extend((centre+pair["half_gap"]*direction, centre-pair["half_gap"]*direction))
    centres.extend(np.asarray(centre) for centre in state["singletons"])
    for centre in centres:
        r = points-centre
        r2 = np.sum(r*r, axis=1)
        kernel = np.exp(-0.5*q*r2)
        values.append(kernel)
        gradients.append(-q*r*kernel[:, None])
        laplacians.append((q*q*r2-2*q)*kernel)
    return np.stack(values, axis=1), np.stack(gradients, axis=1), np.stack(laplacians, axis=1)


def evaluate_state(state, points):
    values, gradients, _ = basis_fields(state, points)
    coefficients = np.asarray(state["coefficients"])
    return values @ coefficients, np.einsum("npd,p->nd", gradients, coefficients)


def torch_basis(vector, template, points):
    values, gradients, cursor = [], [], 0
    q = 1.0/E.SCALE**2
    centres = []
    for pair in template["pairs"]:
        if pair["completed"]:
            raise ValueError("Raw-coordinate optimization cannot complete pairs")
        centre, angle = vector[cursor:cursor+2], vector[cursor+2]
        h = torch.exp(vector[cursor+3])
        cursor += 4
        direction = torch.stack((torch.cos(angle), torch.sin(angle)))
        centres.extend((centre+h*direction, centre-h*direction))
    for _ in template["singletons"]:
        centres.append(vector[cursor:cursor+2])
        cursor += 2
    for centre in centres:
        r = points-centre
        kernel = torch.exp(-0.5*q*torch.sum(r*r, dim=1))
        values.append(kernel)
        gradients.append(-q*r*kernel[:, None])
    return torch.stack(values, dim=1), torch.stack(gradients, dim=1)


def relative_h1(state, order=96, stable_evaluation=False):
    points, weights = E.quadrature(order)
    exact = E.manufactured_jet_solution(*points.T)
    exact_gradient = E.manufactured_jet_gradient(*points.T)
    value, gradient = (E.evaluate_state(to_stable_state(state), points) if stable_evaluation
                       else evaluate_state(state, points))
    return float(np.sqrt(weights @ ((value-exact)**2+np.sum((gradient-exact_gradient)**2, axis=1))
                         / (weights @ (exact**2+np.sum(exact_gradient**2, axis=1)))))


def energy(state, order):
    points, weights = E.quadrature(order)
    value, gradient = evaluate_state(state, points)
    force = E.manufactured_jet_load(*points.T)
    boundary_points, boundary_weights, normals = E.boundary_quadrature(order)
    boundary_value, _ = evaluate_state(state, boundary_points)
    gradient_term = float(0.5*weights @ np.sum(gradient*gradient, axis=1))
    load = float(weights @ (force*value))
    boundary = float(boundary_weights @ (E.manufactured_neumann_data(boundary_points, normals)*boundary_value))
    mean = float(weights @ value)
    mean_term = float(0.5*E.SIGMA*(mean-E.TARGET_MEAN)**2)
    return {"quadrature_order": order, "energy": gradient_term-load-boundary+mean_term,
            "gradient_energy": gradient_term, "load": load, "neumann_work": boundary,
            "mean_term": mean_term, "state_mean": mean,
            "h1_norm": float(np.sqrt(weights @ (value*value+np.sum(gradient*gradient, axis=1)))),
            "stable_coefficient_norm": float(np.linalg.norm(state["derived_stable_coefficients"])),
            "raw_weight_norm": float(np.linalg.norm(state["coefficients"])),
            "kernel_weight_norm": float(np.linalg.norm(state["coefficients"])),
            "completed_pairs": 0, "multiplicity_budget": E.NSLOTS}


class RawCoefficientFailure(E.NumericalFailure):
    def __init__(self, message, fit):
        super().__init__(message)
        self.fit = fit


class RawSolver(E.Solver):
    def __init__(self, args, seed):
        super().__init__(args, seed, "ordinary")
        annotate_state(self.state)

    def evaluate(self, vector):
        signature = np.asarray(vector).tobytes()
        if signature in self.cache:
            return self.cache[signature]
        if self.solves >= self.args.max_solves:
            raise E.WorkLimit
        self.solves += 1
        geometry = torch.tensor(vector, requires_grad=True)
        values, gradients = torch_basis(geometry, self.state, self.points_t)
        boundary_values, _ = torch_basis(geometry, self.state, self.boundary_points_t)
        try:
            coefficients, fit = E.fit_coefficients(
                values.detach().numpy(), gradients.detach().numpy(), self.weights, self.force,
                boundary_values.detach().numpy(), self.boundary_weights, self.neumann)
        except (E.NumericalFailure, np.linalg.LinAlgError) as exc:
            fit = {"solve_count": self.solves, "converged": False, "failure_reason": str(exc),
                   "trial_geometry": np.asarray(vector).tolist()}
            try:
                condition = float(str(exc).rsplit(": ", 1)[-1])
                fit["gradient_basis_condition"] = condition if math.isfinite(condition) else None
            except ValueError:
                fit["gradient_basis_condition"] = None
            self.failed_fits.append(fit)
            raise RawCoefficientFailure(str(exc), fit) from exc
        self.newton_iterations += fit["newton_iterations"]
        if not fit["converged"]:
            self.failed_fits.append({**fit, "solve_count": self.solves})
            raise E.InnerSolveFailure("Raw coefficient solve missed the paper's residual criterion",
                                      {**fit, "solve_count": self.solves})
        c = torch.from_numpy(coefficients)
        value = values @ c
        gradient = torch.einsum("npd,p->nd", gradients, c)
        interior = torch.sum(self.weights_t*(0.5*torch.sum(gradient*gradient, dim=1)-self.force_t*value))
        boundary = torch.sum(self.boundary_weights_t*self.neumann_t*(boundary_values @ c))
        mean = torch.sum(self.weights_t*value)
        objective = interior-boundary+0.5*E.SIGMA*(mean-E.TARGET_MEAN)**2
        objective.backward()
        state = E.unpack_geometry(np.asarray(vector), self.state, self.coordinate)
        state["coefficients"] = coefficients.tolist()
        annotate_state(state)
        result = {"objective": float(objective.detach()), "gradient": geometry.grad.detach().numpy(),
                  "state": state, "solve_count": self.solves, **fit}
        if not math.isfinite(result["objective"]) or not np.all(np.isfinite(result["gradient"])):
            raise E.NumericalFailure("Nonfinite raw geometry objective or gradient")
        self.cache[signature] = result
        return result

    def record(self, phase, event):
        self.history.append({"solve_count": self.solves, "phase": phase, "event": event,
                             "state": copy.deepcopy(self.state), **energy(self.state, self.args.quadrature_order)})

    def run(self):
        started = time.perf_counter()
        initial, status, error = None, "work_limit", None
        try:
            vector, bounds = E.pack_geometry(self.state, self.coordinate)
            self.state = copy.deepcopy(self.evaluate(vector)["state"])
            initial = copy.deepcopy(self.state)
            self.record("initial", "raw_coefficient_solve")
            result = E.minimize(self.objective, vector, jac=True, method="L-BFGS-B", bounds=bounds,
                                callback=self.accepted,
                                options={"maxiter": self.args.max_solves,
                                         "maxfun": self.args.max_solves-self.solves,
                                         "ftol": self.args.ftol, "gtol": self.args.gtol,
                                         "maxls": 40, "maxcor": 12})
            self.state = copy.deepcopy(self.evaluate(result.x)["state"])
            self.blocks.append({"start_solve": 1, "end_solve": self.solves,
                                "status": int(result.status), "success": bool(result.success),
                                "message": str(result.message), "iterations": int(result.nit)})
            status = "optimizer_converged" if result.success else ("work_limit" if result.status == 1 else "optimizer_failure")
        except RawCoefficientFailure as exc:
            status = "coefficient_condition_failure" if "ill conditioned" in str(exc) else "coefficient_solve_failure"
            error = {"type": type(exc).__name__, "message": str(exc), "fit": exc.fit,
                     "traceback": traceback.format_exc()}
        except E.InnerSolveFailure as exc:
            status = "inner_solve_failure"
            error = {"type": type(exc).__name__, "message": str(exc), "fit": exc.fit,
                     "traceback": traceback.format_exc()}
        except E.WorkLimit:
            status = "work_limit"
        except Exception as exc:
            status = "numerical_failure"
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        self.record("final", "returned_last_accepted_state")
        fitted = list(self.cache.values())
        audits = [energy(self.state, order) for order in (64, 96)]
        direct_error = relative_h1(self.state)
        points, weights = E.quadrature(96)
        raw_value, raw_gradient = evaluate_state(self.state, points)
        stable_value, stable_gradient = E.evaluate_state(to_stable_state(self.state), points)
        evaluation_change = float(np.sqrt(weights @ ((raw_value-stable_value)**2
                                      +np.sum((raw_gradient-stable_gradient)**2, axis=1))))
        sv, sg, _ = E.basis_fields(to_stable_state(self.state), self.points)
        bv, _, _ = E.basis_fields(to_stable_state(self.state), self.boundary_points)
        stiffness = np.einsum("n,npd,nqd->pq", self.weights, sg, sg)
        means = sv.T @ self.weights
        load = sv.T @ (self.weights*self.force) + bv.T @ (self.boundary_weights*self.neumann)
        stable_matrix = stiffness+E.SIGMA*np.outer(means, means)
        stable_load = load+E.SIGMA*E.TARGET_MEAN*means
        scales = np.sqrt(np.diag(stiffness)+np.einsum("n,np,np->p", self.weights, sv, sv))
        residual = (stable_matrix @ np.asarray(self.state["derived_stable_coefficients"])-stable_load)/scales
        stable_stationarity = float(np.linalg.norm(residual)/max(1.0, np.linalg.norm(stable_load/scales)))
        result = {"seed": self.seed, "method": "raw_ordinary", "gap_coordinate": "log",
                  "training_quadrature_order": self.args.quadrature_order,
                  "initial_state": initial, "final_state": copy.deepcopy(self.state),
                  "history": self.history, "transfers": [], "restart_events": [],
                  "optimizer_blocks": self.blocks, "status": status, "error": error,
                  "coefficient_solves": self.solves, "final_diagnostics": energy(self.state, self.args.quadrature_order),
                  "relative_h1_error": direct_error, "error_evaluation_quadrature_order": 96,
                  "independent_quadrature": audits,
                  "maximum_relative_coefficient_stationarity": max((x["relative_coefficient_stationarity"] for x in fitted), default=None),
                  "maximum_gradient_basis_condition": max((x["gradient_basis_condition"] for x in fitted), default=None),
                  "inner_solve_summary": {"fits": len(fitted), "not_met": len(self.failed_fits)},
                  "failed_fits": self.failed_fits,
                  "stable_evaluation_audit": {"relative_h1_error": relative_h1(self.state, stable_evaluation=True),
                                               "absolute_h1_change_from_raw_evaluation": evaluation_change,
                                               "relative_stable_coordinate_stationarity": stable_stationarity,
                                               "note": "Same raw state, converted only for passive diagnostics; no stable refit or retry."},
                  "elapsed_seconds": time.perf_counter()-started}
        print(f"seed={self.seed} raw_ordinary: {status}; solves={self.solves}; H1={direct_error:.6e}", flush=True)
        return result


def validate(args):
    solver = RawSolver(args, 1729)
    stable = E.initial_state(1729)
    stable["coefficients"] = np.linspace(-1.0, 1.0, E.NSLOTS).tolist()
    raw = copy.deepcopy(stable)
    weights = []
    for i, pair in enumerate(raw["pairs"]):
        a, b = stable["coefficients"][2*i:2*i+2]
        weights.extend((a/2+b/(2*pair["half_gap"]), a/2-b/(2*pair["half_gap"])))
    weights.extend(stable["coefficients"][2*E.NPAIRS:])
    raw["coefficients"] = weights
    annotate_state(raw)
    rv, rg = evaluate_state(raw, solver.points)
    sv, sg = E.evaluate_state(stable, solver.points)
    value_difference = float(np.max(np.abs(rv-sv))/max(1.0, np.max(np.abs(sv))))
    gradient_difference = float(np.max(np.abs(rg-sg))/max(1.0, np.max(np.abs(sg))))
    vector, _ = E.pack_geometry(solver.state, solver.coordinate)
    result = solver.evaluate(vector)
    numerical = []
    for i in range(len(vector)):
        step = 2e-5*max(1.0, abs(vector[i]))
        plus, minus = vector.copy(), vector.copy()
        plus[i] += step
        minus[i] -= step
        numerical.append((solver.evaluate(plus)["objective"]-solver.evaluate(minus)["objective"])/(2*step))
    numerical = np.asarray(numerical)
    gradient_error = float(np.max(np.abs(numerical-result["gradient"]))
                           / max(1e-10, np.max(np.abs(numerical)), np.max(np.abs(result["gradient"]))))
    stable_solver = E.Solver(args, 1729, "ordinary")
    stable_result = stable_solver.evaluate(vector)
    raw_value, raw_gradient = evaluate_state(result["state"], solver.points)
    reference_value, reference_gradient = E.evaluate_state(stable_result["state"], solver.points)
    refit_change = float(np.sqrt(solver.weights @ ((raw_value-reference_value)**2
                                  +np.sum((raw_gradient-reference_gradient)**2, axis=1))))
    reference_norm = float(np.sqrt(solver.weights @ (reference_value**2+np.sum(reference_gradient**2, axis=1))))
    report = {"pair_column_order": "center + h*d, then center - h*d",
              "moderate_gap_value_relative_max_error": value_difference,
              "moderate_gap_spatial_gradient_relative_max_error": gradient_difference,
              "raw_reduced_autodiff_vs_central_difference_relative_max_error": gradient_error,
              "initial_raw_vs_stable_refit_relative_h1_change": refit_change/reference_norm,
              "initial_raw_vs_stable_objective_absolute_change": abs(result["objective"]-stable_result["objective"]),
              "passed": value_difference < 1e-12 and gradient_difference < 1e-12
                        and gradient_error < 2e-5 and refit_change/reference_norm < 1e-10}
    if not report["passed"]:
        raise RuntimeError(f"Raw Poisson validation failed: {report}")
    return report


def run_seed(task):
    args, seed = task
    return RawSolver(args, seed).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/raw_coordinates/poisson")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.quadrature_order, args.max_solves = 32, 8000
    args.ftol = args.gtol = 0.0
    args.initial_half_gap_min, args.initial_half_gap_max = 0.02, 0.05
    args.block_iterations, args.proximity_threshold, args.transfer_tolerance = 40, 0.12, 0.03
    if args.output.exists():
        parser.error("Output directory exists; choose a new path to preserve records")
    validation = validate(args)
    print(json.dumps(validation, indent=2), flush=True)
    if args.validate_only:
        return
    args.output.mkdir(parents=True)
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__), PAPER_SOURCE)}
    configuration = {"seeds": args.seeds, "quadrature_order": 32, "error_quadrature_order": 96,
                     "max_solves": 8000, "geometry": "paper centers/angles/log h; identical bounds",
                     "optimizer": "L-BFGS-B ftol0 gtol0 maxcor12 maxls40; no triggered restart",
                     "coefficient_solve": "ordinary translated Gaussian columns; original diagonal scaling, condition cutoff1e14, residual criterion; no ridge or rank truncation",
                     "training_evaluation": "direct raw Gaussian values and spatial derivatives",
                     "state_coefficients": "ordinary translate weights; derived_stable_coefficients is postprocessing only",
                     "pair_column_order": "center + h*d, then center - h*d"}
    payload = {"schema_version": 1, "configuration": configuration,
               "software": {"python": E.platform.python_version(), "numpy": np.__version__,
                            "scipy": E.scipy.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
               "source_files_sha256": source_hashes, "validation": validation, "runs": []}
    (args.output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    tasks = [(args, seed) for seed in args.seeds]
    if args.jobs > 1:
        pool = ProcessPoolExecutor(max_workers=args.jobs)
        outputs = pool.map(run_seed, tasks)
    else:
        pool, outputs = None, map(run_seed, tasks)
    for result in outputs:
        payload["runs"].append(result)
        (args.output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    if pool is not None:
        pool.shutdown()
    errors = [run["relative_h1_error"] for run in payload["runs"]]
    summary = {"starts": args.seeds, "status_counts": dict(Counter(run["status"] for run in payload["runs"])),
               "relative_h1_error_range": [min(errors), max(errors)], "relative_h1_error_median": float(np.median(errors)),
               "per_start": [{"seed": r["seed"], "status": r["status"], "solves": r["coefficient_solves"],
                              "relative_h1_error": r["relative_h1_error"],
                              "raw_weight_norm": r["final_diagnostics"]["raw_weight_norm"],
                              "derived_stable_coefficient_norm": r["final_diagnostics"]["stable_coefficient_norm"],
                              "min_half_gap": min(pair["half_gap"] for pair in r["final_state"]["pairs"])}
                             for r in payload["runs"]]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"provenance.json").write_text(json.dumps({"source_files_sha256": source_hashes,
             "results_sha256": hashlib.sha256((args.output/"results.json").read_bytes()).hexdigest(),
             "command": " ".join(sys.argv), "software": payload["software"]}, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
