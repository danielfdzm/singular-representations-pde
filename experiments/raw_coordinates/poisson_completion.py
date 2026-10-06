#!/usr/bin/env python3
"""Raw finite-pair Poisson optimization with explicit derivative-block transfers.

Finite pairs always train ordinary weights on directly translated Gaussians.
Only transferred pairs use a kernel and its minus-directional derivative.
The paper's geometry, quadrature, inner solve, transfer tolerances, and
L-BFGS-B/work protocol are retained. Stable-coordinate conversions are audits.
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
RAW_SOURCE = Path(__file__).with_name("poisson_raw.py")
spec = importlib.util.spec_from_file_location("poisson_raw_for_completion", RAW_SOURCE)
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
E, np, torch = R.E, R.np, R.torch
quadrature = E.quadrature
manufactured_jet_solution = E.manufactured_jet_solution
manufactured_jet_gradient = E.manufactured_jet_gradient


def annotate_state(state):
    diagnostic, kinds = [], []
    for i, pair in enumerate(state["pairs"]):
        first, second = state["coefficients"][2*i:2*i+2]
        if pair["completed"]:
            diagnostic.extend((first, second))
            kinds.append("kernel_and_minus_directional_derivative")
        else:
            diagnostic.extend((first+second, pair["half_gap"]*(first-second)))
            kinds.append("ordinary_translate_weights_at_plus_then_minus_center")
    diagnostic.extend(state["coefficients"][2*len(state["pairs"]):])
    state["stored_coefficients"] = list(state["coefficients"])
    state["derived_stable_coefficients"] = diagnostic
    state["pair_coefficient_types"] = kinds
    state["coefficient_coordinates"] = "raw_finite_pairs_and_direct_completed_blocks"
    return state


def to_stable_state(state):
    result = annotate_state(copy.deepcopy(state))
    result["coefficients"] = result["derived_stable_coefficients"]
    result["coefficient_coordinates"] = "derived_pair_coefficients_for_audit_only"
    return result


def basis_fields(state, points):
    points = np.asarray(points, dtype=float)
    q = 1.0/E.SCALE**2
    values, gradients, laps = [], [], []
    for pair in state["pairs"]:
        c = np.asarray(pair["center"])
        d = np.array((math.cos(pair["angle"]), math.sin(pair["angle"])))
        if pair["completed"]:
            r = points-c
            r2 = np.sum(r*r, axis=1)
            t = r @ d
            k = np.exp(-0.5*q*r2)
            b = q*t*k
            values.extend((k, b))
            gradients.extend((-q*r*k[:, None], q*k[:, None]*(d[None, :]-q*t[:, None]*r)))
            laps.extend(((q*q*r2-2*q)*k, (q*q*r2-4*q)*b))
        else:
            for centre in (c+pair["half_gap"]*d, c-pair["half_gap"]*d):
                r = points-centre
                r2 = np.sum(r*r, axis=1)
                k = np.exp(-0.5*q*r2)
                values.append(k)
                gradients.append(-q*r*k[:, None])
                laps.append((q*q*r2-2*q)*k)
    for centre in state["singletons"]:
        r = points-np.asarray(centre)
        r2 = np.sum(r*r, axis=1)
        k = np.exp(-0.5*q*r2)
        values.append(k)
        gradients.append(-q*r*k[:, None])
        laps.append((q*q*r2-2*q)*k)
    return np.stack(values, axis=1), np.stack(gradients, axis=1), np.stack(laps, axis=1)


def evaluate_state(state, points):
    values, gradients, _ = basis_fields(state, points)
    c = np.asarray(state["coefficients"])
    return values @ c, np.einsum("npd,p->nd", gradients, c)


def torch_basis(vector, template, points):
    values, gradients, cursor = [], [], 0
    q = 1.0/E.SCALE**2
    for pair in template["pairs"]:
        c, angle = vector[cursor:cursor+2], vector[cursor+2]
        cursor += 3
        d = torch.stack((torch.cos(angle), torch.sin(angle)))
        if pair["completed"]:
            r = points-c
            t = r @ d
            k = torch.exp(-0.5*q*torch.sum(r*r, dim=1))
            values.extend((k, q*t*k))
            gradients.extend((-q*r*k[:, None], q*k[:, None]*(d[None, :]-q*t[:, None]*r)))
        else:
            h = torch.exp(vector[cursor])
            cursor += 1
            for centre in (c+h*d, c-h*d):
                r = points-centre
                k = torch.exp(-0.5*q*torch.sum(r*r, dim=1))
                values.append(k)
                gradients.append(-q*r*k[:, None])
    for _ in template["singletons"]:
        r = points-vector[cursor:cursor+2]
        cursor += 2
        k = torch.exp(-0.5*q*torch.sum(r*r, dim=1))
        values.append(k)
        gradients.append(-q*r*k[:, None])
    return torch.stack(values, dim=1), torch.stack(gradients, dim=1)


def relative_h1(state, order=96, stable_evaluation=False):
    points, weights = E.quadrature(order)
    exact = E.manufactured_jet_solution(*points.T)
    exact_gradient = E.manufactured_jet_gradient(*points.T)
    u, g = (E.evaluate_state(to_stable_state(state), points) if stable_evaluation else evaluate_state(state, points))
    return float(np.sqrt(weights @ ((u-exact)**2+np.sum((g-exact_gradient)**2, axis=1))
                         / (weights @ (exact**2+np.sum(exact_gradient**2, axis=1)))))


def energy(state, order):
    points, weights = E.quadrature(order)
    u, g = evaluate_state(state, points)
    bp, bw, normals = E.boundary_quadrature(order)
    bu, _ = evaluate_state(state, bp)
    gradient_term = float(0.5*weights @ np.sum(g*g, axis=1))
    load = float(weights @ (E.manufactured_jet_load(*points.T)*u))
    boundary = float(bw @ (E.manufactured_neumann_data(bp, normals)*bu))
    mean = float(weights @ u)
    mean_term = float(0.5*E.SIGMA*(mean-E.TARGET_MEAN)**2)
    stored_norm = float(np.linalg.norm(state["coefficients"]))
    return {"quadrature_order": order, "energy": gradient_term-load-boundary+mean_term,
            "gradient_energy": gradient_term, "load": load, "neumann_work": boundary,
            "mean_term": mean_term, "state_mean": mean,
            "h1_norm": float(np.sqrt(weights @ (u*u+np.sum(g*g, axis=1)))),
            "stable_coefficient_norm": float(np.linalg.norm(state["derived_stable_coefficients"])),
            "stored_coefficient_norm": stored_norm, "representation_coefficient_norm": stored_norm,
            "raw_weight_norm": stored_norm, "kernel_weight_norm": stored_norm,
            "completed_pairs": sum(pair["completed"] for pair in state["pairs"]),
            "multiplicity_budget": E.NSLOTS}


def transfer_predictor(state, pair_index):
    result = copy.deepcopy(state)
    pair = result["pairs"][pair_index]
    if pair["completed"]:
        raise ValueError("Pair is already completed")
    first, second = result["coefficients"][2*pair_index:2*pair_index+2]
    result["coefficients"][2*pair_index:2*pair_index+2] = [first+second, pair["half_gap"]*(first-second)]
    pair.update({"completed": True, "half_gap": 0.0})
    return annotate_state(result)


def transfer_bound(state, pair_index, order=64):
    points, weights = E.quadrature(order)
    u, g = evaluate_state(state, points)
    v, h = evaluate_state(transfer_predictor(state, pair_index), points)
    measured = float(np.sqrt(weights @ ((u-v)**2+np.sum((g-h)**2, axis=1))))
    return {"absolute_h1_bound_float64": measured, "measured_h1_state_change": measured,
            "quadrature_order": order, "certification": "Measured float64 H1 change, not an enclosure",
            "conversion": "a=w_plus+w_minus; b=h*(w_plus-w_minus); same center and angle"}


class MixedSolver(R.RawSolver):
    def __init__(self, args, seed):
        E.Solver.__init__(self, args, seed, "completion")
        annotate_state(self.state)

    def evaluate(self, vector):
        signature = (tuple(pair["completed"] for pair in self.state["pairs"]), np.asarray(vector).tobytes())
        if signature in self.cache:
            return self.cache[signature]
        if self.solves >= self.args.max_solves:
            raise E.WorkLimit
        self.solves += 1
        geometry = torch.tensor(vector, requires_grad=True)
        values, gradients = torch_basis(geometry, self.state, self.points_t)
        bvalues, _ = torch_basis(geometry, self.state, self.boundary_points_t)
        try:
            coefficients, fit = E.fit_coefficients(values.detach().numpy(), gradients.detach().numpy(),
                self.weights, self.force, bvalues.detach().numpy(), self.boundary_weights, self.neumann)
        except (E.NumericalFailure, np.linalg.LinAlgError) as exc:
            fit = {"solve_count": self.solves, "converged": False, "failure_reason": str(exc),
                   "trial_geometry": np.asarray(vector).tolist()}
            self.failed_fits.append(fit)
            raise R.RawCoefficientFailure(str(exc), fit) from exc
        self.newton_iterations += fit["newton_iterations"]
        if not fit["converged"]:
            self.failed_fits.append({**fit, "solve_count": self.solves})
            raise E.InnerSolveFailure("Mixed raw coefficient solve missed the paper residual criterion", {**fit, "solve_count": self.solves})
        c = torch.from_numpy(coefficients)
        u = values @ c
        g = torch.einsum("npd,p->nd", gradients, c)
        interior = torch.sum(self.weights_t*(0.5*torch.sum(g*g, dim=1)-self.force_t*u))
        boundary = torch.sum(self.boundary_weights_t*self.neumann_t*(bvalues @ c))
        mean = torch.sum(self.weights_t*u)
        objective = interior-boundary+0.5*E.SIGMA*(mean-E.TARGET_MEAN)**2
        objective.backward()
        state = E.unpack_geometry(np.asarray(vector), self.state, self.coordinate)
        state["coefficients"] = coefficients.tolist()
        annotate_state(state)
        result = {"objective": float(objective.detach()), "gradient": geometry.grad.detach().numpy(),
                  "state": state, "solve_count": self.solves, **fit}
        if not math.isfinite(result["objective"]) or not np.all(np.isfinite(result["gradient"])):
            raise E.NumericalFailure("Nonfinite mixed raw objective or gradient")
        self.cache[signature] = result
        return result

    def record(self, phase, event):
        self.history.append({"solve_count": self.solves, "phase": phase, "event": event,
                             "state": copy.deepcopy(self.state), **energy(self.state, self.args.quadrature_order)})

    def eligible_transfers(self):
        return [i for i, pair in enumerate(self.state["pairs"])
                if not pair["completed"] and 2*pair["half_gap"] <= self.args.proximity_threshold
                and transfer_bound(self.state, i)["measured_h1_state_change"] <= self.args.transfer_tolerance]

    def attempt_transfers(self):
        changed = False
        for i in self.eligible_transfers():
            if self.solves >= self.args.max_solves:
                break
            before = copy.deepcopy(self.state)
            bound = transfer_bound(before, i)
            if bound["measured_h1_state_change"] > self.args.transfer_tolerance:
                continue
            checkpoint = self.solves
            predictor = transfer_predictor(before, i)
            self.state = predictor
            vector, _ = E.pack_geometry(self.state, self.coordinate)
            try:
                refit = self.evaluate(vector)
            except E.WorkLimit:
                self.state = before
                break
            except Exception:
                self.state = before
                raise
            self.state = copy.deepcopy(refit["state"])
            self.transfers.append({"solve_count": checkpoint, "refit_solve_count": self.solves,
                "pair_index": i, "before_state": before, "predictor_state": predictor,
                "after_state": copy.deepcopy(self.state), "bound": bound})
            self.record("transfer", f"completed_pair_{i}")
            changed = True
        return changed

    def run(self):
        started = time.perf_counter()
        initial, status, error = None, "work_limit", None
        try:
            vector, _ = E.pack_geometry(self.state, self.coordinate)
            self.state = copy.deepcopy(self.evaluate(vector)["state"])
            initial = copy.deepcopy(self.state)
            self.record("initial", "raw_coefficient_solve")
            while self.solves < self.args.max_solves:
                vector, bounds = E.pack_geometry(self.state, self.coordinate)
                before, result = self.solves, None
                try:
                    result = E.minimize(self.objective, vector, jac=True, method="L-BFGS-B", bounds=bounds,
                        callback=self.accepted, options={"maxiter": self.args.max_solves,
                        "maxfun": self.args.max_solves-self.solves, "ftol": self.args.ftol,
                        "gtol": self.args.gtol, "maxls": 40, "maxcor": 12})
                    self.state = copy.deepcopy(self.evaluate(result.x)["state"])
                    self.blocks.append({"start_solve": before, "end_solve": self.solves,
                        "status": int(result.status), "success": bool(result.success),
                        "message": str(result.message), "iterations": int(result.nit)})
                except E.TransferReady:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "transfer_ready"})
                except E.WorkLimit:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "work_limit"})
                    break
                transferred = self.attempt_transfers()
                if not transferred and result is not None and result.success:
                    status = "optimizer_converged"
                    break
                if not transferred and result is not None and result.status != 1:
                    status = "optimizer_failure"
                    break
                if before == self.solves:
                    status = "no_new_evaluations"
                    break
        except (R.RawCoefficientFailure, E.InnerSolveFailure) as exc:
            status = ("coefficient_condition_failure" if "ill conditioned" in str(exc)
                      else "inner_solve_failure")
            error = {"type": type(exc).__name__, "message": str(exc), "fit": exc.fit, "traceback": traceback.format_exc()}
        except Exception as exc:
            status = "numerical_failure"
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        self.record("final", "returned_last_accepted_state")
        fitted = list(self.cache.values())
        points, weights = E.quadrature(96)
        u, g = evaluate_state(self.state, points)
        v, h = E.evaluate_state(to_stable_state(self.state), points)
        audits = [energy(self.state, order) for order in (64, 96)]
        result = {"seed": self.seed, "method": "completion", "gap_coordinate": "log",
            "training_quadrature_order": self.args.quadrature_order,
            "initial_state": initial, "final_state": copy.deepcopy(self.state),
            "history": self.history, "transfers": self.transfers, "restart_events": [],
            "optimizer_blocks": self.blocks, "status": status, "error": error,
            "coefficient_solves": self.solves, "final_diagnostics": energy(self.state, self.args.quadrature_order),
            "relative_h1_error": relative_h1(self.state), "error_evaluation_quadrature_order": 96,
            "independent_quadrature": audits,
            "maximum_relative_coefficient_stationarity": max((x["relative_coefficient_stationarity"] for x in fitted), default=None),
            "maximum_gradient_basis_condition": max((x["gradient_basis_condition"] for x in fitted), default=None),
            "inner_solve_summary": {"fits": len(fitted), "not_met": len(self.failed_fits)},
            "failed_fits": self.failed_fits,
            "stable_evaluation_audit": {"relative_h1_error": relative_h1(self.state, stable_evaluation=True),
                "absolute_h1_change_from_direct_evaluation": float(np.sqrt(weights @ ((u-v)**2+np.sum((g-h)**2, axis=1)))),
                "note": "Same mixed raw state; postprocessing conversion only, no stable refit."},
            "elapsed_seconds": time.perf_counter()-started}
        print(f"seed={self.seed} n={self.args.quadrature_order} raw_completion: {status}; fits={self.solves}; transfers={len(self.transfers)}; H1={result['relative_h1_error']:.6e}", flush=True)
        return result


def validate(args):
    reports = []
    for mask in ((False,)*4, (True, False, False, False), (True, False, True, False), (True,)*4):
        solver = MixedSolver(args, 1729)
        stable = E.initial_state(1729, (args.initial_half_gap_min, args.initial_half_gap_max))
        stable["coefficients"] = np.linspace(-1, 1, E.NSLOTS).tolist()
        mixed = copy.deepcopy(stable)
        c = []
        for i, (pair, completed) in enumerate(zip(mixed["pairs"], mask)):
            a, b = stable["coefficients"][2*i:2*i+2]
            if completed:
                pair.update({"completed": True, "half_gap": 0.0})
                stable["pairs"][i].update({"completed": True, "half_gap": 0.0})
                c.extend((a, b))
            else:
                c.extend((a/2+b/(2*pair["half_gap"]), a/2-b/(2*pair["half_gap"])))
        c.extend(stable["coefficients"][2*E.NPAIRS:])
        mixed["coefficients"] = c
        annotate_state(mixed)
        mv, mg = evaluate_state(mixed, solver.points)
        sv, sg = E.evaluate_state(stable, solver.points)
        difference = max(float(np.max(np.abs(mv-sv))/max(1, np.max(np.abs(sv)))),
                         float(np.max(np.abs(mg-sg))/max(1, np.max(np.abs(sg)))))
        lap_difference = float(np.max(np.abs(basis_fields(mixed, solver.points)[2] @ np.asarray(c)
                                    -E.basis_fields(stable, solver.points)[2] @ np.asarray(stable["coefficients"]))))
        solver.state = mixed
        vector, _ = E.pack_geometry(mixed, solver.coordinate)
        out = solver.evaluate(vector)
        finite_difference = []
        for i in range(len(vector)):
            step = 2e-5*max(1, abs(vector[i]))
            plus, minus = vector.copy(), vector.copy()
            plus[i] += step
            minus[i] -= step
            finite_difference.append((solver.evaluate(plus)["objective"]-solver.evaluate(minus)["objective"])/(2*step))
        finite_difference = np.asarray(finite_difference)
        error = float(np.max(np.abs(finite_difference-out["gradient"]))
                      / max(1e-10, np.max(np.abs(finite_difference)), np.max(np.abs(out["gradient"]))))
        reference = E.Solver(args, 1729, "completion")
        reference.state = stable
        refit = reference.evaluate(vector)
        u, g = evaluate_state(out["state"], solver.points)
        v, h = E.evaluate_state(refit["state"], solver.points)
        refit_change = float(np.sqrt(solver.weights @ ((u-v)**2+np.sum((g-h)**2, axis=1))
                                    / (solver.weights @ (v*v+np.sum(h*h, axis=1)))))
        transfer_error = 0.0
        finite_indices = [i for i, flag in enumerate(mask) if not flag]
        if finite_indices:
            i = finite_indices[0]
            measured = transfer_bound(out["state"], i)["measured_h1_state_change"]
            control = E.transfer_bound(to_stable_state(out["state"]), i)["measured_h1_state_change"]
            transfer_error = abs(measured-control)
        passed = difference < 1e-12 and lap_difference < 1e-11 and error < 2e-5 and refit_change < 1e-10 and transfer_error < 1e-11
        reports.append({"completed_mask": list(mask), "direct_field_relative_max_error": difference,
            "laplacian_absolute_max_error": lap_difference, "reduced_gradient_relative_max_error": error,
            "raw_mixed_vs_stable_refit_relative_h1_change": refit_change,
            "transfer_h1_measurement_absolute_difference": transfer_error, "passed": passed})
    if not all(report["passed"] for report in reports):
        raise RuntimeError(f"Mixed raw Poisson validation failed: {reports}")
    return reports


def run_seed(task):
    args, seed = task
    result = (MixedSolver(args, seed) if args.method == "completion" else R.RawSolver(args, seed)).run()
    result["method"] = "completion" if args.method == "completion" else "ordinary"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("standard", "completion"), default="completion")
    parser.add_argument("--quadrature-order", type=int, choices=(32, 64), default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.max_solves, args.ftol, args.gtol = 8000, 0.0, 0.0
    args.initial_half_gap_min, args.initial_half_gap_max = 0.02, 0.05
    args.block_iterations, args.proximity_threshold, args.transfer_tolerance = 40, 0.12, 0.03
    if args.output.exists():
        parser.error("Output directory exists; preserve records and choose a new path")
    validation = validate(args) if args.method == "completion" else R.validate(args)
    print(json.dumps(validation, indent=2), flush=True)
    if args.validate_only:
        return
    args.output.mkdir(parents=True)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (Path(__file__), RAW_SOURCE, R.PAPER_SOURCE)}
    method_name = "completion" if args.method == "completion" else "ordinary"
    configuration = {"method": method_name, "seeds": args.seeds, "quadrature_order": args.quadrature_order,
        "error_quadrature_order": 96, "max_solves": 8000, "block_iterations": 40,
        "proximity_threshold": 0.12, "transfer_tolerance": 0.03, "transfer_quadrature_order": 64,
        "geometry": "Original paper starts/bounds; finite gaps use log h",
        "optimizer": "L-BFGS-B ftol0 gtol0 maxcor12 maxls40; restart after transfers only",
        "coefficient_solve": "Original diagonal scaling, condition cutoff1e14, residual test; no ridge/rank truncation",
        "training_basis": "Direct translated Gaussians for finite pairs; direct K and minus-directional K for completed pairs",
        "coefficient_semantics": "Finite slots store w_plus,w_minus; completed slots store a,b. Derived stable coefficients are audits only."}
    payload = {"schema_version": 1, "configuration": configuration,
        "software": {"python": E.platform.python_version(), "numpy": np.__version__, "scipy": E.scipy.__version__,
                     "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
        "source_files_sha256": hashes, "validation": validation, "runs": []}
    (args.output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    tasks = [(args, seed) for seed in args.seeds]
    if args.jobs > 1:
        pool = ProcessPoolExecutor(max_workers=args.jobs)
        outputs = pool.map(run_seed, tasks)
    else:
        pool, outputs = None, map(run_seed, tasks)
    for result in outputs:
        reference = E.initial_state(result["seed"], (args.initial_half_gap_min, args.initial_half_gap_max))
        initial = result["initial_state"]
        assert initial["singletons"] == reference["singletons"], "Initial singleton geometry differs from paper start"
        for actual, expected in zip(initial["pairs"], reference["pairs"]):
            assert actual["center"] == expected["center"] and actual["angle"] == expected["angle"]
            assert actual["completed"] == expected["completed"]
            # Geometry packing uses log h and unpacking uses exp(log h).
            assert math.isclose(actual["half_gap"], expected["half_gap"], rel_tol=2e-15, abs_tol=0.0), "Initial gap differs from paper start"
        payload["runs"].append(result)
        (args.output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    if pool is not None:
        pool.shutdown()
    errors = [run["relative_h1_error"] for run in payload["runs"]]
    summary = {"starts": args.seeds, "method": method_name, "training_quadrature_order": args.quadrature_order,
        "status_counts": dict(Counter(r["status"] for r in payload["runs"])),
        "relative_h1_error_range": [min(errors), max(errors)], "relative_h1_error_median": float(np.median(errors)),
        "transfer_count_range": [min(len(r["transfers"]) for r in payload["runs"]), max(len(r["transfers"]) for r in payload["runs"])],
        "per_start": [{"seed": r["seed"], "status": r["status"], "fits": r["coefficient_solves"],
            "relative_h1_error": r["relative_h1_error"], "transfers": len(r["transfers"]),
            "representation_coefficient_norm": float(np.linalg.norm(r["final_state"]["coefficients"])),
            "derived_stable_coefficient_norm": float(np.linalg.norm(r["final_state"]["derived_stable_coefficients"]))}
            for r in payload["runs"]]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"provenance.json").write_text(json.dumps({"source_files_sha256": hashes,
        "results_sha256": hashlib.sha256((args.output/"results.json").read_bytes()).hexdigest(),
        "command": " ".join(sys.argv), "software": payload["software"]}, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
