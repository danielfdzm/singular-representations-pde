"""Matérn variable projection in ordinary translated-kernel weights.

The original experiment is loaded in its own module namespace. Its quadrature,
data, Newton coefficient fit, L-BFGS-B, work cap, and stopping rules are reused.
Training columns are single translated kernels; no pair divided differences or
rescaled pair coefficients are used. Stable coordinates appear only in the
separate validation and postprocessing of returned states.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "matern_swift_hohenberg_2d" / "experiment.py"
spec = importlib.util.spec_from_file_location("matern_original_for_raw_baseline", SOURCE)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
np, torch = E.np, E.torch
_stable_basis_fields = E.basis_fields
RAW_ORDER = ["pair0_center_plus_hd", "pair0_center_minus_hd",
             "pair1_center_plus_hd", "pair1_center_minus_hd", "singleton"]


def raw_basis_fields(state, points):
    """Direct radial values, gradients, and Laplacians of five translates."""
    centers = []
    for pair in state["pairs"]:
        if pair["completed"]:
            raise ValueError("The raw baseline has only positive-gap translates")
        direction = np.array([np.cos(pair["angle"]), np.sin(pair["angle"])])
        center, offset = np.asarray(pair["center"]), pair["half_gap"]*direction
        centers.extend((center+offset, center-offset))
    centers.append(np.asarray(state["background_center"]))
    values, gradients, laps = [], [], []
    for center in centers:
        r = np.asarray(points)-center
        radius = np.sqrt(np.sum(r*r, axis=1))
        values.append(E.matern.radial("f0", radius))
        gradients.append(r*E.matern.radial("c1", radius)[:, None])
        laps.append(E.matern.radial("lap", radius))
    return np.stack(values, axis=1), np.stack(gradients, axis=1), np.stack(laps, axis=1)


def raw_torch_basis(vector, template, points):
    centers, cursor = [], 0
    for pair in template["pairs"]:
        if pair["completed"]:
            raise ValueError("The raw baseline cannot complete a pair")
        center, angle = vector[cursor:cursor+2], vector[cursor+2]
        direction = torch.stack((torch.cos(angle), torch.sin(angle)))
        h = torch.exp(vector[cursor+3]); cursor += 4
        centers.extend((center+h*direction, center-h*direction))
    centers.append(vector[cursor:cursor+2])
    values, gradients, laps = [], [], []
    for center in centers:
        r = points-center
        radius = torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-30))
        values.append(E.matern.torch_radial(torch, "f0", radius))
        gradients.append(r*E.matern.torch_radial(torch, "c1", radius)[:, None])
        laps.append(E.matern.torch_radial(torch, "lap", radius))
    return torch.stack(values, dim=1), torch.stack(gradients, dim=1), torch.stack(laps, dim=1)


def to_stable_state(state):
    """Postprocessing only: represent these same translates by pair a,b."""
    result = copy.deepcopy(state)
    coefficients = list(state["coefficients"])
    for i, pair in enumerate(state["pairs"]):
        plus, minus = state["coefficients"][2*i:2*i+2]
        coefficients[2*i:2*i+2] = [plus+minus, pair["half_gap"]*(plus-minus)]
    result["coefficients"] = coefficients
    return result


def evaluate_state(state, points):
    values, gradients, _ = raw_basis_fields(state, points)
    c = np.asarray(state["coefficients"])
    return values@c, np.einsum("npd,p->nd", gradients, c)


def stable_evaluate_state(state, points):
    stable = to_stable_state(state)
    values, gradients, _ = _stable_basis_fields(stable, points)
    c = np.asarray(stable["coefficients"])
    return values@c, np.einsum("npd,p->nd", gradients, c)


def raw_diagnostics(state, case, amplitude, order):
    result = E.energy(state, case, amplitude, order)
    result.update(raw_weight_norm=float(np.linalg.norm(state["coefficients"])),
                  remaining_raw_pair_coefficient_norm=float(np.linalg.norm(state["coefficients"][:4])),
                  completed_pairs=0, multiplicity_budget=5)
    return result


# These bindings affect only the private imported module, never source files.
# Its run/record/energy routines now evaluate the same raw state as training.
E.basis_fields = raw_basis_fields
E.diagnostics = raw_diagnostics


class RawSolver(E.Solver):
    def columns(self, geometry):
        values, _, laps = raw_torch_basis(geometry, self.state, self.points_t)
        bvalues, bgradients, _ = raw_torch_basis(geometry, self.state, self.boundary_points_t)
        normal = torch.einsum("npd,nd->np", bgradients, self.boundary_normals_t)
        return values, laps, bvalues, normal


def paper_args(max_solves=8000):
    return SimpleNamespace(max_solves=max_solves, quadrature_order=64, block_iterations=40,
                           proximity_threshold=0.05, transfer_tolerance=0.03, ftol=0.0, gtol=0.0,
                           initial_half_gap_min=0.02, initial_half_gap_max=0.05)


def _h1(u, g, weights):
    return float(np.sqrt(np.dot(weights, u*u+np.sum(g*g, axis=1))))


def validate():
    """Validate physical equivalence and variable-projection geometry gradients."""
    reports, args = [], paper_args()
    points, weights = E.quadrature(64)
    for seed in range(10):
        raw, stable = RawSolver(args, "manufactured_jet", 1.0, seed, "ordinary"), E.Solver(
            args, "manufactured_jet", 1.0, seed, "ordinary")
        vector, _ = E.pack_geometry(raw.state, "log")
        rr, sr = raw.evaluate(vector), stable.evaluate(vector)
        converted = copy.deepcopy(sr["state"])
        for i, pair in enumerate(converted["pairs"]):
            a, b = sr["state"]["coefficients"][2*i:2*i+2]
            converted["coefficients"][2*i:2*i+2] = [a/2+b/(2*pair["half_gap"]), a/2-b/(2*pair["half_gap"])]
        rv, rg, rl = raw_basis_fields(converted, points)
        sv, sg, sl = _stable_basis_fields(sr["state"], points)
        rc, sc = np.asarray(converted["coefficients"]), np.asarray(sr["state"]["coefficients"])
        realization = max(float(np.max(np.abs(rv@rc-sv@sc))),
                          float(np.max(np.abs(np.einsum("npd,p->nd", rg, rc)-np.einsum("npd,p->nd", sg, sc)))),
                          float(np.max(np.abs(rl@rc-sl@sc))))
        ru, rgrad = evaluate_state(rr["state"], points)
        su, sgrad = sv@sc, np.einsum("npd,p->nd", sg, sc)
        refit = _h1(ru-su, rgrad-sgrad, weights)/_h1(su, sgrad, weights)
        gradient = float(np.max(np.abs(rr["gradient"]-sr["gradient"])))
        reports.append({"seed": seed, "realization_max_absolute_difference": realization,
                        "initial_refit_relative_h1_difference": refit,
                        "initial_reduced_gradient_max_absolute_difference": gradient})
    solver = RawSolver(args, "manufactured_jet", 1.0, 0, "ordinary")
    vector, _ = E.pack_geometry(solver.state, "log")
    result = solver.evaluate(vector)
    finite = []
    for i in range(len(vector)):
        step = 2e-5*max(1.0, abs(vector[i]))
        plus, minus = vector.copy(), vector.copy()
        plus[i] += step; minus[i] -= step
        finite.append((solver.evaluate(plus)["objective"]-solver.evaluate(minus)["objective"])/(2*step))
    finite = np.asarray(finite)
    fd_error = float(np.max(np.abs(finite-result["gradient"])))
    fd_relative = fd_error/max(1e-10, float(np.max(np.abs(finite))), float(np.max(np.abs(result["gradient"]))))
    tolerances = {"realization_absolute": 1e-10, "initial_refit_relative_h1": 1e-9,
                  "reduced_gradient_absolute": 1e-8, "finite_difference_relative": 2e-5}
    passed = all(r["realization_max_absolute_difference"] < tolerances["realization_absolute"]
                 and r["initial_refit_relative_h1_difference"] < tolerances["initial_refit_relative_h1"]
                 and r["initial_reduced_gradient_max_absolute_difference"] < tolerances["reduced_gradient_absolute"]
                 for r in reports) and fd_relative < tolerances["finite_difference_relative"]
    report = {"passed": passed, "tolerances": tolerances, "starts": reports,
              "finite_difference": {"seed": 0, "relative_maximum_error": fd_relative,
                                    "maximum_absolute_error": fd_error,
                                    "analytic": result["gradient"].tolist(), "finite_difference": finite.tolist()}}
    if not passed:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def postprocess(run):
    """Independent order96 direct-raw errors and stable reevaluation diagnostics."""
    points, weights = E.quadrature(96)
    target = E.manufactured_jet_solution(points[:, 0], points[:, 1])
    target_gradient = E.manufactured_jet_gradient(points[:, 0], points[:, 1])
    reference = _h1(target, target_gradient, weights)
    cached = {}

    def annotate(state):
        key = json.dumps(state, sort_keys=True)
        if key not in cached:
            u, g = evaluate_state(state, points)
            su, sg = stable_evaluate_state(state, points)
            stable = to_stable_state(state)["coefficients"]
            cached[key] = {"relative_h1_error": _h1(u-target, g-target_gradient, weights)/reference,
                           "absolute_h1_error": _h1(u-target, g-target_gradient, weights),
                           "stable_reevaluation_relative_h1_discrepancy": _h1(u-su, g-sg, weights)/reference,
                           "stable_reevaluation_relative_h1_error": _h1(su-target, sg-target_gradient, weights)/reference,
                           "stable_coefficient_norm": float(np.linalg.norm(stable)),
                           "raw_weight_norm": float(np.linalg.norm(state["coefficients"])),
                           "diagnostic_quadrature_order": 96}
        state["raw_weights"] = list(state["coefficients"])
        state["diagnostic_stable_coefficients"] = to_stable_state(state)["coefficients"]
        return cached[key]

    run["initial_diagnostics_order96"] = annotate(run["initial_state"])
    for row in run["history"]:
        row.update(annotate(row["state"]))
    run["final_diagnostics"].update(annotate(run["final_state"]))
    run["final_stationarity_diagnostics"] = {
        "quadrature_order": 64, "refitted": False,
        "raw_scaled_coordinates": coefficient_residual(run["final_state"], raw_basis_fields),
        "stable_scaled_coordinates": coefficient_residual(to_stable_state(run["final_state"]), _stable_basis_fields)}
    run["coefficient_coordinates"] = "ordinary_translated_kernel_weights"
    run["raw_weights_order"] = RAW_ORDER
    run["final_state_semantics"] = "last accepted fitted state; failed line-search probes never replace it"
    return run


def coefficient_residual(state, basis):
    """Passive residual of the supplied coefficients; never refit or accept."""
    points, weights = E.quadrature(64)
    bp, bw, bn = E.boundary_quadrature(64)
    values, _, laps = basis(state, points)
    bvalues, bgradient, _ = basis(state, bp)
    bnormal = np.einsum("npd,nd->np", bgradient, bn)
    curvature = values+laps
    scales = np.sqrt(np.einsum("n,np,np->p", weights, curvature, curvature)
                     + np.einsum("n,np,np->p", weights, values, values))
    scales = np.where(scales > 0, scales, 1.0)
    columns, curvature = values/scales, curvature/scales
    quadratic = (np.einsum("n,np,nq->pq", weights, curvature, curvature)
                 + E.RHO*np.einsum("n,np,nq->pq", weights, columns, columns))
    m, q = E.manufactured_boundary_data(bp, bn)
    load = (columns.T@(weights*E.forcing_values("manufactured_jet", 1.0, points))
            + (bnormal/scales).T@(bw*m) + (bvalues/scales).T@(bw*q))
    alpha = np.asarray(state["coefficients"])*scales
    u = columns@alpha
    residual = quadratic@alpha+columns.T@(weights*E.nonlinear_term(u))-load
    return {"relative_residual": float(np.linalg.norm(residual)/max(1.0, np.linalg.norm(load))),
            "absolute_residual": float(np.linalg.norm(residual)), "residual": residual.tolist(),
            "quadratic_condition": float(np.linalg.cond(quadratic))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE.parents[2]/"outputs"/"raw_coordinates"/"matern")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max-solves", type=int, default=8000)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous results")
    args.output.mkdir(parents=True)
    hashes = {str(p.relative_to(SOURCE.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), SOURCE, SOURCE.with_name("matern.py"))}
    payload = {"schema_version": 1, "configuration": {
        "case": "manufactured_jet", "amplitude": 1.0, "seeds": args.seeds,
        "max_solves": args.max_solves, "training_quadrature_order": 64, "diagnostic_quadrature_order": 96,
        "initial_half_gap_range": [0.02, 0.05], "coefficient_coordinates": "ordinary_translated_kernel_weights",
        "raw_weights_order": RAW_ORDER, "gap_coordinate": "log", "optimizer": "original L-BFGS-B ftol0 gtol0 maxcor12 maxls40",
        "coefficient_fit": "original column-scaled damped Newton, identical stationarity acceptance and condition cutoff; no ridge, rank truncation or stable retry",
        "training_evaluation": "direct single translated radial kernel columns, gradients and Laplacians",
        "diagnostic_evaluation": "direct raw order96; stable reevaluation discrepancy is separate postprocessing",
        "interpretation": "exact reduced objective is coordinate-invariant; comparison isolates finite-precision raw column evaluation and coefficient conditioning"},
        "source_sha256": hashes, "software": {"python": platform.python_version(), "numpy": np.__version__,
                                                  "scipy": E.scipy.__version__, "torch": torch.__version__}, "runs": []}
    (args.output/"manifest_before_runs.json").write_text(json.dumps(payload, indent=2)+"\n")
    payload["validation"] = validate()
    (args.output/"validation.json").write_text(json.dumps(payload["validation"], indent=2)+"\n")
    (args.output/"source_matern_raw.py").write_bytes(Path(__file__).read_bytes())
    tasks = [(seed, args.max_solves) for seed in args.seeds]
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.jobs)
        outputs = pool.map(_run_one, tasks)
    else:
        pool, outputs = None, map(_run_one, tasks)
    for run in outputs:
        payload["runs"].append(run)
        (args.output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    if pool is not None:
        pool.shutdown()


def _run_one(task):
    seed, max_solves = task
    return postprocess(RawSolver(paper_args(max_solves), "manufactured_jet", 1.0, seed, "ordinary").run())


if __name__ == "__main__":
    main()
