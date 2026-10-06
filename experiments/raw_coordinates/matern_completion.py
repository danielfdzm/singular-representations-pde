"""Raw finite-pair Matérn training with optional conversion to derivative blocks.

Finite pairs use ordinary weights and direct translated kernels. Only after a
transfer do their two columns become Phi and -D_d Phi. The original experiment's
Newton fit, optimizer, work/stopping rules, and manufactured problem are reused.
Stable positive-gap columns are confined to validation and passive diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

RAW_SOURCE = Path(__file__).with_name("matern_raw.py")
spec = importlib.util.spec_from_file_location("matern_raw_for_completion", RAW_SOURCE)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
E, np, torch = R.E, R.np, R.torch


def basis_fields(state, points):
    values, gradients, laps = [], [], []
    points = np.asarray(points)
    for pair in state["pairs"]:
        center = np.asarray(pair["center"])
        direction = np.array([np.cos(pair["angle"]), np.sin(pair["angle"])])
        if pair["completed"]:
            r = points-center
            radius, t = np.sqrt(np.sum(r*r, axis=1)), r@direction
            c1, c1p = E.matern.radial("c1", radius), E.matern.radial("c1p", radius)
            values.extend((E.matern.radial("f0", radius), -c1*t))
            gradients.extend((r*c1[:, None], -(r*(c1p*t)[:, None]+direction*c1[:, None])))
            laps.extend((E.matern.radial("lap", radius), -E.matern.radial("c3", radius)*t))
        else:
            for translated_center in (center+pair["half_gap"]*direction, center-pair["half_gap"]*direction):
                r = points-translated_center
                radius = np.sqrt(np.sum(r*r, axis=1))
                values.append(E.matern.radial("f0", radius))
                gradients.append(r*E.matern.radial("c1", radius)[:, None])
                laps.append(E.matern.radial("lap", radius))
    r = points-np.asarray(state["background_center"])
    radius = np.sqrt(np.sum(r*r, axis=1))
    values.append(E.matern.radial("f0", radius))
    gradients.append(r*E.matern.radial("c1", radius)[:, None])
    laps.append(E.matern.radial("lap", radius))
    return np.stack(values, axis=1), np.stack(gradients, axis=1), np.stack(laps, axis=1)


def torch_basis(vector, template, points):
    values, gradients, laps, cursor = [], [], [], 0
    for pair in template["pairs"]:
        center, angle = vector[cursor:cursor+2], vector[cursor+2]; cursor += 3
        direction = torch.stack((torch.cos(angle), torch.sin(angle)))
        if pair["completed"]:
            r = points-center
            radius = torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-30))
            t = r@direction
            c1 = E.matern.torch_radial(torch, "c1", radius)
            c1p = E.matern.torch_radial(torch, "c1p", radius)
            values.extend((E.matern.torch_radial(torch, "f0", radius), -c1*t))
            gradients.extend((r*c1[:, None], -(r*(c1p*t)[:, None]+direction*c1[:, None])))
            laps.extend((E.matern.torch_radial(torch, "lap", radius), -E.matern.torch_radial(torch, "c3", radius)*t))
        else:
            h = torch.exp(vector[cursor]); cursor += 1
            for translated_center in (center+h*direction, center-h*direction):
                r = points-translated_center
                radius = torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-30))
                values.append(E.matern.torch_radial(torch, "f0", radius))
                gradients.append(r*E.matern.torch_radial(torch, "c1", radius)[:, None])
                laps.append(E.matern.torch_radial(torch, "lap", radius))
    r = points-vector[cursor:cursor+2]
    radius = torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-30))
    values.append(E.matern.torch_radial(torch, "f0", radius))
    gradients.append(r*E.matern.torch_radial(torch, "c1", radius)[:, None])
    laps.append(E.matern.torch_radial(torch, "lap", radius))
    return torch.stack(values, dim=1), torch.stack(gradients, dim=1), torch.stack(laps, dim=1)


def to_stable_state(state):
    result = copy.deepcopy(state)
    for i, pair in enumerate(state["pairs"]):
        if not pair["completed"]:
            plus, minus = state["coefficients"][2*i:2*i+2]
            result["coefficients"][2*i:2*i+2] = [plus+minus, pair["half_gap"]*(plus-minus)]
    return result


def transfer_predictor(state, index):
    """Convert this pair's ordinary weights to the actual zero-gap block."""
    result = copy.deepcopy(state)
    pair = result["pairs"][index]
    if pair["completed"]:
        raise ValueError("Pair is already completed")
    plus, minus = state["coefficients"][2*index:2*index+2]
    result["coefficients"][2*index:2*index+2] = [plus+minus, pair["half_gap"]*(plus-minus)]
    pair.update(completed=True, half_gap=0.0)
    return result


def evaluate_state(state, points):
    v, g, _ = basis_fields(state, points)
    c = np.asarray(state["coefficients"])
    return v@c, np.einsum("npd,p->nd", g, c)


def stable_evaluate_state(state, points):
    stable = to_stable_state(state)
    v, g, _ = R._stable_basis_fields(stable, points)
    c = np.asarray(stable["coefficients"])
    return v@c, np.einsum("npd,p->nd", g, c)


def transfer_bound(state, index, order=64):
    points, weights = E.quadrature(order)
    before, gb = evaluate_state(state, points)
    after, ga = evaluate_state(transfer_predictor(state, index), points)
    h1 = R._h1(after-before, ga-gb, weights)
    return {"absolute_h1_bound_float64": h1, "measured_h1_state_change": h1,
            "quadrature_order": order, "formula": "direct mixed-state H1 difference after explicit raw-weight conversion",
            "certification": "Measured float64 quadrature difference, not a validated enclosure",
            "conversion": "a=w_plus+w_minus, b=h*(w_plus-w_minus), centre/angle retained"}


def remaining_weights(state):
    raw, order = [], []
    for i, pair in enumerate(state["pairs"]):
        if not pair["completed"]:
            raw.extend(state["coefficients"][2*i:2*i+2])
            order.extend([f"pair{i}_center_plus_hd", f"pair{i}_center_minus_hd"])
    return raw+[state["coefficients"][-1]], order+["singleton"]


def diagnostics(state, case, amplitude, order):
    result = E.energy(state, case, amplitude, order)
    weights, _ = remaining_weights(state)
    result.update(raw_weight_norm=float(np.linalg.norm(weights)),
                  remaining_raw_pair_coefficient_norm=float(np.linalg.norm(weights[:-1])),
                  completed_pairs=sum(p["completed"] for p in state["pairs"]), multiplicity_budget=5)
    return result


E.basis_fields, E.diagnostics = basis_fields, diagnostics


class MixedSolver(E.Solver):
    def __init__(self, *args):
        super().__init__(*args)
        self.transfer_attempts = []

    def columns(self, geometry):
        values, _, laps = torch_basis(geometry, self.state, self.points_t)
        bvalues, bg, _ = torch_basis(geometry, self.state, self.boundary_points_t)
        return values, laps, bvalues, torch.einsum("npd,nd->np", bg, self.boundary_normals_t)

    def eligible_transfers(self):
        return [i for i, pair in enumerate(self.state["pairs"])
                if not pair["completed"] and 2*pair["half_gap"] <= self.args.proximity_threshold
                and transfer_bound(self.state, i)["absolute_h1_bound_float64"] <= self.args.transfer_tolerance]

    def attempt_transfers(self):
        if self.method != "completion":
            return False
        changed = False
        for i in self.eligible_transfers():
            if self.solves >= self.args.max_solves:
                break
            before = copy.deepcopy(self.state)
            bound = transfer_bound(before, i)  # recheck after any preceding nonlinear refit
            if bound["absolute_h1_bound_float64"] > self.args.transfer_tolerance:
                continue
            predictor = transfer_predictor(before, i)
            attempt = {"solve_count": self.solves, "accepted_steps": self.accepted_steps,
                       "pair_index": i, "before_state": before, "predictor_state": copy.deepcopy(predictor),
                       "bound": bound, "status": "pending"}
            self.transfer_attempts.append(attempt)
            self.state = predictor
            vector, _ = E.pack_geometry(self.state, self.coordinate)
            try:
                refit = self.evaluate(vector)
            except E.WorkLimit:
                self.state = before
                attempt["status"] = "work_limit_rolled_back"
                break
            except Exception as exc:
                self.state = before  # every failed refit returns the last accepted fitted state
                attempt.update(status="refit_failure_rolled_back", error_type=type(exc).__name__, error_message=str(exc))
                raise
            transfer = {"solve_count": attempt["solve_count"], "pair_index": i,
                        "before_state": before, "predictor_state": copy.deepcopy(predictor), "bound": bound,
                        "coefficient_conversion": "a=w_plus+w_minus; b=h*(w_plus-w_minus)"}
            for order in (80, 96):
                points, weights = E.quadrature(order)
                u, gu = evaluate_state(before, points)
                v, gv = evaluate_state(predictor, points)
                transfer[f"observed_transfer_h1_order{order}"] = R._h1(u-v, gu-gv, weights)
            self.transfers.append(transfer)
            self.history.append({"solve_count": attempt["solve_count"], "phase": "transfer",
                                 "event": "raw_weight_conversion_predictor", "state": copy.deepcopy(predictor),
                                 **diagnostics(predictor, self.case, self.amplitude, self.args.quadrature_order)})
            self.state = copy.deepcopy(refit["state"])
            transfer["after_state"] = copy.deepcopy(self.state)
            attempt.update(status="accepted", after_state=copy.deepcopy(self.state), refit_solve_count=refit["solve_count"])
            self.record("transfer", "nonlinear_refit")
            changed = True
            print(f"raw-completion seed={self.seed} q={self.args.quadrature_order}: pair={i} fit={self.solves} change={bound['measured_h1_state_change']:.3g}", flush=True)
        return changed

    def run(self):
        result = super().run()
        result["transfer_attempts"] = self.transfer_attempts
        return result


def paper_args(order=64, max_solves=8000):
    args = R.paper_args(max_solves)
    args.quadrature_order = order
    return args


def validate(order):
    args, reports = paper_args(order), []
    for mask in ((), (0,), (1,), (0, 1)):
        solver = MixedSolver(args, "manufactured_jet", 1.0, 1729, "completion")
        for index in mask:
            solver.state = transfer_predictor(solver.state, index)
        vector, _ = E.pack_geometry(solver.state)
        result = solver.evaluate(vector)
        finite = []
        for i in range(len(vector)):
            step = 2e-5*max(1.0, abs(vector[i]))
            plus, minus = vector.copy(), vector.copy()
            plus[i] += step; minus[i] -= step
            finite.append((solver.evaluate(plus)["objective"]-solver.evaluate(minus)["objective"])/(2*step))
        finite = np.asarray(finite)
        relative = float(np.max(np.abs(finite-result["gradient"]))) / max(
            1e-10, float(np.max(np.abs(finite))), float(np.max(np.abs(result["gradient"]))))
        stable_solver = E.Solver(args, "manufactured_jet", 1.0, 1729, "completion")
        stable_solver.state = to_stable_state(solver.state)
        stable_result = stable_solver.evaluate(vector)
        points, weights = E.quadrature(96)
        u, gu = evaluate_state(result["state"], points)
        sv, sg, _ = R._stable_basis_fields(stable_result["state"], points)
        sc = np.asarray(stable_result["state"]["coefficients"])
        v, gv = sv@sc, np.einsum("npd,p->nd", sg, sc)
        reports.append({"completed_pair_indices": list(mask), "finite_difference_relative_maximum_error": relative,
                        "raw_vs_stable_refit_relative_h1_difference": R._h1(u-v, gu-gv, weights)/R._h1(v, gv, weights),
                        "raw_vs_stable_reduced_gradient_max_absolute_difference": float(np.max(np.abs(result["gradient"]-stable_result["gradient"]))),
                        "analytic": result["gradient"].tolist(), "finite_difference": finite.tolist()})
    state = E.initial_state(0, (0.02, 0.05))
    state["coefficients"] = [25.0, -23.0, 30.0, -31.0, 0.7]
    points, weights = E.quadrature(96)
    conversions = []
    for index in (0, 1):
        stable_before = to_stable_state(state)
        stable_after = copy.deepcopy(stable_before)
        stable_after["pairs"][index].update(completed=True, half_gap=0.0)
        v0, g0, _ = R._stable_basis_fields(stable_before, points)
        v1, g1, _ = R._stable_basis_fields(stable_after, points)
        c = np.asarray(stable_before["coefficients"])
        independent = R._h1(v1@c-v0@c, np.einsum("npd,p->nd", g1-g0, c), weights)
        measured = transfer_bound(state, index, 96)["measured_h1_state_change"]
        predictor = transfer_predictor(state, index)
        u, gu = evaluate_state(predictor, points)
        discrepancy = R._h1(u-v1@c, gu-np.einsum("npd,p->nd", g1, c), weights)
        conversions.append({"pair_index": index, "measured_h1_change": measured,
                            "independent_stable_h1_change": independent,
                            "change_absolute_difference": abs(measured-independent),
                            "converted_predictor_realization_h1_difference": discrepancy})
    passed = all(r["finite_difference_relative_maximum_error"] < 2e-5
                 and r["raw_vs_stable_refit_relative_h1_difference"] < 1e-9
                 and r["raw_vs_stable_reduced_gradient_max_absolute_difference"] < 1e-8 for r in reports)
    passed &= all(c["change_absolute_difference"] < 1e-11 and c["converted_predictor_realization_h1_difference"] < 1e-11 for c in conversions)
    rollback_solver = MixedSolver(args, "manufactured_jet", 1.0, 0, "completion")
    vector, _ = E.pack_geometry(rollback_solver.state)
    rollback_solver.state = copy.deepcopy(rollback_solver.evaluate(vector)["state"])
    before = copy.deepcopy(rollback_solver.state)
    def failed_refit(vector):
        raise E.NumericalFailure("injected transfer refit failure for rollback validation")
    rollback_solver.evaluate = failed_refit
    try:
        rollback_solver.attempt_transfers()
    except E.NumericalFailure:
        pass
    rollback_passed = (rollback_solver.state == before and len(rollback_solver.transfer_attempts) == 1
                       and rollback_solver.transfer_attempts[0]["status"] == "refit_failure_rolled_back")
    passed &= rollback_passed
    report = {"passed": bool(passed), "training_quadrature_order": order, "geometry_gradients": reports,
              "physical_transfer_conversion": conversions,
              "failed_refit_rollback_passed": rollback_passed,
              "tolerances": {"finite_difference_relative": 2e-5, "refit_relative_h1": 1e-9,
                             "gradient_absolute": 1e-8, "transfer_absolute_h1": 1e-11}}
    if not passed:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def coefficient_residual(state, basis, order):
    points, weights = E.quadrature(order)
    bp, bw, bn = E.boundary_quadrature(order)
    v, _, lap = basis(state, points)
    bv, bg, _ = basis(state, bp)
    curvature = v+lap
    scales = np.sqrt(np.einsum("n,np,np->p", weights, curvature, curvature)+np.einsum("n,np,np->p", weights, v, v))
    scales = np.where(scales > 0, scales, 1.0)
    columns, curvature = v/scales, curvature/scales
    quadratic = np.einsum("n,np,nq->pq", weights, curvature, curvature)+E.RHO*np.einsum("n,np,nq->pq", weights, columns, columns)
    m, q = E.manufactured_boundary_data(bp, bn)
    load = (columns.T@(weights*E.forcing_values("manufactured_jet", 1.0, points))
            + (np.einsum("npd,nd->np", bg, bn)/scales).T@(bw*m)+(bv/scales).T@(bw*q))
    alpha = np.asarray(state["coefficients"])*scales
    residual = quadratic@alpha+columns.T@(weights*E.nonlinear_term(columns@alpha))-load
    return {"relative_residual": float(np.linalg.norm(residual)/max(1.0, np.linalg.norm(load))),
            "absolute_residual": float(np.linalg.norm(residual)), "quadratic_condition": float(np.linalg.cond(quadratic))}


def postprocess(run):
    points, weights = E.quadrature(96)
    target = E.manufactured_jet_solution(points[:, 0], points[:, 1])
    gradient = E.manufactured_jet_gradient(points[:, 0], points[:, 1])
    reference, cache = R._h1(target, gradient, weights), {}

    def annotate(state):
        key = json.dumps({k: state[k] for k in ("pairs", "background_center", "coefficients")}, sort_keys=True)
        if key not in cache:
            u, g = evaluate_state(state, points)
            v, gv = stable_evaluate_state(state, points)
            raw, order = remaining_weights(state)
            stable = to_stable_state(state)["coefficients"]
            cache[key] = {"relative_h1_error": R._h1(u-target, g-gradient, weights)/reference,
                          "absolute_h1_error": R._h1(u-target, g-gradient, weights), "diagnostic_quadrature_order": 96,
                          "stable_reevaluation_relative_h1_discrepancy": R._h1(u-v, g-gv, weights)/reference,
                          "stable_reevaluation_relative_h1_error": R._h1(v-target, gv-gradient, weights)/reference,
                          "stable_coefficient_norm": float(np.linalg.norm(stable)), "raw_weight_norm": float(np.linalg.norm(raw)),
                          "remaining_raw_pair_coefficient_norm": float(np.linalg.norm(raw[:-1]))}
        state["raw_weights"], state["raw_weights_order"] = remaining_weights(state)
        state["diagnostic_stable_coefficients"] = to_stable_state(state)["coefficients"]
        state["coefficient_semantics"] = ["Phi,-D_dPhi" if p["completed"] else "ordinary_plus_hd,ordinary_minus_hd" for p in state["pairs"]]+["singleton_Phi"]
        return cache[key]

    run["initial_diagnostics_order96"] = annotate(run["initial_state"])
    for row in run["history"]:
        row.update(annotate(row["state"]))
    run["final_diagnostics"].update(annotate(run["final_state"]))
    for event in run["transfers"]+run["transfer_attempts"]:
        for key in ("before_state", "predictor_state", "after_state"):
            if key in event:
                annotate(event[key])
    order = run["training_quadrature_order"]
    run["final_stationarity_diagnostics"] = {"quadrature_order": order, "refitted": False,
        "training_mixed_scaled_coordinates": coefficient_residual(run["final_state"], basis_fields, order),
        "stable_scaled_coordinates": coefficient_residual(to_stable_state(run["final_state"]), R._stable_basis_fields, order)}
    run["coefficient_coordinates"] = "ordinary weights for finite pairs; Phi,-D_dPhi coefficients for completed blocks"
    run["final_state_semantics"] = "last accepted fitted state; failed optimization probes and failed transfers do not replace it"
    return run


def _run_one(task):
    seed, method, order, cap = task
    return postprocess(MixedSolver(paper_args(order, cap), "manufactured_jet", 1.0, seed, method).run())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quadrature-order", type=int, choices=[64, 96], default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-solves", type=int, default=8000)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--reuse-ordinary", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve prior results")
    sources = [Path(__file__), RAW_SOURCE, R.SOURCE, R.SOURCE.with_name("matern.py")]
    hashes = {str(p.relative_to(R.SOURCE.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    reuse = []
    if args.reuse_ordinary:
        reused = json.loads(args.reuse_ordinary.read_text())
        reuse = [copy.deepcopy(r) for r in reused["runs"] if r["method"] == "ordinary" and r["seed"] in args.seeds]
        if len(reuse) != len(args.seeds) or any(r["training_quadrature_order"] != args.quadrature_order for r in reuse):
            parser.error("Reused raw ordinary records must match every requested seed and quadrature")
        for path, digest in reused["source_sha256"].items():
            if path in hashes and hashes[path] != digest:
                parser.error("Reused raw ordinary source hashes differ")
    methods = ["completion"] if reuse else ["ordinary", "completion"]
    tasks = [(seed, method, args.quadrature_order, args.max_solves) for seed in args.seeds for method in methods]
    payload = {"schema_version": 1, "configuration": {
        "case": "manufactured_jet", "amplitude": 1.0, "seeds": args.seeds, "max_solves": args.max_solves,
        "training_quadrature_order": args.quadrature_order, "diagnostic_quadrature_order": 96,
        "initial_half_gap_range": [0.02, 0.05], "methods": ["ordinary", "completion"], "gap_coordinate": "log",
        "finite_pair_columns": "direct translates at c+hd,c-hd with ordinary weights",
        "completed_pair_columns": "direct Phi,-D_dPhi at c with a=w_plus+w_minus,b=h*(w_plus-w_minus)",
        "coefficient_fit": "original scaled damped Newton, unchanged acceptance/cutoff, no ridge/truncation/stable retry",
        "optimizer": "original L-BFGS-B ftol0 gtol0 maxcor12 maxls40",
        "transfer": "original trigger: full gap<=0.05 and measured order64 H1 change<=0.03; checks every40 accepted iterates and after optimizer returns; sequential refits and rechecks",
        "evaluation": "direct raw finite translates/direct completed derivatives; stable finite-pair reevaluation is passive diagnostic only"},
        "source_sha256": hashes, "software": {"python": platform.python_version(), "numpy": np.__version__,
                                               "scipy": E.scipy.__version__, "torch": torch.__version__},
        "reused_ordinary": None if not reuse else {"path": str(args.reuse_ordinary), "sha256": hashlib.sha256(args.reuse_ordinary.read_bytes()).hexdigest(),
                                                   "source_sha256": reused["source_sha256"], "validation": reused["validation"]},
        "planned_new_runs": [{"seed": seed, "method": method, "quadrature_order": order} for seed, method, order, cap in tasks],
        "runs": reuse}
    args.output.mkdir(parents=True)
    (args.output/"manifest_before_runs.json").write_text(json.dumps(payload, indent=2)+"\n")
    for path in sources:
        (args.output/("source_"+path.name)).write_bytes(path.read_bytes())
    payload["validation"] = validate(args.quadrature_order)
    (args.output/"validation.json").write_text(json.dumps(payload["validation"], indent=2)+"\n")
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.jobs)
        outputs = pool.map(_run_one, tasks)
    else:
        pool, outputs = None, map(_run_one, tasks)
    for run in outputs:
        payload["runs"].append(run)
        payload["runs"].sort(key=lambda r: (r["seed"], r["method"]))
        (args.output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    if pool:
        pool.shutdown()


if __name__ == "__main__":
    main()
