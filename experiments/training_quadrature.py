"""Training-quadrature floor at the exact completed target; no optimization.

The geometry is fixed at the manufactured target, where the continuum
minimizer is represented exactly, and only the solver's own coefficient fit
is performed with training quadrature of order q. The relative H1 error of
that fit, evaluated with independent higher-order quadrature, is the error
that coefficient discretization alone introduces at the target. It is not a
lower bound on trained errors, since geometry optimization can offset part of
it, but it sets the scale below which differences between trained states
cannot be read without higher-order training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {"matern": ROOT/"experiments/matern_swift_hohenberg_2d/experiment.py",
           "poisson": ROOT/"experiments/poisson_neumann_2d/experiment.py"}
DEFAULT_ORDERS = {"matern": (64, 96), "poisson": (32, 64)}


def load(study):
    path = STUDIES[study]
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"{study}_experiment", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def target_state(module, study):
    jet = module.MANUFACTURED_JET
    state = {"pairs": [{"center": list(c), "angle": angle, "half_gap": 0.0, "completed": True}
                       for c, angle, a, b in jet["pairs"]],
             "coefficients": [w for c, angle, a, b in jet["pairs"] for w in (a, b)]}
    if study == "matern":
        state["background_center"] = list(jet["background"][0])
        state["coefficients"].append(jet["background"][1])
    else:
        state["singletons"] = [list(c) for c, a in jet["singletons"]]
        state["coefficients"].extend(a for c, a in jet["singletons"])
    return state


def refit(module, study, state, order):
    points, weights = module.quadrature(order)
    boundary_points, boundary_weights, normals = module.boundary_quadrature(order)
    if study == "matern":
        values, _, laplacians = module.basis_fields(state, points)
        bvalues, bgradients, _ = module.basis_fields(state, boundary_points)
        bnormal = np.einsum("npd,nd->np", bgradients, normals)
        force = module.forcing_values("manufactured_jet", 1.0, points)
        load_m, load_q = module.manufactured_boundary_data(boundary_points, normals)
        coefficients, fit = module.fit_coefficients(values, laplacians, weights, force, bvalues, bnormal,
                                                    boundary_weights, load_m, load_q)
    else:
        values, gradients, _ = module.basis_fields(state, points)
        bvalues, _, _ = module.basis_fields(state, boundary_points)
        force = module.manufactured_jet_load(points[:, 0], points[:, 1])
        neumann = module.manufactured_neumann_data(boundary_points, normals)
        coefficients, fit = module.fit_coefficients(values, gradients, weights, force, bvalues,
                                                    boundary_weights, neumann)
    return {**state, "coefficients": coefficients.tolist()}, fit


def relative_h1(module, state, order):
    points, weights = module.quadrature(order)
    exact = module.manufactured_jet_solution(points[:, 0], points[:, 1])
    gradient = module.manufactured_jet_gradient(points[:, 0], points[:, 1])
    value, derivative = module.evaluate_state(state, points)
    return math.sqrt((weights @ ((value-exact)**2 + np.sum((derivative-gradient)**2, axis=1)))
                     / (weights @ (exact**2 + np.sum(gradient**2, axis=1))))


def audit(study, orders, evaluation_orders):
    module = load(study)
    target = target_state(module, study)
    rows = []
    for order in orders:
        state, fit = refit(module, study, target, order)
        coefficient_error = (np.linalg.norm(np.subtract(state["coefficients"], target["coefficients"]))
                             / np.linalg.norm(target["coefficients"]))
        rows.append({"training_order": order, "relative_coefficient_error": float(coefficient_error),
                     "relative_h1_error": {str(e): relative_h1(module, state, e) for e in evaluation_orders},
                     "fit_criterion": fit["stationarity_criterion"]})
        print(f"{study} training order {order:3d}: H1 error "
              + ", ".join(f"@{e} {rows[-1]['relative_h1_error'][str(e)]:.3e}" for e in evaluation_orders), flush=True)
    return {"study": study, "definition": __doc__.split("\n\n")[1].strip(),
            "target_coefficients_error_at_evaluation_orders": {str(e): relative_h1(module, target, e) for e in evaluation_orders},
            "evaluation_orders": list(evaluation_orders), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=sorted(STUDIES), nargs="+", default=sorted(STUDIES))
    parser.add_argument("--evaluation-orders", type=int, nargs="+", default=[128])
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/audits/training_quadrature.json")
    args = parser.parse_args()
    report = {study: audit(study, DEFAULT_ORDERS[study], args.evaluation_orders) for study in args.study}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
