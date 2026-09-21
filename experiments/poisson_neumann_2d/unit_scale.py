"""Poisson-Neumann Deep Ritz experiment on a wider Gaussian realization class.

Solve -Laplacian u = f on Omega=(-1,1)^2 with the nonhomogeneous Neumann
condition du/dn = g, by minimizing the Deep Ritz energy

  J(u) = int_Omega [ 0.5 |grad u|^2 - f u ] dx - int_dOmega g u ds
         + (sigma/2) ( int_Omega u dx - M )^2 .

The pure Neumann problem determines u only up to an additive constant, so the
first two terms alone are neither strongly convex nor coercive on H^1 and
Assumption 1 fails. The last term restores both: the quadratic part
|grad u|^2 + sigma (int u)^2 is equivalent to the H^1 norm by the
Poincare-Wirtinger inequality. It also fixes the constant without changing the
equation, because the stationarity condition is

  -Laplacian u + sigma ( int_Omega u - M ) = f  in Omega,   du/dn = g  on dOmega,

and the manufactured target satisfies int u* = M, so at u* the extra term
vanishes and the interior equation is exactly -Laplacian u* = f. The load and
the boundary data are manufactured from u*, which is a completed-class state:
the exact minimizer is representable at zero gap and only approached at any
positive gap. The compatibility condition int f + int g = 0 holds identically,
since f = -Laplacian u* and g = du*/dn.

The class is wider than the one used for the fourth-order study: four pairs and
two singletons, ten multiplicity slots. The energy is quadratic in the
coefficients, so a coefficient solve is one linear solve rather than a Newton
loop; the work unit is one distinct coefficient solve.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy
from scipy.optimize import minimize
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SIGMA = 1.0          # weight of the mean-value term that makes J strongly convex

# Fixed initial anchors of the model: four pairs and two singletons.
ANCHOR_PAIRS = (((-0.45, -0.40), 0.30),
                ((0.42, -0.38), 1.20),
                ((-0.40, 0.44), 2.10),
                ((0.46, 0.41), 2.80))
ANCHOR_SINGLETONS = ((-0.05, 0.10), (0.12, -0.08))
NPAIRS = len(ANCHOR_PAIRS)
NSINGLETONS = len(ANCHOR_SINGLETONS)
NSLOTS = 2*NPAIRS+NSINGLETONS

# The manufactured target: a completed-class state near, but not at, the
# anchors. Its blocks are (centre, angle, even weight, odd weight).
MANUFACTURED_JET = {
    "pairs": (((-0.47, -0.38), 0.34, 1.1, -1.3),
              ((0.44, -0.36), 1.26, -0.8, 1.5),
              ((-0.38, 0.46), 2.04, 1.4, 0.9),
              ((0.48, 0.39), 2.86, -1.2, -1.1)),
    "singletons": (((-0.06, 0.12), 0.7), ((0.10, -0.10), -0.5)),
}


def _manufactured_jet_parts(x, y):
    """u*, its gradient and its Laplacian, from exact Gaussian identities.

    For K(y)=exp(-|y|^2/2) and P=(d.y)K, with r=x-c and t=d.r,
      grad K = -r K,             Lap K = (|r|^2-2) K,
      grad P = (d - t r) K,      Lap P = t (|r|^2-4) K.
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    G = np.zeros(x.shape); Gx = np.zeros(x.shape); Gy = np.zeros(x.shape)
    GL = np.zeros(x.shape)
    for (cx, cy), angle, a, b in MANUFACTURED_JET["pairs"]:
        rx, ry = x-cx, y-cy
        r2 = rx*rx+ry*ry
        K = np.exp(-0.5*r2)
        dx, dy = math.cos(angle), math.sin(angle)
        t = dx*rx+dy*ry
        G = G + a*K + b*(t*K)
        Gx = Gx + a*(-rx*K) + b*((dx-t*rx)*K)
        Gy = Gy + a*(-ry*K) + b*((dy-t*ry)*K)
        GL = GL + a*((r2-2)*K) + b*(t*(r2-4)*K)
    for (cx, cy), a0 in MANUFACTURED_JET["singletons"]:
        rx, ry = x-cx, y-cy
        r2 = rx*rx+ry*ry
        K = np.exp(-0.5*r2)
        G = G + a0*K; Gx = Gx + a0*(-rx*K); Gy = Gy + a0*(-ry*K)
        GL = GL + a0*((r2-2)*K)
    return G, Gx, Gy, GL


def manufactured_jet_solution(x, y):
    return _manufactured_jet_parts(x, y)[0]


def manufactured_jet_gradient(x, y):
    _, Gx, Gy, _ = _manufactured_jet_parts(x, y)
    return np.stack((Gx, Gy), axis=-1)


def manufactured_jet_load(x, y):
    """The load f = -Laplacian u*."""
    return -_manufactured_jet_parts(x, y)[3]


def manufactured_neumann_data(points, normals):
    """g = du*/dn for the prescribed outward normals."""
    gradient = manufactured_jet_gradient(points[:, 0], points[:, 1])
    return np.sum(gradient*normals, axis=1)


def quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel())), (weights[:, None]*weights[None, :]).ravel()


def boundary_quadrature(order: int):
    """Gauss-Legendre nodes, weights and outward normals on the four edges."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points, quad, normals = [], [], []
    for axis, value, normal in ((0, -1.0, (-1.0, 0.0)), (0, 1.0, (1.0, 0.0)),
                                (1, -1.0, (0.0, -1.0)), (1, 1.0, (0.0, 1.0))):
        edge = np.empty((order, 2))
        edge[:, axis] = value
        edge[:, 1-axis] = nodes
        points.append(edge); quad.append(weights)
        normals.append(np.repeat(np.asarray(normal)[None, :], order, axis=0))
    return np.vstack(points), np.concatenate(quad), np.vstack(normals)


def target_mean(order: int = 96) -> float:
    """M = int_Omega u* dx, the constant the mean-value term pins."""
    points, weights = quadrature(order)
    return float(np.dot(weights, manufactured_jet_solution(points[:, 0], points[:, 1])))


def _sinhc(z):
    small = np.abs(z) < 1e-3
    safe = np.where(small, 1.0, z)
    return np.where(small, 1+z*z/6+z**4/120+z**6/5040, np.sinh(z)/safe)


def basis_fields(state: dict, points: np.ndarray):
    """Stable pair and singleton columns, their gradients and Laplacians.

    Each pair contributes A_h = [K(.-c-hd)+K(.-c+hd)]/2 and
    B_h = [K(.-c-hd)-K(.-c+hd)]/(2h), evaluated by the hyperbolic forms
    E cosh(z) and E t sinhc(z) rather than by subtracting nearby Gaussians.
    At h=0 they are K and -D_dK, and the same holds for their Laplacians.
    """
    points = np.asarray(points, dtype=float)
    values, gradients, laplacians = [], [], []
    for pair in state["pairs"]:
        r = points-np.asarray(pair["center"])
        direction = np.array([math.cos(pair["angle"]), math.sin(pair["angle"])])
        t = r @ direction
        h = 0.0 if pair["completed"] else pair["half_gap"]
        z = h*t
        r2 = np.sum(r*r, axis=1)
        envelope = np.exp(-0.5*r2-0.5*h*h)
        ch, sh, sc = np.cosh(z), np.sinh(z), _sinhc(z)
        a = envelope*ch
        b = envelope*t*sc
        ga = -r*a[:, None]+direction[None, :]*(envelope*h*sh)[:, None]
        gb = -r*b[:, None]+direction[None, :]*(envelope*ch)[:, None]
        values.extend((a, b)); gradients.extend((ga, gb))
        laplacians.append(envelope*((r2+h*h-2)*ch-2*z*sh))
        laplacians.append(envelope*t*((r2+h*h-2)*sc-2*ch))
    for centre in state["singletons"]:
        r = points-np.asarray(centre)
        r2 = np.sum(r*r, axis=1)
        k = np.exp(-0.5*r2)
        values.append(k); gradients.append(-r*k[:, None])
        laplacians.append((r2-2)*k)
    return (np.stack(values, axis=1), np.stack(gradients, axis=1),
            np.stack(laplacians, axis=1))


def evaluate_state(state: dict, points: np.ndarray):
    values, gradients, _ = basis_fields(state, points)
    c = np.asarray(state["coefficients"])
    return values @ c, np.einsum("npd,p->nd", gradients, c)


def evaluate_state_laplacian(state: dict, points: np.ndarray) -> np.ndarray:
    return basis_fields(state, points)[2] @ np.asarray(state["coefficients"])


def relative_residual(state: dict, order: int) -> float:
    """Normalized strong residual of -Laplacian u - f over the domain."""
    points, weights = quadrature(order)
    force = manufactured_jet_load(points[:, 0], points[:, 1])
    residual = -evaluate_state_laplacian(state, points)-force
    return float(np.sqrt(np.dot(weights, residual*residual)
                         / np.dot(weights, force*force)))


def energy(state: dict, order: int) -> dict:
    points, weights = quadrature(order)
    u, gradient = evaluate_state(state, points)
    force = manufactured_jet_load(points[:, 0], points[:, 1])
    gradient_term = float(0.5*np.dot(weights, np.sum(gradient*gradient, axis=1)))
    load_term = float(np.dot(weights, force*u))
    bpoints, bweights, bnormals = boundary_quadrature(order)
    bvalues, _ = evaluate_state(state, bpoints)
    boundary_term = float(np.dot(bweights, manufactured_neumann_data(bpoints, bnormals)*bvalues))
    mean = float(np.dot(weights, u))
    mean_term = float(0.5*SIGMA*(mean-TARGET_MEAN)**2)
    return {"quadrature_order": order,
            "energy": gradient_term-load_term-boundary_term+mean_term,
            "gradient_energy": gradient_term, "load": load_term,
            "neumann_work": boundary_term, "mean_term": mean_term,
            "state_mean": mean,
            "h1_norm": float(np.sqrt(np.dot(weights, u*u+np.sum(gradient*gradient, axis=1))))}


def initial_state(seed: int, half_gap_range=(0.35, 0.5)) -> dict:
    low, high = map(float, half_gap_range)
    if not (0 < low <= high <= 1.5):
        raise ValueError("Initial half-gaps must satisfy 0 < min <= max <= 1.5")
    rng = np.random.default_rng(seed)
    pairs = []
    for centre, angle in ANCHOR_PAIRS:
        pairs.append({"center": (np.asarray(centre)+rng.uniform(-0.10, 0.10, 2)).tolist(),
                      "angle": float(angle+rng.uniform(-0.18, 0.18)),
                      "half_gap": float(rng.uniform(low, high)), "completed": False})
    singletons = [(np.asarray(c)+rng.uniform(-0.10, 0.10, 2)).tolist()
                  for c in ANCHOR_SINGLETONS]
    return {"pairs": pairs, "singletons": singletons, "coefficients": [0.0]*NSLOTS}


class WorkLimit(Exception):
    pass


class TransferReady(Exception):
    pass


class RestartReady(Exception):
    pass


class NumericalFailure(Exception):
    pass


def pack_geometry(state):
    vector, bounds = [], []
    for pair in state["pairs"]:
        vector.extend(pair["center"]); bounds.extend([(-2.0, 2.0)]*2)
        vector.append(pair["angle"]); bounds.append((None, None))
        if not pair["completed"]:
            vector.append(math.log(pair["half_gap"])); bounds.append((None, math.log(1.5)))
    for centre in state["singletons"]:
        vector.extend(centre); bounds.extend([(-2.0, 2.0)]*2)
    return np.asarray(vector), bounds


def unpack_geometry(vector, template):
    state = copy.deepcopy(template)
    cursor = 0
    for pair in state["pairs"]:
        pair["center"] = vector[cursor:cursor+2].tolist()
        pair["angle"] = float(vector[cursor+2]); cursor += 3
        if not pair["completed"]:
            pair["half_gap"] = float(np.exp(vector[cursor])); cursor += 1
            if pair["half_gap"] == 0.0:
                raise NumericalFailure("Finite half-gap underflow: no positive floor was imposed")
    singletons = []
    for _ in state["singletons"]:
        singletons.append(vector[cursor:cursor+2].tolist()); cursor += 2
    state["singletons"] = singletons
    return state


def _torch_basis(vector, template, points):
    values, gradients, cursor = [], [], 0
    for pair in template["pairs"]:
        center = vector[cursor:cursor+2]; angle = vector[cursor+2]; cursor += 3
        direction = torch.stack((torch.cos(angle), torch.sin(angle)))
        if pair["completed"]:
            h = vector.new_tensor(0.0)
        else:
            h = torch.exp(vector[cursor]); cursor += 1
        r = points-center; t = r @ direction; z = h*t
        envelope = torch.exp(-0.5*torch.sum(r*r, dim=1)-0.5*h*h)
        small = torch.abs(z) < 1e-3
        safe = torch.where(small, torch.ones_like(z), z)
        sinhc = torch.where(small, 1+z*z/6+z**4/120+z**6/5040, torch.sinh(z)/safe)
        a, b = envelope*torch.cosh(z), envelope*t*sinhc
        ga = -r*a[:, None]+direction[None, :]*(envelope*h*torch.sinh(z))[:, None]
        gb = -r*b[:, None]+direction[None, :]*(envelope*torch.cosh(z))[:, None]
        values.extend((a, b)); gradients.extend((ga, gb))
    for _ in template["singletons"]:
        r = points-vector[cursor:cursor+2]; cursor += 2
        k = torch.exp(-0.5*torch.sum(r*r, dim=1))
        values.append(k); gradients.append(-r*k[:, None])
    return torch.stack(values, dim=1), torch.stack(gradients, dim=1)


def fit_coefficients(values, gradients, weights, force,
                     boundary_values, boundary_weights, neumann):
    """One linear solve for the quadratic Deep Ritz energy.

    J(c) = 0.5 c' S c - c' L + (sigma/2) (m'c - M)^2 with S the stiffness
    matrix, L the combined interior and Neumann load, and m the column means.
    Stationarity is (S + sigma m m') c = L + sigma M m. The matrix is positive
    definite on the span: S is positive semidefinite with only constants in its
    kernel, and the rank-one term is positive on them. Column scaling is a
    change of coordinates only; no ridge or rank truncation is applied.
    """
    stiffness = np.einsum("n,npd,nqd->pq", weights, gradients, gradients)
    means = values.T @ weights
    load = values.T @ (weights*force) + boundary_values.T @ (boundary_weights*neumann)
    scales = np.sqrt(np.maximum(np.diag(stiffness)
                                + np.einsum("n,np,np->p", weights, values, values), 1e-300))
    matrix = (stiffness+SIGMA*np.outer(means, means))/scales[:, None]/scales[None, :]
    vector = (load+SIGMA*TARGET_MEAN*means)/scales
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e14:
        raise NumericalFailure(f"Deep Ritz coefficient matrix is ill conditioned: {condition}")
    alpha = np.linalg.solve(matrix, vector)
    residual = float(np.linalg.norm(matrix @ alpha - vector))
    reference = max(1.0, float(np.linalg.norm(vector)))
    return alpha/scales, {"newton_iterations": 1, "newton_backtracks": 0,
                          "coefficient_stationarity": residual,
                          "relative_coefficient_stationarity": residual/reference,
                          "gradient_basis_condition": condition}


def transfer_bound(state, pair_index):
    """Analytic whole-plane H1 bound on the change made by closing one pair.

    There is no boundary multiplier in this Neumann formulation, so the
    whole-plane bound applies to the trial function directly and no multiplier
    factor is needed. Omega is contained in the plane, so the bound is valid on
    Omega as well.
    """
    pair = state["pairs"][pair_index]
    a, b = state["coefficients"][2*pair_index:2*pair_index+2]
    h = pair["half_gap"]
    whole_plane = h*h*(abs(a)*math.sqrt(3*math.pi)/2+abs(b)*math.sqrt(75*math.pi/8)/6)
    return {"absolute_h1_bound_float64": whole_plane,
            "whole_plane_pair_bound_float64": whole_plane,
            "formula": "h^2*(abs(a)*sqrt(3*pi)/2+abs(b)*sqrt(75*pi/8)/6)",
            "certification": "Analytic bound evaluated in float64; not a validated rounding enclosure",
            "note": "Stored pair coefficients, centre and angle are preserved; only h becomes zero"}


def diagnostics(state, order):
    result = energy(state, order)
    raw = []
    for index, pair in enumerate(state["pairs"]):
        a, b = state["coefficients"][2*index:2*index+2]
        if not pair["completed"]:
            h = pair["half_gap"]
            raw.extend((a/2+b/(2*h), a/2-b/(2*h)))
        else:
            raw.append(math.hypot(a, b))
    for index in range(NSINGLETONS):
        raw.append(state["coefficients"][2*NPAIRS+index])
    result.update({"stable_coefficient_norm": float(np.linalg.norm(state["coefficients"])),
                   "kernel_weight_norm": float(np.linalg.norm(raw)),
                   "completed_pairs": sum(pair["completed"] for pair in state["pairs"]),
                   "multiplicity_budget": NSLOTS})
    return result


TARGET_MEAN = target_mean()


class Solver:
    def __init__(self, args, seed, method, restart_checkpoint=None):
        self.args, self.seed, self.method = args, seed, method
        self.points, self.weights = quadrature(args.quadrature_order)
        (self.boundary_points, self.boundary_weights,
         self.boundary_normals) = boundary_quadrature(args.quadrature_order)
        self.neumann = manufactured_neumann_data(self.boundary_points, self.boundary_normals)
        self.force = manufactured_jet_load(self.points[:, 0], self.points[:, 1])
        self.points_t = torch.from_numpy(self.points)
        self.weights_t = torch.from_numpy(self.weights)
        self.force_t = torch.from_numpy(self.force)
        self.boundary_points_t = torch.from_numpy(self.boundary_points)
        self.boundary_weights_t = torch.from_numpy(self.boundary_weights)
        self.neumann_t = torch.from_numpy(self.neumann)
        self.state = initial_state(seed, (args.initial_half_gap_min, args.initial_half_gap_max))
        self.solves = self.accepted_steps = self.newton_iterations = 0
        self.cache, self.history, self.transfers, self.blocks = {}, [], [], []
        self.restart_checkpoint = copy.deepcopy(restart_checkpoint)
        self.restart_events = []

    def scheduled_restart_ready(self):
        """Restart the finite optimizer at the completion branch's first switch."""
        checkpoint = self.restart_checkpoint
        if self.method != "finite_restart" or checkpoint is None or self.restart_events:
            return False
        if self.solves < checkpoint["solve_count"]:
            return False
        actual, _ = pack_geometry(self.state)
        expected, _ = pack_geometry(checkpoint["before_state"])
        if self.solves != checkpoint["solve_count"] or not np.array_equal(actual, expected):
            raise NumericalFailure("Finite restart did not reproduce the saved completion checkpoint exactly")
        if not np.array_equal(self.state["coefficients"], checkpoint["before_state"]["coefficients"]):
            raise NumericalFailure("Finite restart checkpoint coefficients disagree")
        self.restart_events.append({"solve_count": self.solves, "state": copy.deepcopy(self.state),
                                    "matched_completion_pair": checkpoint["pair_index"],
                                    "remaining_fit_budget": self.args.max_solves-self.solves,
                                    "geometry_and_coefficients_agree_exactly": True})
        self.record("restart", "matched_finite_optimizer_restart")
        return True

    def evaluate(self, vector):
        signature = (tuple(pair["completed"] for pair in self.state["pairs"]),
                     np.asarray(vector).tobytes())
        if signature in self.cache:
            return self.cache[signature]
        if self.solves >= self.args.max_solves:
            raise WorkLimit
        self.solves += 1
        geometry = torch.tensor(vector, requires_grad=True)
        values, gradients = _torch_basis(geometry, self.state, self.points_t)
        bvalues, _ = _torch_basis(geometry, self.state, self.boundary_points_t)
        coefficients, fit = fit_coefficients(
            values.detach().numpy(), gradients.detach().numpy(), self.weights, self.force,
            bvalues.detach().numpy(), self.boundary_weights, self.neumann)
        self.newton_iterations += fit["newton_iterations"]
        c = torch.from_numpy(coefficients)
        u = values @ c
        field = torch.einsum("npd,p->nd", gradients, c)
        interior = torch.sum(self.weights_t*(0.5*torch.sum(field*field, dim=1)-self.force_t*u))
        boundary = torch.sum(self.boundary_weights_t*self.neumann_t*(bvalues @ c))
        mean = torch.sum(self.weights_t*u)
        objective = interior-boundary+0.5*SIGMA*(mean-TARGET_MEAN)**2
        objective.backward()
        state = unpack_geometry(np.asarray(vector), self.state)
        state["coefficients"] = coefficients.tolist()
        result = {"objective": float(objective.detach()), "gradient": geometry.grad.detach().numpy(),
                  "state": state, "solve_count": self.solves, **fit}
        if not np.isfinite(result["objective"]) or not np.all(np.isfinite(result["gradient"])):
            raise NumericalFailure("Nonfinite geometry objective or gradient")
        self.cache[signature] = result
        return result

    def objective(self, vector):
        result = self.evaluate(vector)
        return result["objective"], result["gradient"]

    def record(self, phase, event):
        self.history.append({"solve_count": self.solves, "phase": phase, "event": event,
                             "state": copy.deepcopy(self.state),
                             **diagnostics(self.state, self.args.quadrature_order)})

    def eligible_transfers(self):
        return [i for i, pair in enumerate(self.state["pairs"])
                if not pair["completed"] and 2*pair["half_gap"] <= self.args.proximity_threshold
                and transfer_bound(self.state, i)["absolute_h1_bound_float64"] <= self.args.transfer_tolerance]

    def accepted(self, vector):
        self.state = copy.deepcopy(self.evaluate(vector)["state"])
        self.record("optimization", "accepted_iterate")
        self.accepted_steps += 1
        if self.scheduled_restart_ready():
            raise RestartReady
        if (self.method == "completion" and self.accepted_steps % self.args.block_iterations == 0
                and self.eligible_transfers()):
            raise TransferReady

    def attempt_transfers(self):
        if self.method != "completion":
            return False
        changed = False
        for i in self.eligible_transfers():
            if self.solves >= self.args.max_solves:
                break
            before = copy.deepcopy(self.state)
            bound = transfer_bound(before, i)
            self.state["pairs"][i].update({"completed": True, "half_gap": 0.0})
            vector, _ = pack_geometry(self.state)
            try:
                self.state = copy.deepcopy(self.evaluate(vector)["state"])
            except WorkLimit:
                self.state = before
                break
            self.transfers.append({"solve_count": self.solves, "pair_index": i,
                                   "before_state": before, "after_state": copy.deepcopy(self.state),
                                   "bound": bound})
            self.record("transfer", f"completed_pair_{i}")
            changed = True
        return changed

    def run(self):
        started = time.perf_counter()
        status, initial = "running", None
        error = None
        try:
            vector, _ = pack_geometry(self.state)
            self.state = copy.deepcopy(self.evaluate(vector)["state"])
            initial = copy.deepcopy(self.state)
            self.record("initial", "coefficient_solve")
            while self.solves < self.args.max_solves:
                vector, bounds = pack_geometry(self.state)
                before = self.solves
                result = None
                try:
                    result = minimize(self.objective, vector, jac=True, method="L-BFGS-B",
                                      bounds=bounds, callback=self.accepted,
                                      options={"maxiter": self.args.max_solves,
                                               "maxfun": self.args.max_solves-self.solves,
                                               "ftol": self.args.ftol, "gtol": self.args.gtol,
                                               "maxls": 40, "maxcor": 12})
                    self.state = copy.deepcopy(self.evaluate(result.x)["state"])
                    self.blocks.append({"start_solve": before, "end_solve": self.solves,
                                        "status": int(result.status), "success": bool(result.success),
                                        "message": str(result.message), "iterations": int(result.nit)})
                except TransferReady:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "transfer_ready"})
                except RestartReady:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "matched_finite_restart"})
                except WorkLimit:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "work_limit"})
                    break
                if result is not None and self.scheduled_restart_ready():
                    continue
                transferred = self.attempt_transfers()
                if not transferred and result is not None and result.success:
                    status = "optimizer_converged"; break
                if not transferred and result is not None and result.status != 1:
                    status = "optimizer_failure"; break
                if before == self.solves:
                    status = "no_new_evaluations"; break
        except Exception as exc:
            status = "numerical_failure"
            error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        self.record("final", "returned_state")
        audits = [energy(self.state, order) for order in (64, 96)]
        fitted = list(self.cache.values())
        output = {"seed": self.seed, "method": self.method,
                  "initial_state": initial, "final_state": copy.deepcopy(self.state),
                  "history": self.history, "transfers": self.transfers,
                  "optimizer_blocks": self.blocks, "status": status, "error": error,
                  "restart_events": self.restart_events,
                  "restart_checkpoint_from_completion": self.restart_checkpoint,
                  "coefficient_solves": self.solves,
                  "final_diagnostics": diagnostics(self.state, self.args.quadrature_order),
                  "relative_strong_residual": relative_residual(self.state, 96),
                  "independent_quadrature": audits,
                  "energy_order64_order96_difference": abs(audits[0]["energy"]-audits[1]["energy"]),
                  "maximum_relative_coefficient_stationarity": max((x["relative_coefficient_stationarity"] for x in fitted), default=None),
                  "maximum_gradient_basis_condition": max((x["gradient_basis_condition"] for x in fitted), default=None),
                  "elapsed_seconds": time.perf_counter()-started}
        print(f"seed={self.seed} {self.method}: {status}; solves={self.solves}; "
              f"J96={audits[-1]['energy']:.10g}; transfers={len(self.transfers)}", flush=True)
        return output


def validate_gradients(args):
    reports = []
    for completed in (False, True):
        solver = Solver(args, 1729, "ordinary")
        if completed:
            solver.state["pairs"][0].update({"completed": True, "half_gap": 0.0})
        vector, _ = pack_geometry(solver.state)
        result = solver.evaluate(vector)
        finite_difference = []
        for i in range(len(vector)):
            step = 2e-5*max(1, abs(vector[i]))
            positive, negative = vector.copy(), vector.copy()
            positive[i] += step; negative[i] -= step
            finite_difference.append((solver.evaluate(positive)["objective"]
                                      - solver.evaluate(negative)["objective"])/(2*step))
        finite_difference = np.asarray(finite_difference)
        error = float(np.max(np.abs(finite_difference-result["gradient"])))
        relative = error/max(1e-10, float(np.max(np.abs(finite_difference))),
                             float(np.max(np.abs(result["gradient"]))))
        reports.append({"one_completed_pair": completed, "maximum_absolute_error": error,
                        "relative_maximum_error": relative, "passed": relative < 2e-5})
    if not all(x["passed"] for x in reports):
        raise NumericalFailure(f"Geometry envelope-gradient validation failed: {reports}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--max-solves", type=int, default=8000)
    parser.add_argument("--quadrature-order", type=int, default=32)
    parser.add_argument("--block-iterations", type=int, default=40)
    parser.add_argument("--proximity-threshold", type=float, default=0.12)
    parser.add_argument("--transfer-tolerance", type=float, default=3e-2)
    parser.add_argument("--ftol", type=float, default=0.0)
    parser.add_argument("--gtol", type=float, default=0.0)
    parser.add_argument("--initial-half-gap-min", type=float, default=0.02)
    parser.add_argument("--initial-half-gap-max", type=float, default=0.05)
    parser.add_argument("--finite-restart-control", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory exists; preserve prior results and choose a new path")
    args.output.mkdir(parents=True)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update({
        "domain": "(-1,1)^2", "equation": "-Laplacian u = f",
        "boundary": "du/dn = g, nonhomogeneous Neumann, manufactured from u*",
        "objective": "integral (0.5|grad u|^2 - f u) - boundary g u + (sigma/2)(integral u - M)^2",
        "sigma": SIGMA, "target_mean": TARGET_MEAN,
        "kernel": "Gaussian exp(-|y|^2/2); stable pair columns E cosh z and E t sinhc z",
        "multiplicity_budget": NSLOTS,
        "pairing": f"{NPAIRS} supplied pairs plus {NSINGLETONS} singletons",
        "geometry_bounds": "centres [-2,2]^2, angles unbounded, 0<h<=1.5 without positive floor",
        "coefficient_fit": "one linear solve of (S + sigma m m') c = L + sigma M m after column scaling",
        "work_accounting": "one unit per distinct coefficient solve, including line-search probes and post-transfer refits",
        "optimizer": "L-BFGS-B ftol0 gtol0 maxcor12 maxls40; history reset only on actual completion",
        "transfer": "analytic whole-plane H1 pair bound; no multiplier factor is needed in this Neumann formulation",
        "limitations": "fixed Gaussian scale and supplied pairing; no certified objective gate; quadrature agreement is not validated integration"})
    methods = ("ordinary", "completion", "finite_restart") if args.finite_restart_control else ("ordinary", "completion")
    payload = {"schema_version": 1, "configuration": configuration,
               "software": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "torch": torch.__version__},
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "planned_runs": [{"seed": s, "method": m} for s in args.seeds for m in methods],
               "runs": []}
    (args.output/"manifest_before_runs.json").write_text(json.dumps(payload, indent=2)+"\n")
    (args.output/"source_experiment.py").write_bytes(Path(__file__).read_bytes())
    payload["gradient_validation"] = validate_gradients(args)
    (args.output/"gradient_validation.json").write_text(json.dumps(payload["gradient_validation"], indent=2)+"\n")
    for plan in payload["planned_runs"]:
        checkpoint = None
        if plan["method"] == "finite_restart":
            run = next(r for r in payload["runs"] if r["method"] == "completion" and r["seed"] == plan["seed"])
            if run["transfers"]:
                checkpoint = run["transfers"][0]
        output = Solver(args, plan["seed"], plan["method"], checkpoint).run()
        payload["runs"].append(output)
        (args.output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        with (args.output/"summary.csv").open("w", newline="") as stream:
            columns = ["seed", "method", "status", "coefficient_solves", "transfers",
                       "energy", "relative_strong_residual", "elapsed_seconds"]
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for run in payload["runs"]:
                writer.writerow({k: len(run["transfers"]) if k == "transfers" else
                                 run.get(k, run["final_diagnostics"].get(k)) for k in columns})


if __name__ == "__main__":
    main()
