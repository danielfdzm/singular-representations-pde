"""Swift-Hohenberg Ritz experiment on a Matern nu=9/2 realization class.

Minimize

  J(u) = int_Omega [ 0.5 ((1+Lap)u)^2 + (rho/2) u^2 + 0.25 u^4 - f u ] dx
         - int_dOmega [ m du/dn + q u ] ds

on Omega=(-1,1)^2, whose Euler-Lagrange equation is (1+Lap)^2 u + rho u +
u^3 = f with the two natural Neumann-type conditions carried by m and q.
The Swift-Hohenberg control parameter is written here as +rho with rho > 0,
the convex regime; the pattern-forming regime is the opposite sign. With
rho > 0 the energy is strictly convex. The load and boundary data are
manufactured from a completed-class
target u*, so the exact minimizer is representable at zero gap and only
approached at any positive gap.

The realization class uses the Matern kernel of smoothness nu=9/2 at length
scale ell=3/sqrt(7), which is in C^8 and therefore supports the fourth
derivatives the energy and the manufactured load require, and which agrees with
exp(-|y|^2/2) through quadratic order at its center. The Matern kernel does not
factorize the way the Gaussian does, so the cosh/sinhc pair formulas have no
analogue; matern.py replaces them with an exact divided-difference identity
that removes the half gap from every denominator. See that module.
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matern
import scipy
from scipy.optimize import minimize
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

FORCINGS = {"manufactured_jet": "(1+Laplacian)^2 u* + rho*u* + (u*)^3, rho=1"}

# --- Manufactured completed-class target -----------------------------------
# Two Matérn kernel/derivative pairs and one singleton, with natural boundary data.
RHO = 1.0   # Swift-Hohenberg control parameter, positive = convex regime


# The quartic term is convex; rho>0 makes the energy strictly convex.
def nonlinear_potential(u):
    return 0.25*u**4


def nonlinear_term(u):
    return u**3


def nonlinear_derivative(u):
    return 3.0*u*u


MANUFACTURED_JET = {
    "pairs": (((-0.32, -0.27), 0.40, 1.2, 1.5),
              ((0.33, 0.27), 1.75, -0.9, 1.1)),
    "background": ((0.00, 0.05), 0.6),
}


def _manufactured_jet_parts(x, y):
    """G and the derivative combinations the Swift-Hohenberg data needs.

    Returns G, grad G, Lap G, grad(Lap G) and Lap^2 G. Each atom uses the
    Matern nu=9/2 radial combinations of ``matern.py``, all of which are
    polynomial multiples of exp(-sqrt(7) r); the even atom is phi and the odd
    atom is psi = -D_d phi, which is exactly the h -> 0 limit of the pair's
    odd column. Writing yhat = x-c, t = d.yhat and r = |yhat|,
      grad phi = c1 yhat,   Lap phi = lap,   grad(Lap phi) = c3 yhat,
      Lap^2 phi = bilap,    psi = -c1 t,     grad psi = -(c1p t yhat + c1 d),
      Lap psi = -c3 t,      grad(Lap psi) = -(c3p t yhat + c3 d),
      Lap^2 psi = -c4 t.
    These were verified against symbolic Cartesian differentiation.
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    shape = x.shape
    G = np.zeros(shape); Gx = np.zeros(shape); Gy = np.zeros(shape)
    L = np.zeros(shape); Lx = np.zeros(shape); Ly = np.zeros(shape); L2 = np.zeros(shape)
    def add(cx, cy, direction, weight_even, weight_odd):
        nonlocal G, Gx, Gy, L, Lx, Ly, L2
        rx, ry = x-cx, y-cy
        r = np.sqrt(rx*rx+ry*ry)
        c1, c3 = matern.radial("c1", r), matern.radial("c3", r)
        if weight_even:
            G = G + weight_even*matern.radial("f0", r)
            Gx = Gx + weight_even*(c1*rx); Gy = Gy + weight_even*(c1*ry)
            L = L + weight_even*matern.radial("lap", r)
            Lx = Lx + weight_even*(c3*rx); Ly = Ly + weight_even*(c3*ry)
            L2 = L2 + weight_even*matern.radial("bilap", r)
        if weight_odd:
            dx, dy = direction
            t = dx*rx+dy*ry
            c1p, c3p = matern.radial("c1p", r), matern.radial("c3p", r)
            G = G + weight_odd*(-c1*t)
            Gx = Gx + weight_odd*(-(c1p*t*rx+c1*dx)); Gy = Gy + weight_odd*(-(c1p*t*ry+c1*dy))
            L = L + weight_odd*(-c3*t)
            Lx = Lx + weight_odd*(-(c3p*t*rx+c3*dx)); Ly = Ly + weight_odd*(-(c3p*t*ry+c3*dy))
            L2 = L2 + weight_odd*(-matern.radial("c4", r)*t)
    for (cx, cy), angle, a, b in MANUFACTURED_JET["pairs"]:
        add(cx, cy, (math.cos(angle), math.sin(angle)), a, b)
    (cx, cy), a0 = MANUFACTURED_JET["background"]
    add(cx, cy, (1.0, 0.0), a0, 0.0)
    return G, Gx, Gy, L, Lx, Ly, L2


def manufactured_jet_solution(x, y):
    """The exact solution u*, a completed-class state with no boundary factor."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    G = _manufactured_jet_parts(x, y)[0]
    return G


def manufactured_jet_gradient(x, y):
    """Analytic gradient of u*."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    _, Gx, Gy, _, _, _, _ = _manufactured_jet_parts(x, y)
    return np.stack((Gx, Gy), axis=-1)


def manufactured_boundary_data(points, normals):
    """The two natural boundary loads that make u* stationary.

    m = (1+Lap)u* is conjugate to du/dn, and q = -d[(1+Lap)u*]/dn to u itself.
    """
    G, Gx, Gy, L, Lx, Ly, _ = _manufactured_jet_parts(points[:, 0], points[:, 1])
    m = G+L
    q = -(np.stack((Gx+Lx, Gy+Ly), axis=-1)*normals).sum(axis=1)
    return m, q


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


def manufactured_jet_load(x, y):
    """The load (1+Laplacian)^2 u* + rho*u* + (u*)^3."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    G, _, _, L, _, _, L2 = _manufactured_jet_parts(x, y)
    return G + 2.0*L + L2 + RHO*G + nonlinear_term(G)



def forcing_values(case: str, amplitude: float, points: np.ndarray) -> np.ndarray:
    x, y = np.asarray(points).T
    if case == "manufactured_jet":
        if float(amplitude) != 1.0:
            raise ValueError("The manufactured jet load is tied to its exact solution; "
                             "scaling it would change the manufactured solution. Use amplitude 1.")
        return manufactured_jet_load(x, y)
    # The recorded envelope-gradient check uses this fixed auxiliary load.
    if case == "mixed_modes":
        positive = np.cos(np.pi*x/2)*np.cos(np.pi*y/2)
        odd = np.sin(np.pi*x)*np.cos(np.pi*y/2)
        quadrupole = np.sin(np.pi*x)*np.sin(np.pi*y)
        return amplitude*(positive+0.5*odd+0.25*quadrupole)
    raise ValueError("Only the paper's manufactured target is supported.")


def quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel())), (weights[:, None]*weights[None, :]).ravel()


def _sinhc(z):
    small = np.abs(z) < 1e-3
    safe = np.where(small, 1.0, z)
    return np.where(small, 1+z*z/6+z**4/120+z**6/5040, np.sinh(z)/safe)


def boundary_fields(points: np.ndarray):
    x, y = points.T
    b = (1-x*x)*(1-y*y)
    gradient = np.column_stack((-2*x*(1-y*y), -2*y*(1-x*x)))
    return b, gradient


def basis_fields(state: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    values, gradients = [], []
    for pair in state["pairs"]:
        r = points-np.asarray(pair["center"])
        direction = np.array([math.cos(pair["angle"]), math.sin(pair["angle"])])
        h = 0.0 if pair["completed"] else pair["half_gap"]
        lo, hi, delta, t = matern.pair_geometry(r, direction, h)
        a, b = matern.pair_combination("f0", lo, hi, delta, t)
        even1, odd1 = matern.pair_combination("c1", lo, hi, delta, t)
        ga = r*even1[:, None]-direction[None, :]*(h*h*odd1)[:, None]
        gb = r*odd1[:, None]-direction[None, :]*even1[:, None]
        values.extend((a, b)); gradients.extend((ga, gb))
    r = points-np.asarray(state["background_center"])
    radius = np.sqrt(np.sum(r*r, axis=1))
    k = matern.radial("f0", radius)
    values.append(k); gradients.append(r*matern.radial("c1", radius)[:, None])
    values, gradients = np.stack(values, axis=1), np.stack(gradients, axis=1)
    laplacians = np.stack(_raw_laplacians(state, points), axis=1)
    return values, gradients, laplacians


def _raw_laplacians(state: dict, points: np.ndarray) -> list:
    """Stable Laplacians of the pair and singleton columns.

    Both forms are smooth at h=0: at zero gap they reduce to the Laplacians of
    phi and of -D_d phi. The odd column is formed from the exact divided
    difference of the radial combination, so no nearly equal kernels are ever
    subtracted and the half gap never appears in a denominator.
    """
    out = []
    for pair in state["pairs"]:
        r = points-np.asarray(pair["center"])
        direction = np.array([math.cos(pair["angle"]), math.sin(pair["angle"])])
        h = 0.0 if pair["completed"] else pair["half_gap"]
        lo, hi, delta, t = matern.pair_geometry(r, direction, h)
        even, odd = matern.pair_combination("lap", lo, hi, delta, t)
        out.extend((even, odd))
    r = points-np.asarray(state["background_center"])
    out.append(matern.radial("lap", np.sqrt(np.sum(r*r, axis=1))))
    return out


def evaluate_state(state: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, gradients, _ = basis_fields(state, points)
    coefficients = np.asarray(state["coefficients"])
    return values @ coefficients, np.einsum("npd,p->nd", gradients, coefficients)


def evaluate_state_laplacian(state: dict, points: np.ndarray) -> np.ndarray:
    _, _, laplacians = basis_fields(state, points)
    return laplacians @ np.asarray(state["coefficients"])


def energy(state: dict, case: str, amplitude: float, order: int) -> dict:
    points, weights = quadrature(order)
    values, gradient, laplacian = basis_fields(state, points)
    c = np.asarray(state["coefficients"])
    u = values @ c
    curvature = (values+laplacian) @ c
    force = forcing_values(case, amplitude, points)
    curvature_term = float(0.5*np.dot(weights, curvature*curvature))
    quadratic_term = float(0.5*RHO*np.dot(weights, u*u))
    potential_term = float(np.dot(weights, nonlinear_potential(u)))
    load_term = float(np.dot(weights, force*u))
    bp, bw, bn = boundary_quadrature(order)
    bv, bg, _ = basis_fields(state, bp)
    m, q = manufactured_boundary_data(bp, bn)
    bdn = np.einsum("npd,nd->np", bg, bn) @ c
    boundary_term = float(np.dot(bw, m*bdn + q*(bv @ c)))
    return {"quadrature_order": order,
            "energy": curvature_term+quadratic_term+potential_term-load_term-boundary_term,
            "curvature_energy": curvature_term, "quadratic_energy": quadratic_term,
            "potential_energy": potential_term, "load": load_term,
            "boundary_work": boundary_term,
            "h1_norm": float(np.sqrt(np.dot(weights, u*u+np.sum(np.einsum("npd,p->nd", gradient, c)**2, axis=1))))}


def initial_state(seed: int, half_gap_range=(0.35, 0.5)) -> dict:
    low, high = map(float, half_gap_range)
    if not (0 < low <= high <= 1.5):
        raise ValueError("Initial half-gaps must satisfy 0 < min <= max <= 1.5")
    rng = np.random.default_rng(seed)
    pairs = []
    for center, angle in (([-0.3, -0.25], 0.35), ([0.3, 0.25], 1.65)):
        pairs.append({"center": (np.asarray(center)+rng.uniform(-0.12, 0.12, 2)).tolist(),
                      "angle": float(angle+rng.uniform(-0.2, 0.2)),
                      "half_gap": float(rng.uniform(low, high)), "completed": False})
    return {"pairs": pairs, "background_center": rng.uniform(-0.12, 0.12, 2).tolist(),
            "coefficients": [0.0]*5}


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
    vector.extend(state["background_center"]); bounds.extend([(-2.0, 2.0)]*2)
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
    state["background_center"] = vector[cursor:cursor+2].tolist()
    return state


def _torch_basis(vector, template, points):
    values, gradients, laplacians, cursor = [], [], [], 0
    for pair in template["pairs"]:
        center = vector[cursor:cursor+2]; angle = vector[cursor+2]; cursor += 3
        direction = torch.stack((torch.cos(angle), torch.sin(angle)))
        if pair["completed"]:
            h = vector.new_tensor(0.0)
        else:
            h = torch.exp(vector[cursor]); cursor += 1
        r = points-center
        lo, hi, delta, t = matern.torch_geometry(torch, r, direction, h)
        a, b = matern.torch_pair(torch, "f0", lo, hi, delta, t)
        even1, odd1 = matern.torch_pair(torch, "c1", lo, hi, delta, t)
        ga = r*even1[:, None]-direction[None, :]*(h*h*odd1)[:, None]
        gb = r*odd1[:, None]-direction[None, :]*even1[:, None]
        la, lb = matern.torch_pair(torch, "lap", lo, hi, delta, t)
        values.extend((a, b)); gradients.extend((ga, gb)); laplacians.extend((la, lb))
    r = points-vector[cursor:cursor+2]
    radius = torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-30))
    k = matern.torch_radial(torch, "f0", radius)
    values.append(k)
    gradients.append(r*matern.torch_radial(torch, "c1", radius)[:, None])
    laplacians.append(matern.torch_radial(torch, "lap", radius))
    values, gradients = torch.stack(values, dim=1), torch.stack(gradients, dim=1)
    laplacians = torch.stack(laplacians, dim=1)
    return values, gradients, laplacians


def fit_coefficients(values, laplacians, weights, force,
                     bvalues, bnormal_derivatives, bweights, bload_m, bload_q):
    """Damped Newton for the strictly convex coefficient subproblem.

    J(c) = 1/2 |(1+Lap)v|^2 + rho/2 |v|^2 + 1/4 |v|^4 - (f,v) - <m, dv/dn> - <q, v>.
    The first three terms are convex and the quadratic part is positive definite,
    so the stationary point is the unique minimizer. Column scaling changes
    coordinates only.
    """
    curvature = values+laplacians                      # (1+Lap) applied to each column
    scales = np.sqrt(np.einsum("n,np,np->p", weights, curvature, curvature)
                     + np.einsum("n,np,np->p", weights, values, values))
    scales = np.where(scales > 0, scales, 1.0)
    curvature = curvature/scales
    columns = values/scales
    bcolumns = bvalues/scales
    bnormals = bnormal_derivatives/scales
    quadratic = (np.einsum("n,np,nq->pq", weights, curvature, curvature)
                 + RHO*np.einsum("n,np,nq->pq", weights, columns, columns))
    load = (columns.T @ (weights*force)
            + bnormals.T @ (bweights*bload_m)
            + bcolumns.T @ (bweights*bload_q))
    condition = float(np.linalg.cond(quadratic))
    if not np.isfinite(condition) or condition > 1e14:
        raise NumericalFailure(f"Swift-Hohenberg quadratic form is ill conditioned: {condition}")
    alpha = np.linalg.solve(quadratic, load)
    backtracks = 0
    reference = max(1.0, float(np.linalg.norm(load)))

    def energy_at(a):
        v = columns @ a
        return float(0.5*a @ quadratic @ a + np.dot(weights, nonlinear_potential(v))
                     - np.dot(load, a))

    for iteration in range(80):
        v = columns @ alpha
        residual = quadratic @ alpha + columns.T @ (weights*nonlinear_term(v)) - load
        stationarity = float(np.linalg.norm(residual))
        if stationarity <= 1e-13*reference:
            return alpha/scales, {"newton_iterations": iteration, "newton_backtracks": backtracks,
                                  "coefficient_stationarity": stationarity,
                                  "relative_coefficient_stationarity": stationarity/reference,
                                  "gradient_basis_condition": condition}
        hessian = quadratic + np.einsum("n,np,nq->pq", weights*nonlinear_derivative(v), columns, columns)
        step = np.linalg.solve(hessian, -residual)
        current = energy_at(alpha); directional = float(np.dot(residual, step)); rate = 1.0
        for _ in range(45):
            candidate = alpha+rate*step
            if energy_at(candidate) <= current+1e-4*rate*directional+5e-15*max(1, abs(current)):
                alpha = candidate; break
            rate *= 0.5; backtracks += 1
        else:
            raise NumericalFailure("Swift-Hohenberg Newton line search failed")
    return alpha/scales, {"newton_iterations": 80, "newton_backtracks": backtracks,
                          "coefficient_stationarity": stationarity,
                          "relative_coefficient_stationarity": stationarity/reference,
                          "gradient_basis_condition": condition}


def transfer_bound(state, pair_index, order=64):
    """H1 size of the state change the transfer makes, measured on Omega.

    This is the H1 norm of the actual difference between the pre- and
    post-transfer states, evaluated by Gauss-Legendre quadrature of order64.
    It is a measured difference, not a certified upper bound.
    """
    before = copy.deepcopy(state)
    after = copy.deepcopy(state)
    after["pairs"][pair_index].update({"completed": True, "half_gap": 0.0})
    points, weights = quadrature(order)
    u0, g0 = evaluate_state(before, points)
    u1, g1 = evaluate_state(after, points)
    difference = u1-u0
    h1 = float(np.sqrt(np.dot(weights, difference*difference
                              + np.sum((g1-g0)**2, axis=1))))
    return {"absolute_h1_bound_float64": h1,
            "measured_h1_state_change": h1,
            "quadrature_order": order,
            "formula": "||u_completed - u_gap||_{H^1(Omega)} by Gauss-Legendre quadrature",
            "certification": "Measured difference in float64; not a validated enclosure or an analytic bound",
            "note": "Stored pair coefficients, center and angle are preserved; only h becomes zero"}


def diagnostics(state, case, amplitude, order):
    result = energy(state, case, amplitude, order)
    raw = []
    for index, pair in enumerate(state["pairs"]):
        a, b = state["coefficients"][2*index:2*index+2]
        if not pair["completed"]:
            h = pair["half_gap"]
            raw.extend((a/2+b/(2*h), a/2-b/(2*h)))
    result.update({
                   "stable_coefficient_norm": float(np.linalg.norm(state["coefficients"])),
                   "remaining_raw_pair_coefficient_norm": float(np.linalg.norm(raw)),
                   "completed_pairs": sum(pair["completed"] for pair in state["pairs"]),
                   "multiplicity_budget": 5})
    return result


class Solver:
    def __init__(self, args, case, amplitude, seed, method, restart_checkpoint=None):
        self.args, self.case, self.amplitude, self.seed, self.method = args, case, amplitude, seed, method
        self.points, self.weights = quadrature(args.quadrature_order)
        (self.boundary_points, self.boundary_weights,
         self.boundary_normals) = boundary_quadrature(args.quadrature_order)
        self.bload_m, self.bload_q = manufactured_boundary_data(
            self.boundary_points, self.boundary_normals)
        self.boundary_points_t = torch.from_numpy(self.boundary_points)
        self.boundary_weights_t = torch.from_numpy(self.boundary_weights)
        self.bload_m_t = torch.from_numpy(self.bload_m)
        self.bload_q_t = torch.from_numpy(self.bload_q)
        self.boundary_normals_t = torch.from_numpy(self.boundary_normals)
        self.force = forcing_values(case, amplitude, self.points)
        self.points_t, self.weights_t = torch.from_numpy(self.points), torch.from_numpy(self.weights)
        self.force_t = torch.from_numpy(self.force)
        self.state = initial_state(seed, (args.initial_half_gap_min, args.initial_half_gap_max))
        self.solves = self.accepted_steps = self.newton_iterations = 0
        self.cache, self.history, self.transfers, self.blocks = {}, [], [], []
        self.restart_checkpoint = copy.deepcopy(restart_checkpoint)
        self.restart_events = []

    def scheduled_restart_ready(self):
        """Restart the finite optimizer at the completion branch's first switch.

        The finite branch reruns the common prefix. Exact geometry agreement
        is checked at the saved accepted checkpoint before clearing only the
        optimizer history; the representation and evaluation cache persist.
        """
        checkpoint = self.restart_checkpoint
        if self.method != "finite_restart" or checkpoint is None or self.restart_events:
            return False
        target_count = checkpoint["solve_count"]
        if self.solves < target_count:
            return False
        actual, _ = pack_geometry(self.state)
        expected, _ = pack_geometry(checkpoint["before_state"])
        if self.solves != target_count or not np.array_equal(actual, expected):
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
        signature = (tuple(pair["completed"] for pair in self.state["pairs"]), np.asarray(vector).tobytes())
        if signature in self.cache:
            return self.cache[signature]
        if self.solves >= self.args.max_solves:
            raise WorkLimit
        self.solves += 1
        geometry = torch.tensor(vector, requires_grad=True)
        values, _, laplacians = _torch_basis(geometry, self.state, self.points_t)
        bvalues, bgradients, _ = _torch_basis(geometry, self.state, self.boundary_points_t)
        bnormal = torch.einsum("npd,nd->np", bgradients, self.boundary_normals_t)
        coefficients, fit = fit_coefficients(
            values.detach().numpy(), laplacians.detach().numpy(), self.weights, self.force,
            bvalues.detach().numpy(), bnormal.detach().numpy(), self.boundary_weights,
            self.bload_m, self.bload_q)
        self.newton_iterations += fit["newton_iterations"]
        c = torch.from_numpy(coefficients)
        u = values @ c
        curvature = (values+laplacians) @ c
        interior = torch.sum(self.weights_t*(0.5*curvature*curvature + 0.5*RHO*u*u
                                             + nonlinear_potential(u) - self.force_t*u))
        boundary = torch.sum(self.boundary_weights_t*(self.bload_m_t*(bnormal @ c)
                                                      + self.bload_q_t*(bvalues @ c)))
        objective = interior-boundary
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
                             **diagnostics(self.state, self.case, self.amplitude, self.args.quadrature_order)})

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
            # A preceding transfer may have changed every coefficient during
            # its nonlinear refit. Recheck the second pair against its new bound.
            if bound["absolute_h1_bound_float64"] > self.args.transfer_tolerance:
                continue
            self.state["pairs"][i].update({"completed": True, "half_gap": 0.0})
            predictor = copy.deepcopy(self.state)
            transfer = {"solve_count": self.solves, "pair_index": i, "before_state": before,
                        "predictor_state": predictor, "bound": bound}
            for order in (80, 96):
                points, weights = quadrature(order)
                u, gu = evaluate_state(before, points)
                v, gv = evaluate_state(predictor, points)
                transfer[f"observed_transfer_h1_order{order}"] = float(np.sqrt(np.dot(weights, (u-v)**2+np.sum((gu-gv)**2, axis=1))))
            self.transfers.append(transfer)
            self.record("transfer", "analytic_bound_predictor")
            vector, _ = pack_geometry(self.state)
            self.state = copy.deepcopy(self.evaluate(vector)["state"])
            transfer["after_state"] = copy.deepcopy(self.state)
            self.record("transfer", "nonlinear_refit")
            changed = True
            print(f"{self.case} A={self.amplitude:g} seed={self.seed}: transfer={i} eval={self.solves} bound={bound['absolute_h1_bound_float64']:.3g}", flush=True)
        return changed

    def run(self):
        started = time.perf_counter()
        initial = copy.deepcopy(self.state)
        status = "work_limit"
        error = None
        try:
            vector, _ = pack_geometry(self.state)
            self.state = copy.deepcopy(self.evaluate(vector)["state"])
            initial = copy.deepcopy(self.state)
            self.record("initial", "nonlinear_refit")
            while self.solves < self.args.max_solves:
                vector, bounds = pack_geometry(self.state)
                before = self.solves
                result = None
                try:
                    result = minimize(self.objective, vector, jac=True, method="L-BFGS-B", bounds=bounds,
                                      callback=self.accepted,
                                      options={"maxiter": self.args.max_solves, "maxfun": self.args.max_solves-self.solves,
                                               "ftol": self.args.ftol, "gtol": self.args.gtol, "maxls": 40, "maxcor": 12})
                    self.state = copy.deepcopy(self.evaluate(result.x)["state"])
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": int(result.status),
                                        "success": bool(result.success), "message": str(result.message),
                                        "iterations": int(result.nit)})
                except TransferReady:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "transfer_ready"})
                except RestartReady:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "matched_finite_restart"})
                except WorkLimit:
                    self.blocks.append({"start_solve": before, "end_solve": self.solves, "status": "work_limit"})
                    break
                # Some completion checks follow an optimizer return rather
                # than its periodic callback; support that same checkpoint.
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
        audits = [energy(self.state, self.case, self.amplitude, order) for order in (80, 96)]
        fitted = list(self.cache.values())
        output = {"case": self.case, "amplitude": self.amplitude, "seed": self.seed, "method": self.method,
                  "initial_state": initial, "final_state": copy.deepcopy(self.state), "history": self.history,
                  "transfers": self.transfers, "optimizer_blocks": self.blocks, "status": status, "error": error,
                  "restart_events": self.restart_events,
                  "restart_checkpoint_from_completion": self.restart_checkpoint,
                  "coefficient_solves": self.solves, "coefficient_newton_iterations": self.newton_iterations,
                  "final_diagnostics": diagnostics(self.state, self.case, self.amplitude, self.args.quadrature_order),
                  "independent_quadrature": audits,
                  "energy_order80_order96_difference": abs(audits[0]["energy"]-audits[1]["energy"]),
                  "maximum_relative_coefficient_stationarity": max((x["relative_coefficient_stationarity"] for x in fitted), default=None),
                  "maximum_gradient_basis_condition": max((x["gradient_basis_condition"] for x in fitted), default=None),
                  "elapsed_seconds": time.perf_counter()-started}
        print(f"{self.case} A={self.amplitude:g} seed={self.seed} {self.method}: {status}; eval={self.solves}; J96={audits[-1]['energy']:.10g}; transfers={len(self.transfers)}", flush=True)
        return output


def validate_gradients(args):
    reports = []
    for completed in (False, True):
        solver = Solver(args, "mixed_modes", 10.0, 1729, "ordinary")
        if completed:
            solver.state["pairs"][0].update({"completed": True, "half_gap": 0.0})
        vector, _ = pack_geometry(solver.state)
        result = solver.evaluate(vector)
        finite_difference = []
        for i in range(len(vector)):
            step = 2e-5*max(1, abs(vector[i]))
            positive, negative = vector.copy(), vector.copy()
            positive[i] += step; negative[i] -= step
            finite_difference.append((solver.evaluate(positive)["objective"]-solver.evaluate(negative)["objective"])/(2*step))
        finite_difference = np.asarray(finite_difference)
        error = float(np.max(np.abs(finite_difference-result["gradient"])))
        relative = error/max(1e-10, float(np.max(np.abs(finite_difference))), float(np.max(np.abs(result["gradient"]))))
        reports.append({"one_completed_pair": completed, "maximum_absolute_error": error,
                        "relative_maximum_error": relative, "analytic": result["gradient"].tolist(),
                        "finite_difference": finite_difference.tolist(), "passed": relative < 2e-5})
    if not all(x["passed"] for x in reports):
        raise NumericalFailure(f"Geometry envelope-gradient validation failed: {reports}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=list(FORCINGS), default=list(FORCINGS))
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-solves", type=int, default=8000)
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument("--block-iterations", type=int, default=40)
    parser.add_argument("--proximity-threshold", type=float, default=0.05)
    parser.add_argument("--transfer-tolerance", type=float, default=3e-2)
    parser.add_argument("--ftol", type=float, default=0.0,
                        help="L-BFGS-B relative-reduction stop; 0 runs until the energy cannot be decreased")
    parser.add_argument("--gtol", type=float, default=0.0,
                        help="L-BFGS-B projected-gradient stop; 0 disables the gradient test")
    parser.add_argument("--initial-half-gap-min", type=float, default=0.02)
    parser.add_argument("--initial-half-gap-max", type=float, default=0.05)
    parser.add_argument("--finite-restart-control", action=argparse.BooleanOptionalAction, default=True,
                        help="Add a finite branch restarted at the completion branch's first switch; all prefixes are rerun")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory exists; preserve prior results and choose a new path")
    if args.max_solves < 30 or args.quadrature_order < 12 or args.block_iterations < 1:
        parser.error("Require max-solves>=30, quadrature-order>=12 and block-iterations>=1")
    args.output.mkdir(parents=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"domain": "(-1,1)^2", "equation": "(1+Lap)^2 u + rho u + u^3 = f, rho=1",
                          "boundary": "natural Neumann-type data m, q manufactured from u*",
                          "objective": "integral (0.5*((1+Lap)u)^2 + 0.5*rho*u^2 + 0.25*u^4 - f*u) - boundary work",
                          "kernel": "Matern nu=9/2, ell=3/sqrt(7); C^8; stable confluent columns by exact divided differences",
                          "forcing_formulas": FORCINGS, "boundary_multiplier": "none; the Neumann-type data are natural",
                          "multiplicity_budget": 5, "pairing": "two supplied pairs plus one singleton",
                          "geometry_bounds": "centers [-2,2]^2, angles unbounded, 0<h<=1.5 without positive floor",
                          "coefficient_fit": "gradient-QR whitened damped Newton; no ridge or rank truncation",
                          "work_accounting": "one unit per distinct nonlinear coefficient fit, including line-search probes and post-transfer refits; Newton iterations also recorded",
                          "optimizer": "L-BFGS-B ftol0 gtol0 maxcor12 maxls40; history reset only on actual completion",
                          "transfer": "measured H1 state change at order-64 quadrature; float64, not a validated enclosure",
                          "study_design": "manufactured completed-class target; all requested branches/statuses retained",
                          "finite_restart_protocol": ("finite geometry and fitted coefficients unchanged at the exact first completion checkpoint; same remaining fit budget; only L-BFGS history reset; if completion never transfers, control is ordinary continuation" if args.finite_restart_control else "not run"),
                          "limitations": "fixed Matern nu=9/2 scale and supplied pairing; no certified objective gate; quadrature agreement is not validated integration"})
    methods = ("ordinary", "completion", "finite_restart") if args.finite_restart_control else ("ordinary", "completion")
    payload = {"schema_version": 1, "configuration": configuration,
               "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "planned_runs": [{"case": case, "amplitude": amplitude, "seed": seed, "method": method}
                                for case in args.cases for amplitude in args.amplitudes
                                for seed in args.seeds for method in methods], "runs": []}
    (args.output/"manifest_before_runs.json").write_text(json.dumps(payload, indent=2)+"\n")
    (args.output/"source_experiment.py").write_bytes(Path(__file__).read_bytes())
    payload["gradient_validation"] = validate_gradients(args)
    (args.output/"gradient_validation.json").write_text(json.dumps(payload["gradient_validation"], indent=2)+"\n")
    for plan in payload["planned_runs"]:
        restart_checkpoint = None
        if plan["method"] == "finite_restart":
            completion_run = next(run for run in payload["runs"] if run["method"] == "completion"
                                  and all(run[key] == plan[key] for key in ("case", "amplitude", "seed")))
            if completion_run["transfers"]:
                restart_checkpoint = completion_run["transfers"][0]
        output = Solver(args, plan["case"], plan["amplitude"], plan["seed"], plan["method"], restart_checkpoint).run()
        payload["runs"].append(output)
        (args.output/"results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        with (args.output/"summary.csv").open("w", newline="") as stream:
            columns = ["case", "amplitude", "seed", "method", "status", "coefficient_solves", "transfers",
                       "energy", "energy_order80_order96_difference", "elapsed_seconds"]
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for run in payload["runs"]:
                writer.writerow({key: len(run["transfers"]) if key == "transfers" else
                                 run.get(key, run["final_diagnostics"].get(key)) for key in columns})


if __name__ == "__main__":
    main()
