"""Exact Gaussian Gram formulas used to audit the precision sweep.

Extracted without numerical changes from certified_gaussian_completion.py.
Unrelated optimization and completion routines are deliberately omitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import mpmath as mp
import numpy as np


@dataclass(frozen=True)
class Atom:
    weight: Fraction | float
    center: float
    order: int = 0

    def __post_init__(self):
        object.__setattr__(self, "weight", Fraction(self.weight))
        object.__setattr__(self, "center", float(self.center))
        if not math.isfinite(self.center) or self.order < 0:
            raise ValueError("finite center and nonnegative order required")

    def record(self):
        return {"weight": float(self.weight), "weight_exact_ratio":
                [self.weight.numerator, self.weight.denominator],
                "center": self.center, "order": self.order}


def coalesce(atoms):
    """Combine identical features using exact rational coefficient addition.

    Retaining a Fraction avoids an unbudgeted binary64 rounding error when a
    sum of dyadic coefficients requires more than 53 significant bits.
    """
    grouped = {}
    for a in atoms:
        key = (a.center, a.order)
        grouped[key] = grouped.get(key, Fraction(0)) + a.weight
    return [Atom(w, b, r) for (b, r), w in grouped.items() if w]


def subtract(left, right):
    return coalesce(list(left) + [Atom(-a.weight, a.center, a.order) for a in right])


def operator(atoms):
    """Apply L = 1 - d^2/dx^2 exactly to the stored representation."""
    return coalesce([b for a in atoms for b in
                     (a, Atom(-a.weight, a.center, a.order + 2))])


def _iv_exact(value):
    value = Fraction(value)
    return mp.iv.mpf(value.numerator) / mp.iv.mpf(value.denominator)


def _correlation_poly(order):
    # p_n(t) = exp(t^2/4) d^n/dt^n exp(-t^2/4), exact coefficients.
    p = [Fraction(1)]
    for _ in range(order):
        q = [Fraction(0)] * (len(p) + 1)
        for j, coefficient in enumerate(p):
            if j:
                q[j - 1] += j * coefficient
            q[j + 1] -= coefficient / 2
        p = q
    return p


def _gram_interval(a, b):
    t = _iv_exact(Fraction(a.center) - Fraction(b.center))
    polynomial = mp.iv.mpf(0)
    for c in reversed(_correlation_poly(a.order + b.order)):
        polynomial = polynomial * t + _iv_exact(c)
    return ((-1) ** a.order * polynomial * mp.iv.sqrt(mp.iv.pi)
            * mp.iv.exp(-t * t / 4))


def norm_interval(atoms, sobolev_order=0, digits=80):
    """Enclose sum_{j=0}^m ||D^j u||_L2(R)^2, then its square root.

    All operations, including pi, exp and sqrt, use outward interval rounding.
    Nonnegative intersection uses the a priori fact that a squared norm >= 0.
    """
    mp.iv.dps = digits
    atoms = coalesce(atoms)
    square = mp.iv.mpf(0)
    for derivative in range(sobolev_order + 1):
        for i, a in enumerate(atoms):
            ai = Atom(a.weight, a.center, a.order + derivative)
            for j in range(i, len(atoms)):
                b = atoms[j]
                bj = Atom(b.weight, b.center, b.order + derivative)
                factor = 1 if i == j else 2
                square += (factor * _iv_exact(a.weight * b.weight)
                           * _gram_interval(ai, bj))
    if square.b < 0:
        raise ArithmeticError("negative upper bound for squared norm")
    square = mp.iv.mpf([max(square.a, mp.iv.mpf(0)), square.b])
    return mp.iv.sqrt(square)


def interval_record(value):
    """Outward binary64 endpoints for machine-readable public logs."""
    lo = float(value.a)
    hi = float(value.b)
    return [max(0.0, float(np.nextafter(lo, -np.inf))),
            float(np.nextafter(hi, np.inf))]
