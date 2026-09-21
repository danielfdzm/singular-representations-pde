"""Matern nu=9/2 kernel atoms with exact, stable confluent pair columns.

The Gaussian used elsewhere in this study factorizes,
K(y-hd)=exp(-|y|^2/2-h^2/2)exp(h d.y), which is what turns the even and odd
combinations of a colliding pair into E*cosh(z) and E*t*sinhc(z). A Matern
kernel does not factorize that way, so those closed forms have no analogue and
the odd column [F(rho_-)-F(rho_+)]/(2h) would be a cancelling difference over a
vanishing gap.

This module removes the cancellation exactly rather than by quadrature or by a
series switch. Every radial combination this experiment needs is of the form
F(r) = exp(-gamma r) P(r) with P a polynomial (the 1/r, 1/r^2 and 1/r^3 factors
produced by the radial Laplacian all cancel). For such an F the first divided
difference splits as

    [F(b)-F(a)]/(b-a) = exp(-gamma a) [ expm1(-gamma delta)/delta * P(b)
                                        + (P(b)-P(a))/(b-a) ],

where the polynomial divided difference is the exact finite sum
sum_k p_k sum_{j<k} a^j b^{k-1-j} and delta = b-a is obtained without
cancellation from b^2-a^2 = 4 h t as delta = 4 h t/(a+b). The half gap h then
cancels analytically out of the odd column,

    [F(a)-F(b)]/(2h) = -2 t [F(b)-F(a)]/((a+b)(b-a)) ,

so no quantity is ever divided by h and the h -> 0 limit -D_d F is attained in
floating point, not approached. This is the Matern counterpart of cosh/sinhc.

Smoothness: nu=9/2 puts the kernel in C^8, so the fourth derivatives that the
Swift-Hohenberg load needs (Laplacian^2 of the kernel and of its directional
derivative block) exist and are continuous. The length scale ell=3/sqrt(7) makes
gamma=sqrt(2 nu)/ell=sqrt(7), for which the kernel agrees with exp(-|y|^2/2) through quadratic order and has the same value 1 and the same Laplacian -2 at its center.
"""
from __future__ import annotations

import numpy as np

GAMMA = float(np.sqrt(7.0))
LENGTH_SCALE = 3.0/GAMMA
SMOOTHNESS = 4.5

# Ascending coefficients of P in F(r) = exp(-GAMMA r) P(r), derived symbolically.
#   f0    = phi                     c1    = phi'/r          c1p  = (phi'/r)'/r
#   lap   = Laplacian phi           c3    = (lap)'/r        c3p  = (c3)'/r
#   bilap = Laplacian^2 phi         c4    = (bilap)'/r
POLYNOMIALS = {
    "f0":    (1.0, 2.6457513110645907, 3.0, 1.7638342073763937, 0.4666666666666667),
    "c1":    (-1.0, -2.6457513110645907, -2.8, -1.2346839451634757),
    "c1p":   (1.4, 3.704051835490427, 3.2666666666666666),
    "lap":   (-2.0, -5.291502622129181, -4.2, 1.2346839451634757, 3.2666666666666666),
    "c3":    (5.6, 14.816207341961707, 9.8, -8.642787616144329),
    "c3p":   (-19.6, -51.85672569686598, 22.866666666666667),
    "bilap": (11.2, 29.632414683923415, 0.0, -69.14230092915463, 22.866666666666667),
    "c4":    (-78.4, -207.42690278746392, 274.4, -60.499513313010304),
}


def _horner(coefficients, r):
    out = np.full_like(r, coefficients[-1])
    for coefficient in coefficients[-2::-1]:
        out = out*r+coefficient
    return out


def radial(name: str, r):
    """F(r) = exp(-GAMMA r) P(r) for one of the tabulated combinations."""
    r = np.asarray(r, dtype=float)
    return _horner(POLYNOMIALS[name], r)*np.exp(-GAMMA*r)


def _polynomial_divided_difference(coefficients, a, b):
    """(P(b)-P(a))/(b-a) as the exact finite sum, with no cancellation."""
    out = np.zeros_like(a)
    for k in range(len(coefficients)-1, 0, -1):
        if coefficients[k] == 0.0:
            continue
        inner = np.zeros_like(a)
        for j in range(k):
            inner = inner+a**j*b**(k-1-j)
        out = out+coefficients[k]*inner
    return out


def _expm1_ratio(delta):
    """expm1(-GAMMA delta)/delta, continued to its value -GAMMA at delta=0."""
    small = np.abs(GAMMA*delta) < 1e-5
    safe = np.where(small, 1.0, delta)
    series = -GAMMA+GAMMA*GAMMA*delta/2-GAMMA**3*delta*delta/6
    return np.where(small, series, np.expm1(-GAMMA*np.where(small, 0.0, delta))/safe)


def pair_combination(name: str, a, b, delta, t):
    """Even and odd confluent columns of one radial combination.

    Returns (even, odd) with even = [F(a)+F(b)]/2 and odd = [F(a)-F(b)]/(2h),
    both exact and both finite at h=0, where they reduce to F and to -D_d F.
    """
    coefficients = POLYNOMIALS[name]
    even = 0.5*(radial(name, a)+radial(name, b))
    divided = np.exp(-GAMMA*a)*(_expm1_ratio(delta)*_horner(coefficients, b)
                                + _polynomial_divided_difference(coefficients, a, b))
    total = a+b
    odd = -2.0*t*divided/np.where(total > 1e-300, total, 1.0)
    return even, np.where(total > 1e-300, odd, 0.0)


def pair_geometry(displacement, direction, half_gap):
    """Radii, their cancellation-free difference, and the axial coordinate."""
    displacement = np.asarray(displacement, dtype=float)
    direction = np.asarray(direction, dtype=float)
    t = displacement @ direction
    r2 = np.sum(displacement*displacement, axis=-1)
    h = float(half_gap)
    a = np.sqrt(np.maximum(r2-2*h*t+h*h, 0.0))
    b = np.sqrt(np.maximum(r2+2*h*t+h*h, 0.0))
    total = a+b
    delta = 4*h*t/np.where(total > 1e-300, total, 1.0)
    return a, b, np.where(total > 1e-300, delta, 0.0), t


def torch_pair(torch, name, a, b, delta, t):
    """Torch counterpart of pair_combination, differentiable through the geometry."""
    coefficients = POLYNOMIALS[name]

    def horner(r):
        out = torch.full_like(r, coefficients[-1])
        for coefficient in coefficients[-2::-1]:
            out = out*r+coefficient
        return out

    even = 0.5*(horner(a)*torch.exp(-GAMMA*a)+horner(b)*torch.exp(-GAMMA*b))
    small = torch.abs(GAMMA*delta) < 1e-5
    safe = torch.where(small, torch.ones_like(delta), delta)
    series = -GAMMA+GAMMA*GAMMA*delta/2-GAMMA**3*delta*delta/6
    ratio = torch.where(small, series, torch.expm1(-GAMMA*safe)/safe)
    polydd = torch.zeros_like(a)
    for k in range(len(coefficients)-1, 0, -1):
        if coefficients[k] == 0.0:
            continue
        inner = torch.zeros_like(a)
        for j in range(k):
            inner = inner+a**j*b**(k-1-j)
        polydd = polydd+coefficients[k]*inner
    divided = torch.exp(-GAMMA*a)*(ratio*horner(b)+polydd)
    return even, -2.0*t*divided/torch.clamp(a+b, min=1e-150)


def torch_geometry(torch, displacement, direction, half_gap):
    t = displacement @ direction
    r2 = torch.sum(displacement*displacement, dim=-1)
    h = half_gap
    a = torch.sqrt(torch.clamp(r2-2*h*t+h*h, min=1e-30))
    b = torch.sqrt(torch.clamp(r2+2*h*t+h*h, min=1e-30))
    return a, b, 4*h*t/torch.clamp(a+b, min=1e-150), t


def torch_radial(torch, name: str, r):
    """F(r) = exp(-GAMMA r) P(r) in torch, differentiable through r."""
    coefficients = POLYNOMIALS[name]
    out = torch.full_like(r, coefficients[-1])
    for coefficient in coefficients[-2::-1]:
        out = out*r+coefficient
    return out*torch.exp(-GAMMA*r)
