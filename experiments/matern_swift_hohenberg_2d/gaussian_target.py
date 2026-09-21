"""Unit-scale Gaussian target formula for AppendixD.3 quadrature comparison only."""
import math
import numpy as np
from experiment import MANUFACTURED_JET

def _manufactured_jet_parts(x, y):
    """G and the derivative combinations the Swift-Hohenberg data needs.

    Returns G, grad G, Lap G, grad(Lap G) and Lap^2 G. Each atom uses the exact
    identities for K(y)=exp(-|y|^2/2) and P=(d.y)K, verified symbolically:
      Lap K = (|y|^2-2)K,  Lap^2 K = (|y|^4-8|y|^2+8)K,  grad(Lap K) = y K (4-|y|^2)
      Lap P = t(|y|^2-4)K, Lap^2 P = t(|y|^4-12|y|^2+24)K,
      grad(Lap P) = K[d(|y|^2-4) + 2t y - t(|y|^2-4) y]
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    shape = x.shape
    G = np.zeros(shape); Gx = np.zeros(shape); Gy = np.zeros(shape)
    L = np.zeros(shape); Lx = np.zeros(shape); Ly = np.zeros(shape); L2 = np.zeros(shape)
    def add(cx, cy, direction, weight_even, weight_odd):
        nonlocal G, Gx, Gy, L, Lx, Ly, L2
        rx, ry = x-cx, y-cy
        r2 = rx*rx+ry*ry
        K = np.exp(-0.5*r2)
        if weight_even:
            G = G + weight_even*K
            Gx = Gx + weight_even*(-rx*K); Gy = Gy + weight_even*(-ry*K)
            L = L + weight_even*((r2-2)*K)
            Lx = Lx + weight_even*(rx*K*(4-r2)); Ly = Ly + weight_even*(ry*K*(4-r2))
            L2 = L2 + weight_even*((r2*r2-8*r2+8)*K)
        if weight_odd:
            dx, dy = direction
            t = dx*rx+dy*ry
            G = G + weight_odd*(t*K)
            Gx = Gx + weight_odd*((dx-t*rx)*K); Gy = Gy + weight_odd*((dy-t*ry)*K)
            L = L + weight_odd*(t*(r2-4)*K)
            Lx = Lx + weight_odd*(K*(dx*(r2-4)+2*t*rx-t*(r2-4)*rx))
            Ly = Ly + weight_odd*(K*(dy*(r2-4)+2*t*ry-t*(r2-4)*ry))
            L2 = L2 + weight_odd*(t*(r2*r2-12*r2+24)*K)
    for (cx, cy), angle, a, b in MANUFACTURED_JET["pairs"]:
        add(cx, cy, (math.cos(angle), math.sin(angle)), a, b)
    (cx, cy), a0 = MANUFACTURED_JET["background"]
    add(cx, cy, (1.0, 0.0), a0, 0.0)
    return G, Gx, Gy, L, Lx, Ly, L2
