"""Whole-line manufactured Helmholtz example at a two-Gaussian collision.

The recorded finite models are a prescribed feasible path and stationary
minimizers within the symmetric, cap-active one-parameter family. No global
optimization over all two-translate models is claimed. All energies use exact
Gaussian integral formulas evaluated with arbitrary precision, not quadrature.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import platform
from pathlib import Path
import time

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/manufactured_1d"


def derivative_polynomial(coefficients, order):
    """Polynomial multiplying exp(-s²/4) after `order` derivatives."""
    p = list(map(Fraction, coefficients))
    for _ in range(order):
        result = [Fraction(0)] * (len(p) + 1)
        for k, value in enumerate(p):
            if k:
                result[k - 1] += k * value
            result[k + 1] -= value / 2
        p = result
    return p


POLYNOMIALS = {
    (sobolev, order): derivative_polynomial(
        [1] if sobolev == 0 else [Fraction(3, 2), 0, Fraction(-1, 4)], order
    )
    for sobolev in (0, 1) for order in range(11)
}


def correlation(order, shift, sobolev=1):
    value = mp.mpf(0)
    for coefficient in reversed(POLYNOMIALS[sobolev, order]):
        value = value * shift + mp.mpf(coefficient.numerator) / coefficient.denominator
    return mp.sqrt(mp.pi) * mp.exp(-shift * shift / 4) * value


def gram(r, c, s, d, sobolev=1):
    return (-1) ** r * correlation(r + s, c - d, sobolev)


def atom_norm_square(atoms, sobolev=1):
    return mp.fsum(w * z * gram(r, c, s, d, sobolev)
                   for w, c, r in atoms for z, d, s in atoms)


def helmholtz(atoms):
    return [(w, c, r) for w, c, r in atoms] + [(-w, c, r + 2) for w, c, r in atoms]


def target_coefficients(epsilon):
    return 1 - mp.mpf(7) * epsilon / 20, -epsilon / 6


def target_atoms(epsilon):
    a, b = target_coefficients(epsilon)
    return [(a, mp.mpf(0), 1), (b, mp.mpf(0), 3)]


def reference_energy(epsilon):
    return epsilon ** 2 * 13 * mp.sqrt(mp.pi) / 320


def pair_metrics(amplitude, radius, epsilon):
    """Symmetric pair with h=b/(sqrt(2) R), hence coefficient norm R."""
    h = amplitude / (mp.sqrt(2) * radius)
    a, b = target_coefficients(epsilon)
    target_square = a * a * gram(1, 0, 1, 0) + 2 * a * b * gram(1, 0, 3, 0) + b * b * gram(3, 0, 3, 0)
    pair_square = (correlation(0, 0) - correlation(0, 2 * h)) / (2 * h * h)
    cross = -(a * correlation(1, h) + b * correlation(3, h)) / h
    error_square = amplitude * amplitude * pair_square - 2 * amplitude * cross + target_square
    return h, error_square / 2 - reference_energy(epsilon), error_square, target_square


def hermite(order, x):
    a, b = mp.mpf(1), x
    if order == 0:
        return a
    for j in range(1, order):
        a, b = b, x * b - j * a
    return b


def kernel_derivative(order, x):
    return (-1) ** order * hermite(order, x) * mp.exp(-x * x / 2)


def pointwise_pair(x, amplitude, h):
    exponential = mp.exp(-(x * x + h * h) / 2)
    sinh_over_h = mp.sinh(x * h) / h
    return (-amplitude * exponential * sinh_over_h,
            amplitude * exponential * (x * sinh_over_h - mp.cosh(x * h)))


def number(value):
    return mp.nstr(value, 55)


def run(output, digits=80):
    if digits < 80:
        raise ValueError("At least 80 decimal digits are required for the recorded range and checks.")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    radii = sorted({2.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0,
                    *np.geomspace(2.0, 100000.0, 25).tolist()})
    with mp.workdps(digits):
        q_atoms = [(mp.mpf(1) / 6, mp.mpf(0), 3), (mp.mpf(7) / 20, mp.mpf(0), 1)]
        q_square = atom_norm_square(q_atoms)
        assert abs(q_square - 13 * mp.sqrt(mp.pi) / 160) < mp.mpf("1e-60")
        orthogonality = [mp.fsum(w * gram(r, c, j, 0) for w, c, r in q_atoms) for j in range(3)]
        assert max(map(abs, orthogonality)) < mp.mpf("1e-60")
        cases = []
        for epsilon_text in ("0", "0.01"):
            epsilon = mp.mpf(epsilon_text)
            target = target_atoms(epsilon)
            target_square = atom_norm_square(target)
            forcing_square = atom_norm_square(helmholtz(target), 0)
            completed_error = epsilon * mp.sqrt(q_square)
            completed_residual = epsilon * mp.sqrt(atom_norm_square(helmholtz(q_atoms), 0))
            cases.append({
                "epsilon": epsilon_text,
                "target_h1_norm": number(mp.sqrt(target_square)),
                "completed_h1_error": number(completed_error),
                "completed_relative_h1_error": number(completed_error / mp.sqrt(target_square)),
                "completed_relative_pde_residual": number(completed_residual / mp.sqrt(forcing_square)),
                "completed_reference_energy": number(reference_energy(epsilon)),
                "completed_coefficient_norm": "1",
                "positive_residual_path_limit_R2": number(epsilon * q_square / 2),
                "boundary_prescribed_path_limit_R4": number(15 * mp.sqrt(mp.pi) / 512),
                "boundary_symmetric_optimized_limit_R4": number(q_square / 8),
            })
            for radius_float in radii:
                radius = mp.mpf(str(radius_float))
                objective = lambda amplitude: pair_metrics(amplitude, radius, epsilon)[1]
                amplitude_opt = mp.findroot(lambda amplitude: mp.diff(objective, amplitude),
                                            (mp.mpf("0.98"), mp.mpf("1.02")),
                                            tol=mp.mpf(10) ** (-(digits - 20)))
                curvature = mp.diff(objective, amplitude_opt, 2)
                assert amplitude_opt > 0 and curvature > 0
                for method, amplitude in (("prescribed", mp.mpf(1)), ("symmetric_optimized", amplitude_opt)):
                    h, excess, error_square, target_square = pair_metrics(amplitude, radius, epsilon)
                    assert excess > 0
                    finite = [(radius / mp.sqrt(2), -h, 0), (-radius / mp.sqrt(2), h, 0)]
                    difference = finite + [(-w, c, r) for w, c, r in target]
                    residual_square = atom_norm_square(helmholtz(difference), 0)
                    assert residual_square > 0
                    direct_square = atom_norm_square(difference)
                    assert abs(direct_square - error_square) < mp.mpf("1e-55")
                    row = {
                        "epsilon": epsilon_text, "method": method,
                        "R": number(radius), "amplitude": number(amplitude), "half_gap": number(h),
                        "ordinary_coefficient_norm": number(radius),
                        "stable_coefficient_norm": number(amplitude),
                        "excess_over_completed_reference": number(excess),
                        "h1_error": number(mp.sqrt(error_square)),
                        "relative_h1_error": number(mp.sqrt(error_square / target_square)),
                        "relative_pde_residual": number(mp.sqrt(residual_square / forcing_square)),
                        "R2_excess": number(radius ** 2 * excess),
                        "R4_excess": number(radius ** 4 * excess),
                        "restricted_stationarity_residual": number(abs(mp.diff(objective, amplitude))) if method == "symmetric_optimized" else "",
                        "restricted_curvature": number(curvature) if method == "symmetric_optimized" else "",
                    }
                    rows.append(row)
                assert objective(amplitude_opt) <= objective(mp.mpf(1))
            # Independent whole-line quadrature, using removable-singularity
            # state formulas rather than correlation polynomials.
            for radius_text in ("2", "100"):
                radius = mp.mpf(radius_text)
                h, _, error_square, _ = pair_metrics(mp.mpf(1), radius, epsilon)
                a, b = target_coefficients(epsilon)
                def integrand(x):
                    value, derivative = pointwise_pair(x, mp.mpf(1), h)
                    exact = a * kernel_derivative(1, x) + b * kernel_derivative(3, x)
                    exact_derivative = a * kernel_derivative(2, x) + b * kernel_derivative(4, x)
                    return (value - exact) ** 2 + (derivative - exact_derivative) ** 2
                quadrature = mp.quad(integrand, [-mp.inf, -6, -2, 0, 2, 6, mp.inf])
                relative_difference = abs(quadrature - error_square) / error_square
                assert relative_difference < mp.mpf("1e-45")
                checks.append({"epsilon": epsilon_text, "R": radius_text,
                               "whole_line_quadrature_relative_difference": number(relative_difference)})

        # Store plot states separately; display grid is not used for energies.
        state_rows = []
        for x_float in np.linspace(-4.5, 4.5, 601):
            x = mp.mpf(str(x_float))
            value, _ = pointwise_pair(x, mp.mpf(1), 1 / (2 * mp.sqrt(2)))
            state_rows.append({"x": float(x_float), "g0": float(kernel_derivative(1, x)),
                               "finite_R2": float(value),
                               "f0": float(kernel_derivative(1, x) - kernel_derivative(3, x))})

        summary = {
            "problem": "-g''+g=f on R; K(x)=exp(-x^2/2); H1 norm squared=int(v^2+v'^2)",
            "target": "g_epsilon=K'-epsilon*(K'''+(21/10)*K')/6",
            "forcing": "f_epsilon=A*K'+(B-A)*K'''-B*K^(5), A=1-7*epsilon/20, B=-epsilon/6",
            "completed_reference": "u0=K', with two-slot coefficient vector (0,1)",
            "finite_path": "v=b*(K(x+h)-K(x-h))/(2*h), h=b/(sqrt(2)*R)",
            "optimization_scope": "For each R, a high-precision stationary minimum in b near 1 of the symmetric cap-active family; no global two-translate minimizer certification.",
            "perturbed_scope": "At epsilon=.01 the completed reference is a stationary completed state. The robust theorem applies for sufficiently small epsilon, but this numerical epsilon is not certified to lie in that theorem's global neighborhood.",
            "arithmetic": {"decimal_digits": digits, "integrals": "analytic Gaussian Gram formulas", "interval_certified": False},
            "q_h1_norm_squared": number(q_square),
            "q_tangent_inner_products": list(map(number, orthogonality)),
            "cases": cases,
            "number_of_records": len(rows),
            "validation": checks,
        }
    for name, records in (("sweep.csv", rows), ("states.csv", state_rows)):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {"python": platform.python_version(), "mpmath": mp.__version__,
                  "numpy": np.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "command": f"python code/experiments/manufactured_1d/run.py --output {output} --digits {digits}",
                  "elapsed_seconds": time.perf_counter() - started}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "records": len(rows), "seconds": provenance["elapsed_seconds"]}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--digits", type=int, default=80)
    args = parser.parse_args()
    run(args.output, args.digits)


if __name__ == "__main__":
    main()
