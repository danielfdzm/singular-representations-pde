"""Global coefficient-capped two-atom search for the whole-line example.

For the target g_eps of run.py and a cap R, this computes the quantity of
Theorem 3.3,

    E_g(R) = inf { J_g(w1 K(.-z1) + w2 K(.-z2)) - J_g(u_g) : w, z in R^2, |w|_2 <= R },

without restricting the positions or weights. For fixed centres the weight
problem is a two-dimensional trust-region subproblem, solved exactly from the
eigenvectors of the 2x2 H1 Gram matrix and its secular equation. The search
then runs over the centre c = (z1+z2)/2 and half-gap d = |z2-z1|/2. Since g_eps
is odd, the excess is even in c and c >= 0 suffices. A coarse grid in
(c, log d), with a fine band near the cap-active gap d0 = 1/(sqrt(2) R), is
followed by 80-digit golden-section refinement of the best candidates of
three regions: the whole grid, off-centre configurations (c >= 0.1) and
separated ones (d >= 0.25). A search in the completed one-block family checks
that u_g = K' is its minimizer. The grid uses 24 + 4 ceil(log10 R) digits,
since |w|^2 ~ R^2 cancels down to excesses of order R^-4. This is a numerical
global search with analytic Gram entries, not an interval certificate.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import platform
from pathlib import Path
import time

import mpmath as mp
import numpy as np

import run as M

ROOT = Path(__file__).resolve().parents[2]
EPSILONS = ("0", "0.001", "0.01")
GOLDEN = (mp.sqrt(5)-1)/2


_TARGET_SQUARES = {}


def target_square(epsilon):
    key = (str(epsilon), mp.mp.dps)
    if key not in _TARGET_SQUARES:
        a, b = M.target_coefficients(epsilon)
        _TARGET_SQUARES[key] = a*a*M.gram(1, 0, 1, 0) + 2*a*b*M.gram(1, 0, 3, 0) + b*b*M.gram(3, 0, 3, 0)
    return _TARGET_SQUARES[key]


def capped_excess(epsilon, radius, centre, half_gap):
    """Exact minimum over |w| <= R of J_g - J_g(u_g) for atoms at centre -+ half_gap."""
    a, b = M.target_coefficients(epsilon)
    z1, z2 = centre-half_gap, centre+half_gap
    diagonal, off = M.correlation(0, 0), M.correlation(0, z1-z2)
    load = [a*M.gram(0, z, 1, 0) + b*M.gram(0, z, 3, 0) for z in (z1, z2)]
    # The Gram matrix [[diagonal, off], [off, diagonal]] has eigenvectors (1, +-1)/sqrt(2).
    eigen = (diagonal+off, diagonal-off)
    beta = ((load[0]+load[1])/mp.sqrt(2), (load[0]-load[1])/mp.sqrt(2))

    def norm_square(mu):
        return sum((beta[i]/(eigen[i]+mu))**2 for i in range(2))

    mu = mp.mpf(0)
    if min(eigen) <= 0 or norm_square(mu) > radius*radius:
        # Newton on 1/R - 1/|w(mu)|, which is concave and increasing for mu > -min(eigen).
        mu = max(mp.mpf(0), -min(eigen))*(1+mp.mpf(10)**-10)
        for _ in range(300):
            square = norm_square(mu); norm = mp.sqrt(square)
            derivative = -2*sum(beta[i]**2/(eigen[i]+mu)**3 for i in range(2))
            update = mu - (1/radius-1/norm)/(derivative/(2*square*norm))
            if update <= max(mp.mpf(0), -min(eigen)):
                update = (mu+max(mp.mpf(0), -min(eigen)))/2
            if abs(update-mu) <= mp.mpf(10)**(-(mp.mp.dps-10))*max(1, abs(mu)):
                mu = update
                break
            mu = update
    y = [beta[i]/(eigen[i]+mu) for i in range(2)]
    value = sum(eigen[i]*y[i]*y[i]/2 - beta[i]*y[i] for i in range(2))
    weights = ((y[0]+y[1])/mp.sqrt(2), (y[0]-y[1])/mp.sqrt(2))
    excess = value + target_square(epsilon)/2 - M.reference_energy(epsilon)
    return excess, weights, mu


def golden(function, low, high, iterations=160):
    """Golden-section minimization on [low, high]."""
    x1, x2 = high-GOLDEN*(high-low), low+GOLDEN*(high-low)
    f1, f2 = function(x1), function(x2)
    for _ in range(iterations):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high-GOLDEN*(high-low); f1 = function(x1)
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low+GOLDEN*(high-low); f2 = function(x2)
    return (x1, f1) if f1 <= f2 else (x2, f2)


def coarse_digits(radius):
    return int(24 + 4*mp.ceil(mp.log10(radius)))


def half_gap_grid(radius):
    d0 = 1/(mp.sqrt(2)*radius)
    coarse = [d0*mp.mpf(10)**(mp.mpf(k)/10) for k in range(-20, 91) if d0*mp.mpf(10)**(mp.mpf(k)/10) <= 3]
    fine = [d0*(1+mp.mpf(j)/400) for j in range(0, 121)]
    return sorted(set(coarse+fine))


def bracket(grid, half_gap):
    """Neighbouring grid gaps around a grid point."""
    index = grid.index(half_gap)
    return grid[max(index-1, 0)], grid[min(index+1, len(grid)-1)]


def refine(epsilon, radius, centre, low, high, centre_step):
    """Alternating golden-section refinement in d on [low, high] and in c."""
    best_centre = centre
    for _ in range(4):
        best_gap, value = golden(lambda d: capped_excess(epsilon, radius, best_centre, d)[0], low, high)
        if centre_step == 0:
            break
        c_low, c_high = max(mp.mpf(0), best_centre-centre_step), best_centre+centre_step
        best_centre, value = golden(lambda c: capped_excess(epsilon, radius, c, best_gap)[0], c_low, c_high, 120)
        centre_step /= 4
        low, high = best_gap*(1-mp.mpf(1)/200), best_gap*(1+mp.mpf(1)/200)
    excess, weights, mu = capped_excess(epsilon, radius, best_centre, best_gap)
    return {"centre": best_centre, "half_gap": best_gap, "excess": excess, "weights": weights, "multiplier": mu}


def search(task):
    epsilon_text, radius_text, digits = task
    started = time.perf_counter()
    epsilon, radius = mp.mpf(epsilon_text), mp.mpf(radius_text)
    grid = half_gap_grid(radius)
    centres = [mp.mpf(i)/10 for i in range(0, 21)]
    # |w|^2 ~ R^2 cancels down to excesses as small as R^-4, so the grid
    # precision grows with log10(R) to keep about eight significant digits.
    with mp.workdps(coarse_digits(radius)):
        values = {(i, j): capped_excess(epsilon, radius, c, d)[0]
                  for i, c in enumerate(centres) for j, d in enumerate(grid)}
    regions = {"global": lambda i, j: True,
               "off_centre": lambda i, j: centres[i] >= mp.mpf("0.1"),
               "separated": lambda i, j: grid[j] >= mp.mpf("0.25")}
    result = {"epsilon": epsilon_text, "R": radius_text, "grid_points": len(values)}
    with mp.workdps(digits):
        for name, keep in regions.items():
            i, j = min((key for key in values if keep(*key)), key=values.get)
            step = mp.mpf(0) if name == "global" and i == 0 else mp.mpf("0.05")
            result[name] = refine(epsilon, radius, centres[i], *bracket(grid, grid[j]), step)
        # The target is odd: at c = 0 the capped weights are antisymmetric.
        # Off-centre probes, with the gap reoptimized, check that c = 0 is a
        # strict local minimum.
        probes, gap = {}, result["global"]["half_gap"]
        for offset in ("1e-4", "1e-3", "1e-2"):
            shifted = refine(epsilon, radius, mp.mpf(offset), gap*(1-mp.mpf(1)/100), gap*(1+mp.mpf(1)/100), 0)
            probes[offset] = shifted["excess"]-result["global"]["excess"]
        result["centre_probes"] = probes
    result["seconds"] = time.perf_counter()-started
    return result


def completed_family_check(epsilon_text, digits):
    """Least squares in (a, b) for aK(.-c)+bK'(.-c), scanned and refined in c."""
    epsilon = mp.mpf(epsilon_text)
    target_a, target_b = M.target_coefficients(epsilon)

    def best(c):
        G = mp.matrix([[M.gram(r, c, s, c) for s in (0, 1)] for r in (0, 1)])
        load = mp.matrix([target_a*M.gram(r, c, 1, 0)+target_b*M.gram(r, c, 3, 0) for r in (0, 1)])
        coefficients = mp.lu_solve(G, load)
        value = (coefficients.T*G*coefficients)[0]/2 - (coefficients.T*load)[0] + target_square(epsilon)/2
        return value-M.reference_energy(epsilon), coefficients

    with mp.workdps(digits):
        scan = [(best(mp.mpf(i)/50)[0], mp.mpf(i)/50) for i in range(0, 151)]
        value, centre = min(scan)
        if centre > 0:
            centre, value = golden(lambda c: best(c)[0], max(mp.mpf(0), centre-mp.mpf(1)/50), centre+mp.mpf(1)/50)
        _, coefficients = best(centre)
        return {"epsilon": epsilon_text, "minimizing_centre": M.number(centre),
                "coefficients": [M.number(coefficients[0]), M.number(coefficients[1])],
                "excess_over_u_g": M.number(value),
                "smallest_excess_away_from_centre": M.number(min(v for v, c in scan if c >= mp.mpf("0.1")))}


def run(output, digits, jobs):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    radii = sorted({2.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, *np.geomspace(2.0, 100000.0, 25).tolist()})
    tasks = [(epsilon, repr(radius), digits) for epsilon in EPSILONS for radius in radii]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(search, tasks))
    rows, constants = [], {}
    with mp.workdps(digits):
        q_square = 13*mp.sqrt(mp.pi)/160
        for epsilon in EPSILONS:
            constants[epsilon] = (mp.mpf(epsilon)*q_square/2 if epsilon != "0" else q_square/8)
        for result in results:
            radius = mp.mpf(result["R"]); best = result["global"]
            epsilon = result["epsilon"]
            scale = radius**2 if epsilon != "0" else radius**4
            weight_norm = mp.sqrt(best["weights"][0]**2+best["weights"][1]**2)
            symmetric = min(M.pair_metrics(amplitude, radius, mp.mpf(epsilon))[1]
                            for amplitude in [mp.sqrt(2)*radius*best["half_gap"]])
            rows.append({
                "epsilon": epsilon, "R": M.number(radius),
                "global_excess": M.number(best["excess"]),
                "scaled_global_excess": M.number(scale*best["excess"]),
                "scaled_over_limit": M.number(scale*best["excess"]/constants[epsilon]),
                "scaling": "R^2" if epsilon != "0" else "R^4",
                "centre": M.number(best["centre"]), "half_gap": M.number(best["half_gap"]),
                "half_gap_over_cap_gap": M.number(best["half_gap"]*mp.sqrt(2)*radius),
                "w1": M.number(best["weights"][0]), "w2": M.number(best["weights"][1]),
                "weight_norm_over_R": M.number(weight_norm/radius),
                "symmetric_family_excess_at_same_gap": M.number(symmetric),
                "relative_difference_to_symmetric_family": M.number(abs(best["excess"]-symmetric)/best["excess"]),
                "best_off_centre_excess": M.number(result["off_centre"]["excess"]),
                "best_off_centre_centre": M.number(result["off_centre"]["centre"]),
                "best_separated_excess": M.number(result["separated"]["excess"]),
                "best_separated_half_gap": M.number(result["separated"]["half_gap"]),
                "minimum_centre_probe_increase": M.number(min(result["centre_probes"].values())),
                "grid_points": result["grid_points"]})
        checks = [completed_family_check(epsilon, digits) for epsilon in EPSILONS]
    rows.sort(key=lambda row: (EPSILONS.index(row["epsilon"]), float(row["R"])))
    with (output/"global_search.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "quantity": "E_g(R) = inf over all two-atom networks with |w|_2 <= R of J_g - J_g(u_g), u_g = K'",
        "method": __doc__.split("\n\n")[2].strip(),
        "limits": {"epsilon>0": "R^2 E_g(R) -> C_g = epsilon*||q||^2/2 (Theorem 3.3, b_g = 1, l_g = epsilon*||q||^2)",
                   "epsilon=0": "g in C_2, so Theorem 3.3 does not apply; the symmetric branch gives R^4 E -> ||q||^2/8"},
        "limit_constants": {epsilon: M.number(value) for epsilon, value in constants.items()},
        "completed_family_checks": checks,
        "largest_cap_scaled_over_limit": {epsilon: next(row["scaled_over_limit"] for row in reversed(rows) if row["epsilon"] == epsilon)
                                          for epsilon in EPSILONS},
        "maximum_relative_difference_to_symmetric_family": M.number(max(mp.mpf(row["relative_difference_to_symmetric_family"]) for row in rows)),
        "maximum_abs_centre": M.number(max(abs(mp.mpf(row["centre"])) for row in rows)),
        "minimum_weight_norm_over_R": M.number(min(mp.mpf(row["weight_norm_over_R"]) for row in rows)),
        "minimum_positive_centre_probe_increase": M.number(min(mp.mpf(row["minimum_centre_probe_increase"]) for row in rows)),
        "arithmetic": {"coarse_grid_digits": "24 + 4*ceil(log10(R))", "refinement_digits": digits, "interval_certified": False},
        "number_of_records": len(rows)}
    (output/"global_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    provenance = {"python": platform.python_version(), "mpmath": mp.__version__, "numpy": np.__version__,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "run_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
                  "command": f"python experiments/manufactured_1d/global_search.py --output {output} --digits {digits}",
                  "elapsed_seconds": time.perf_counter()-started}
    (output/"global_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({key: summary[key] for key in ("largest_cap_scaled_over_limit", "maximum_relative_difference_to_symmetric_family",
                                                    "maximum_abs_centre", "minimum_weight_norm_over_R",
                                                    "minimum_positive_centre_probe_increase")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/manufactured_1d")
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.digits, args.jobs)


if __name__ == "__main__":
    main()
