"""Mixed-precision diagnostics for confluent Gaussian RBF finite differences.

This is the Gaussian/RBF analogue of the previous tanh slope-collision figure.
For order m we use m+1 translated Gaussian atoms

    phi_{m,h}(x) = sum_j alpha_{m,j} K(x - offset_{m,j} h),

where the offsets are centered and alpha_{m,j}=(-1)^{m-j} binom(m,j).  Then
phi_{m,h}/h^m converges to the m-th derivative with respect to the center,
namely He_m(x) K(x), while the coefficients scale like h^{-m}.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path

import numpy as np

from artifact_paths import GAUSSIAN_DATA_DIRECTORY, experiment_output_directory
import tanh_collision_mixed_precision as base


GAUSSIAN_OUTPUT_DIRECTORY = experiment_output_directory("gaussian")
OUT_STEM = "gaussian_rbf_instability_unpenalized_mixed_precision"
OUT_PDF = GAUSSIAN_OUTPUT_DIRECTORY / f"{OUT_STEM}.pdf"
OUT_PNG = GAUSSIAN_OUTPUT_DIRECTORY / f"{OUT_STEM}.png"
OUT_JSON = GAUSSIAN_DATA_DIRECTORY / f"{OUT_STEM}.json"

ORDERS = tuple(range(1, 5))
CACHE_VERSION = 1
SQRT_2PI = float(np.sqrt(2.0 * np.pi))
GAUSS_CONST = 1.0 / SQRT_2PI
AMPLITUDE_INIT_SCALE = 0.1


def gaussian(x: np.ndarray) -> np.ndarray:
    return GAUSS_CONST * np.exp(-0.5 * x * x)


def center_derivative_terms(max_order: int, x: np.ndarray) -> list[np.ndarray]:
    """Return d_c^n K(x-c)|_{c=0}=He_n(x)K(x), n=0,...,max_order."""
    k = gaussian(x)
    terms = [k]
    if max_order == 0:
        return terms
    he_prev = np.ones_like(x)
    he_curr = x.copy()
    terms.append(he_curr * k)
    for n in range(1, max_order):
        he_next = x * he_curr - n * he_prev
        terms.append(he_next * k)
        he_prev, he_curr = he_curr, he_next
    return terms


def target_fn(order: int, x: np.ndarray) -> np.ndarray:
    return center_derivative_terms(order, x)[order]


def feature_values_float64(order: int, x: np.ndarray, h: float) -> np.ndarray:
    offsets, coeffs = base.stencil(order)
    phi = np.zeros_like(x, dtype=np.float64)
    for offset, coeff in zip(offsets, coeffs):
        phi += coeff * gaussian(x - offset * h)
    return phi


def feature_float64(order: int, x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    return amplitude * feature_values_float64(order, x, h)


def feature_float32(order: int, x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    offsets, coeffs = base.stencil(order)
    x32 = x.astype(np.float32)
    h32 = np.float32(h)
    amp32 = np.float32(amplitude)
    c32 = np.float32(GAUSS_CONST)
    u = np.zeros_like(x32, dtype=np.float32)
    for offset, coeff in zip(offsets.astype(np.float32), coeffs.astype(np.float32)):
        z = (x32 - offset * h32).astype(np.float32)
        exponent = (np.float32(-0.5) * z * z).astype(np.float32)
        atom = (np.exp(exponent).astype(np.float32) * c32).astype(np.float32)
        u = (u + (amp32 * coeff) * atom).astype(np.float32)
    return u.astype(np.float64)


def cache_path(order: int, q_min: float, q_max: float, n_profile: int, n_quad: int) -> Path:
    q_min_s = f"{q_min:g}".replace("-", "m").replace(".", "p")
    q_max_s = f"{q_max:g}".replace("-", "m").replace(".", "p")
    return GAUSSIAN_DATA_DIRECTORY / (
        f"gaussian_rbf_confluent_order{order}_profile_cache_v{CACHE_VERSION}_"
        f"q{q_min_s}_{q_max_s}_n{n_profile}_N{n_quad}.npz"
    )


def build_profile_cache(
    order: int,
    q_min: float,
    q_max: float,
    n_profile: int,
    n_quad: int,
    chunk_size: int = 80,
) -> dict[str, np.ndarray | float | int]:
    path = cache_path(order, q_min, q_max, n_profile, n_quad)
    if path.exists():
        data = np.load(path)
        return {
            "order": order,
            "q_grid": data["q_grid"],
            "b_scaled": data["b_scaled"],
            "c_scaled": data["c_scaled"],
            "bq_scaled": data["bq_scaled"],
            "cq_scaled": data["cq_scaled"],
            "b_scaled32": data["b_scaled32"],
            "c_scaled32": data["c_scaled32"],
            "bq_scaled32": data["bq_scaled32"],
            "cq_scaled32": data["cq_scaled32"],
            "target_norm_sq": float(data["target_norm_sq"]),
            "target_norm_sq32": float(data["target_norm_sq32"]),
        }

    x = np.linspace(-10.0, 10.0, n_quad)
    dx = x[1] - x[0]
    trap = np.full_like(x, dx)
    trap[0] = trap[-1] = 0.5 * dx
    terms = center_derivative_terms(base.TAYLOR_ORDER, x)
    derivative_terms = {n: terms[n] for n in range(order, base.TAYLOR_ORDER + 1)}
    target = derivative_terms[order]
    target_norm_sq = float(np.sum(trap * target * target))

    x32 = x.astype(np.float32)
    trap32 = trap.astype(np.float32)
    target32 = target.astype(np.float32)
    derivative_terms32 = {n: values.astype(np.float32) for n, values in derivative_terms.items()}
    target_norm_sq32 = float(np.sum(trap32 * target32 * target32, dtype=np.float32))

    q_grid = np.linspace(q_min, q_max, n_profile)
    h_grid = np.exp(q_grid)
    q_grid32 = q_grid.astype(np.float32)
    h_grid32 = np.exp(q_grid32).astype(np.float32)
    offsets, coeffs = base.stencil(order)
    offsets32 = offsets.astype(np.float32)
    coeffs32 = coeffs.astype(np.float32)

    b_scaled = np.empty_like(q_grid)
    c_scaled = np.empty_like(q_grid)
    bq_scaled = np.empty_like(q_grid)
    cq_scaled = np.empty_like(q_grid)
    b_scaled32 = np.empty_like(q_grid32)
    c_scaled32 = np.empty_like(q_grid32)
    bq_scaled32 = np.empty_like(q_grid32)
    cq_scaled32 = np.empty_like(q_grid32)

    for start in range(0, n_profile, chunk_size):
        stop = min(start + chunk_size, n_profile)
        hs = h_grid[start:stop]
        hs32 = h_grid32[start:stop]
        direct_mask = hs > base.TAYLOR_SWITCH
        taylor_mask = ~direct_mask
        local_indices = np.arange(stop - start)

        if np.any(direct_mask):
            rows = local_indices[direct_mask]
            h = hs[direct_mask][:, None]
            phi = np.zeros((rows.size, x.size), dtype=np.float64)
            dphi_dh = np.zeros_like(phi)
            for offset, coeff in zip(offsets, coeffs):
                z = x[None, :] - offset * h
                atom = gaussian(z)
                phi += coeff * atom
                if offset != 0.0:
                    dphi_dh += coeff * offset * z * atom

            b = np.sum(trap[None, :] * phi * phi, axis=1)
            c = np.sum(trap[None, :] * target[None, :] * phi, axis=1)
            b_h = 2.0 * np.sum(trap[None, :] * phi * dphi_dh, axis=1)
            c_h = np.sum(trap[None, :] * target[None, :] * dphi_dh, axis=1)
            hsel = hs[direct_mask]
            b_scaled[start + rows] = b / (hsel ** (2 * order))
            c_scaled[start + rows] = c / (hsel**order)
            bq_scaled[start + rows] = b_h / (hsel ** (2 * order - 1))
            cq_scaled[start + rows] = c_h / (hsel ** (order - 1))

            h32 = hs32[direct_mask][:, None]
            phi32 = np.zeros((rows.size, x32.size), dtype=np.float32)
            dphi_dh32 = np.zeros_like(phi32)
            for offset, coeff in zip(offsets32, coeffs32):
                z32 = (x32[None, :] - offset * h32).astype(np.float32)
                exponent32 = (np.float32(-0.5) * z32 * z32).astype(np.float32)
                atom32 = (np.exp(exponent32).astype(np.float32) * np.float32(GAUSS_CONST)).astype(np.float32)
                phi32 = (phi32 + coeff * atom32).astype(np.float32)
                if float(offset) != 0.0:
                    dphi_dh32 = (dphi_dh32 + coeff * offset * z32 * atom32).astype(np.float32)

            b32 = np.sum(trap32[None, :] * phi32 * phi32, axis=1, dtype=np.float32)
            c32 = np.sum(trap32[None, :] * target32[None, :] * phi32, axis=1, dtype=np.float32)
            b_h32 = np.float32(2.0) * np.sum(
                trap32[None, :] * phi32 * dphi_dh32,
                axis=1,
                dtype=np.float32,
            )
            c_h32 = np.sum(
                trap32[None, :] * target32[None, :] * dphi_dh32,
                axis=1,
                dtype=np.float32,
            )
            hsel32 = hs32[direct_mask]
            b_scaled32[start + rows] = (b32 / (hsel32 ** np.float32(2 * order))).astype(np.float32)
            c_scaled32[start + rows] = (c32 / (hsel32 ** np.float32(order))).astype(np.float32)
            bq_scaled32[start + rows] = (b_h32 / (hsel32 ** np.float32(2 * order - 1))).astype(np.float32)
            cq_scaled32[start + rows] = (c_h32 / (hsel32 ** np.float32(order - 1))).astype(np.float32)

        if np.any(taylor_mask):
            rows = local_indices[taylor_mask]
            psi, dphi_scaled = base.scaled_taylor_features(order, hs[taylor_mask], derivative_terms, np.float64)
            b_scaled[start + rows] = np.sum(trap[None, :] * psi * psi, axis=1)
            c_scaled[start + rows] = np.sum(trap[None, :] * target[None, :] * psi, axis=1)
            bq_scaled[start + rows] = 2.0 * np.sum(trap[None, :] * psi * dphi_scaled, axis=1)
            cq_scaled[start + rows] = np.sum(trap[None, :] * target[None, :] * dphi_scaled, axis=1)

            psi32, dphi_scaled32 = base.scaled_taylor_features(
                order,
                hs32[taylor_mask],
                derivative_terms32,
                np.float32,
            )
            b_scaled32[start + rows] = np.sum(trap32[None, :] * psi32 * psi32, axis=1, dtype=np.float32)
            c_scaled32[start + rows] = np.sum(trap32[None, :] * target32[None, :] * psi32, axis=1, dtype=np.float32)
            bq_scaled32[start + rows] = np.float32(2.0) * np.sum(
                trap32[None, :] * psi32 * dphi_scaled32,
                axis=1,
                dtype=np.float32,
            )
            cq_scaled32[start + rows] = np.sum(
                trap32[None, :] * target32[None, :] * dphi_scaled32,
                axis=1,
                dtype=np.float32,
            )

    np.savez(
        path,
        q_grid=q_grid,
        b_scaled=b_scaled,
        c_scaled=c_scaled,
        bq_scaled=bq_scaled,
        cq_scaled=cq_scaled,
        b_scaled32=b_scaled32,
        c_scaled32=c_scaled32,
        bq_scaled32=bq_scaled32,
        cq_scaled32=cq_scaled32,
        target_norm_sq=np.array(target_norm_sq),
        target_norm_sq32=np.array(target_norm_sq32, dtype=np.float32),
    )
    return {
        "order": order,
        "q_grid": q_grid,
        "b_scaled": b_scaled,
        "c_scaled": c_scaled,
        "bq_scaled": bq_scaled,
        "cq_scaled": cq_scaled,
        "b_scaled32": b_scaled32,
        "c_scaled32": c_scaled32,
        "bq_scaled32": bq_scaled32,
        "cq_scaled32": cq_scaled32,
        "target_norm_sq": target_norm_sq,
        "target_norm_sq32": target_norm_sq32,
    }


def observed_error_for_precision(
    order: int,
    hist: dict[str, list[float]],
    precision: str,
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    errors = []
    for h, amplitude in zip(hist["separation"], hist["amplitude"]):
        if precision == "float32":
            u = feature_float32(order, x, float(h), float(amplitude))
        else:
            u = feature_float64(order, x, float(h), float(amplitude))
        err = base.relative_l2_error(u, target, weights)
        errors.append(float(err) if np.isfinite(err) else base.NONFINITE_ERROR)
    return np.asarray(errors)


def run_order(order: int, args: argparse.Namespace) -> dict[str, object]:
    cache = build_profile_cache(order, args.q_min, args.q_max, args.n_profile, args.n_quad)
    profile = base.Profile(cache)
    hist32 = base.run_adam_optimizer_compiled(profile, args.float32_iter, args.lr, "float32")
    hist64 = base.run_adam_optimizer_compiled(profile, args.float64_iter, args.lr, "float64")

    x = np.arange(-10.0, 10.0 + 0.5e-3, 1e-3)
    trap = np.full_like(x, 1e-3)
    trap[0] = trap[-1] = 0.5e-3
    target = target_fn(order, x)

    err32_explicit = observed_error_for_precision(order, hist32, "float32", x, target, trap)
    err64_explicit = observed_error_for_precision(order, hist64, "float64", x, target, trap)
    err_eval32 = observed_error_for_precision(order, hist64, "float32", x, target, trap)
    profiled_err32 = base.stable_profiled_error(profile, hist32, "float32")
    profiled_err64 = base.stable_profiled_error(profile, hist64, "float64")
    hist32["adam_relative_error"] = list(hist32["profiled_relative_error"])
    hist64["adam_relative_error"] = list(hist64["profiled_relative_error"])
    hist32["profiled_relative_error"] = profiled_err32.tolist()
    hist64["profiled_relative_error"] = profiled_err64.tolist()
    offsets, coeffs = base.stencil(order)

    return {
        "order": order,
        "neurons": order + 1,
        "offsets": offsets.tolist(),
        "coefficients": coeffs.tolist(),
        "coefficient_norm": float(np.linalg.norm(coeffs)),
        "asymptotic": f"a_h scales like h^(-{order})",
        "hist32": hist32,
        "hist64": hist64,
        "err32_explicit": err32_explicit,
        "err64_explicit": err64_explicit,
        "err_eval32": err_eval32,
        "profiled_err32": profiled_err32,
        "profiled_err64": profiled_err64,
        "summary": {
            "float64_trajectory_evaluation": base.summarize_error_comparison(hist64, err64_explicit, err_eval32),
            "float32": base.summarize(hist32, err32_explicit, profiled_err32),
            "float64": base.summarize(hist64, err64_explicit, profiled_err64),
        },
    }


def serializable_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "order": result["order"],
        "neurons": result["neurons"],
        "offsets": result["offsets"],
        "coefficients": result["coefficients"],
        "coefficient_norm": result["coefficient_norm"],
        "asymptotic": result["asymptotic"],
        "float64_trajectory_evaluation": {
            "source": "float64 Adam trajectory",
            "summary": result["summary"]["float64_trajectory_evaluation"],
            "relative_error_float64": np.asarray(result["err64_explicit"]).tolist(),
            "relative_error_float32": np.asarray(result["err_eval32"]).tolist(),
        },
        "float32": {
            "max_iter": float(result["hist32"]["iteration"][-1]),
            "trace": result["hist32"],
            "realized_relative_error": np.asarray(result["err32_explicit"]).tolist(),
            "stable_profiled_relative_error": np.asarray(result["profiled_err32"]).tolist(),
            "summary": result["summary"]["float32"],
        },
        "float64": {
            "max_iter": float(result["hist64"]["iteration"][-1]),
            "trace": result["hist64"],
            "realized_relative_error": np.asarray(result["err64_explicit"]).tolist(),
            "stable_profiled_relative_error": np.asarray(result["profiled_err64"]).tolist(),
            "summary": result["summary"]["float64"],
        },
    }


def results_from_json(path: Path) -> list[dict[str, object]]:
    """Restore the arrays needed by the plotting code without retraining."""

    payload = json.loads(path.read_text())
    results: list[dict[str, object]] = []
    for record in payload["orders"]:
        results.append(
            {
                "order": int(record["order"]),
                "hist32": record["float32"]["trace"],
                "hist64": record["float64"]["trace"],
                "err32_explicit": np.asarray(
                    record["float32"]["realized_relative_error"], dtype=float
                ),
                "err64_explicit": np.asarray(
                    record["float64"]["realized_relative_error"], dtype=float
                ),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--float32-iter", type=int, default=10_000)
    parser.add_argument("--float64-iter", type=int, default=1_000_000_000)
    parser.add_argument("--lr", type=float, default=5e-1)
    parser.add_argument("--q-min", type=float, default=-16.0)
    parser.add_argument("--q-max", type=float, default=0.5)
    parser.add_argument("--n-profile", type=int, default=6000)
    parser.add_argument("--n-quad", type=int, default=20001)
    parser.add_argument("--orders", type=int, nargs="+", default=list(ORDERS))
    parser.add_argument("--workers", type=int, default=min(4, len(ORDERS)))
    parser.add_argument(
        "--plot-from-json",
        type=Path,
        help="redraw the PDF and PNG from an existing JSON trace without optimization",
    )
    args = parser.parse_args()
    GAUSSIAN_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    GAUSSIAN_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    base.OUT_PDF = OUT_PDF
    base.OUT_PNG = OUT_PNG
    base.AMPLITUDE_INIT_SCALE = AMPLITUDE_INIT_SCALE

    if args.plot_from_json is not None:
        results = results_from_json(args.plot_from_json)
        iter32 = int(max(float(result["hist32"]["iteration"][-1]) for result in results))
        iter64 = int(max(float(result["hist64"]["iteration"][-1]) for result in results))
        base.plot_figure(results, iter32, iter64)
        print(f"Wrote {OUT_PDF} from {args.plot_from_json}")
        print(f"Wrote {OUT_PNG} from {args.plot_from_json}")
        return

    orders = tuple(args.orders)
    invalid_orders = [order for order in orders if order not in ORDERS]
    if invalid_orders:
        raise ValueError(f"orders must be drawn from {ORDERS}; got {invalid_orders}")

    base.load_optimizer_lib()

    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(orders))) as executor:
            results = list(executor.map(lambda order: run_order(order, args), orders))
    else:
        results = [run_order(order, args) for order in orders]
    base.plot_figure(results, args.float32_iter, args.float64_iter)

    payload = {
        "model": (
            "a_h sum_j alpha_j K(x - offset_j h), "
            f"K(x)=(2pi)^(-1/2) exp(-x^2/2), orders={list(orders)}"
        ),
        "target": "d_c^m K(x-c)|_{c=0}=He_m(x)K(x)",
        "penalty": "none",
        "optimizer": "ordinary Adam on amplitude a and separation q=log(h), with beta1=0.9, beta2=0.999, epsilon=1e-8, and componentwise gradient clipping at 1e3; a is initialized at amplitude_init_scale times the least-squares value for q_init and is not profiled during training",
        "learning_rate": args.lr,
        "q_init": base.Q_INIT,
        "amplitude_init_scale": AMPLITUDE_INIT_SCALE,
        "orders": [serializable_result(result) for result in results],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_JSON}")
    for result in results:
        order = result["order"]
        normal64 = result["summary"]["float64"]
        normal32 = result["summary"]["float32"]
        print(
            f"m={order}: best64={normal64['best_error']:.3e}, "
            f"final64={normal64['final_error']:.3e}, "
            f"best32={normal32['best_error']:.3e}, "
            f"final32={normal32['final_error']:.3e}, "
            f"final ||a||64={abs(normal64['final_amplitude']):.3e}"
        )


if __name__ == "__main__":
    main()
