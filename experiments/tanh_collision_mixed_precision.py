"""Mixed-precision diagnostics for confluent tanh finite-difference sequences."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from artifact_paths import (
    NATIVE_DIRECTORY,
    OUTPUT_DIRECTORY as REPOSITORY_OUTPUT_DIRECTORY,
    experiment_output_directory,
)
from tanh_collision_diagnostics import max_iter_latex, relative_l2_error


TANH_OUTPUT_DIRECTORY = experiment_output_directory("tanh")
OUT_STEM = "tanh_instability_unpenalized_mixed_precision"
OUT_PDF = TANH_OUTPUT_DIRECTORY / f"{OUT_STEM}.pdf"
OUT_PNG = TANH_OUTPUT_DIRECTORY / f"{OUT_STEM}.png"
OUT_JSON = TANH_OUTPUT_DIRECTORY / f"{OUT_STEM}.json"
C_SRC = NATIVE_DIRECTORY / "tanh_second_derivative_optimizer.c"
BUILD_DIRECTORY = REPOSITORY_OUTPUT_DIRECTORY / ".build"
SHARED_LIBRARY_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
C_LIB = BUILD_DIRECTORY / f"tanh_second_derivative_optimizer{SHARED_LIBRARY_SUFFIX}"

ORDERS = tuple(range(1, 5))
CACHE_VERSION = 4
TAYLOR_ORDER = 22
TAYLOR_SWITCH = 3e-2
ERROR_FLOOR = 1e-16
PROFILE_PLOT_FLOOR = 1e-8
PANEL_A_MAX_ITER = 10_000.0
PANEL_A_ERROR_CAP = 1e2
NONFINITE_ERROR = 1e300
Q_INIT = -1.0
AMPLITUDE_INIT_SCALE = 0.97


def derivative_polynomials(max_order: int) -> list[np.ndarray]:
    polynomials = [np.array([0.0, 1.0], dtype=np.float64)]
    for _ in range(max_order):
        p = polynomials[-1]
        dp = np.array([j * p[j] for j in range(1, len(p))], dtype=np.float64)
        next_p = np.zeros(len(dp) + 2, dtype=np.float64)
        next_p[: len(dp)] += dp
        next_p[2:] -= dp
        polynomials.append(next_p)
    return polynomials


DERIVATIVE_POLYS = derivative_polynomials(TAYLOR_ORDER)


def eval_poly(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for coeff in coeffs[::-1]:
        out = out * x + coeff
    return out


def stencil(order: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(order + 1, dtype=np.float64) - 0.5 * order
    coeffs = np.array(
        [(-1.0) ** (order - j) * math.comb(order, j) for j in range(order + 1)],
        dtype=np.float64,
    )
    return offsets, coeffs


def moments(order: int) -> np.ndarray:
    offsets, coeffs = stencil(order)
    return np.array([float(np.sum(coeffs * offsets**n)) for n in range(TAYLOR_ORDER + 1)])


def target_fn(order: int, x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return x**order * eval_poly(DERIVATIVE_POLYS[order], t)


def s_derivative_terms(order: int, x: np.ndarray) -> dict[int, np.ndarray]:
    t = np.tanh(x)
    return {
        n: x**n * eval_poly(DERIVATIVE_POLYS[n], t)
        for n in range(order, TAYLOR_ORDER + 1)
    }


def scaled_taylor_features(
    order: int,
    h_values: np.ndarray,
    derivative_terms: dict[int, np.ndarray],
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    np_dtype = np.dtype(dtype)
    scalar = np_dtype.type
    h = h_values.astype(np_dtype)[:, None]
    sample = next(iter(derivative_terms.values())).astype(np_dtype)
    psi = np.zeros((h_values.size, sample.size), dtype=np_dtype)
    psi_h = np.zeros_like(psi)
    moment_values = moments(order)
    for n in range(order, TAYLOR_ORDER + 1):
        moment = moment_values[n]
        if abs(moment) < 1e-12:
            continue
        power = n - order
        coeff = scalar(moment / math.factorial(n))
        term = derivative_terms[n].astype(np_dtype)[None, :]
        if power == 0:
            psi = (psi + coeff * term).astype(np_dtype)
        else:
            h_power = (h ** scalar(power)).astype(np_dtype)
            psi = (psi + coeff * h_power * term).astype(np_dtype)
            psi_h = (
                psi_h
                + coeff * scalar(power) * (h ** scalar(power - 1)).astype(np_dtype) * term
            ).astype(np_dtype)
    dphi_dh_scaled = (scalar(order) * psi + h * psi_h).astype(np_dtype)
    return psi, dphi_dh_scaled


def feature_values_float64(order: int, x: np.ndarray, h: float) -> np.ndarray:
    offsets, coeffs = stencil(order)
    phi = np.zeros_like(x, dtype=np.float64)
    for offset, coeff in zip(offsets, coeffs):
        phi += coeff * np.tanh((1.0 + offset * h) * x)
    return phi


def feature_float64(order: int, x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    return amplitude * feature_values_float64(order, x, h)


def feature_float32(order: int, x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    offsets, coeffs = stencil(order)
    x32 = x.astype(np.float32)
    h32 = np.float32(h)
    amp32 = np.float32(amplitude)
    u = np.zeros_like(x32, dtype=np.float32)
    for offset, coeff in zip(offsets.astype(np.float32), coeffs.astype(np.float32)):
        z = ((np.float32(1.0) + offset * h32) * x32).astype(np.float32)
        atom = np.tanh(z).astype(np.float32)
        u = (u + (amp32 * coeff) * atom).astype(np.float32)
    return u.astype(np.float64)


def record_iters(max_iter: int) -> np.ndarray:
    if max_iter > 2_000_000:
        early = np.arange(0, min(max_iter, 10_000) + 1, 250, dtype=np.int64)
        mid = np.arange(12_500, min(max_iter, 50_000) + 1, 2_500, dtype=np.int64)
        log_tail = np.rint(np.geomspace(55_000, max_iter, 400)).astype(np.int64)
        linear_tail = np.rint(np.linspace(55_000, max_iter, 1200)).astype(np.int64)
        values = np.unique(
            np.concatenate(
                [
                    early,
                    mid,
                    log_tail,
                    linear_tail,
                    np.array([max_iter], dtype=np.int64),
                ]
            )
        )
        return values[values <= max_iter].astype(np.int64)

    values = np.unique(
        np.concatenate(
            [
                np.arange(0, 10_001, 250, dtype=np.int64),
                np.arange(12_500, 50_001, 2_500, dtype=np.int64),
                np.arange(55_000, max_iter + 1, 5_000, dtype=np.int64),
                np.array([max_iter], dtype=np.int64),
            ]
        )
    )
    return values[values <= max_iter].astype(np.int64)


def cache_path(order: int, q_min: float, q_max: float, n_profile: int, n_quad: int) -> Path:
    q_min_s = f"{q_min:g}".replace("-", "m").replace(".", "p")
    q_max_s = f"{q_max:g}".replace("-", "m").replace(".", "p")
    return TANH_OUTPUT_DIRECTORY / (
        f"tanh_confluent_order{order}_profile_cache_v{CACHE_VERSION}_"
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
    derivative_terms = s_derivative_terms(order, x)
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
    offsets, coeffs = stencil(order)
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
        direct_mask = hs > TAYLOR_SWITCH
        taylor_mask = ~direct_mask
        local_indices = np.arange(stop - start)

        if np.any(direct_mask):
            rows = local_indices[direct_mask]
            h = hs[direct_mask][:, None]
            phi = np.zeros((rows.size, x.size), dtype=np.float64)
            dphi_dh = np.zeros_like(phi)
            for offset, coeff in zip(offsets, coeffs):
                z = (1.0 + offset * h) * x[None, :]
                atom = np.tanh(z)
                phi += coeff * atom
                if offset != 0.0:
                    dphi_dh += coeff * offset * x[None, :] * (1.0 - atom * atom)

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
                z32 = ((np.float32(1.0) + offset * h32) * x32[None, :]).astype(np.float32)
                atom32 = np.tanh(z32).astype(np.float32)
                phi32 = (phi32 + coeff * atom32).astype(np.float32)
                if float(offset) != 0.0:
                    sech32 = (np.float32(1.0) - atom32 * atom32).astype(np.float32)
                    dphi_dh32 = (dphi_dh32 + coeff * offset * x32[None, :] * sech32).astype(np.float32)

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
            psi, dphi_scaled = scaled_taylor_features(order, hs[taylor_mask], derivative_terms, np.float64)
            b_scaled[start + rows] = np.sum(trap[None, :] * psi * psi, axis=1)
            c_scaled[start + rows] = np.sum(trap[None, :] * target[None, :] * psi, axis=1)
            bq_scaled[start + rows] = 2.0 * np.sum(trap[None, :] * psi * dphi_scaled, axis=1)
            cq_scaled[start + rows] = np.sum(trap[None, :] * target[None, :] * dphi_scaled, axis=1)

            psi32, dphi_scaled32 = scaled_taylor_features(order, hs32[taylor_mask], derivative_terms32, np.float32)
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


class Profile:
    def __init__(self, cache: dict[str, np.ndarray | float | int]) -> None:
        self.order = int(cache["order"])
        self.q_grid = np.asarray(cache["q_grid"], dtype=np.float64)
        self.q_min = float(self.q_grid[0])
        self.q_max = float(self.q_grid[-1])
        self.dq = float(self.q_grid[1] - self.q_grid[0])
        self.inv_dq = 1.0 / self.dq
        self.b_scaled = np.asarray(cache["b_scaled"], dtype=np.float64)
        self.c_scaled = np.asarray(cache["c_scaled"], dtype=np.float64)
        self.bq_scaled = np.asarray(cache["bq_scaled"], dtype=np.float64)
        self.cq_scaled = np.asarray(cache["cq_scaled"], dtype=np.float64)
        self.target_norm_sq = float(cache["target_norm_sq"])
        self.b_scaled32 = np.asarray(cache["b_scaled32"], dtype=np.float32)
        self.c_scaled32 = np.asarray(cache["c_scaled32"], dtype=np.float32)
        self.bq_scaled32 = np.asarray(cache["bq_scaled32"], dtype=np.float32)
        self.cq_scaled32 = np.asarray(cache["cq_scaled32"], dtype=np.float32)
        self.target_norm_sq32 = np.float32(cache["target_norm_sq32"])
        offsets, coeffs = stencil(self.order)
        self.coeff_norm_sq = float(np.sum(coeffs * coeffs))
        self.offset_sq_sum = float(np.sum(offsets * offsets))


def load_optimizer_lib() -> ctypes.CDLL:
    if not C_LIB.exists() or C_LIB.stat().st_mtime < C_SRC.stat().st_mtime:
        compiler = os.environ.get("CC")
        if compiler is None:
            compiler = next(
                (candidate for name in ("clang", "cc", "gcc") if (candidate := shutil.which(name))),
                None,
            )
        if compiler is None:
            raise RuntimeError("no C compiler found; set CC to a C compiler executable")
        BUILD_DIRECTORY.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                compiler,
                "-O3",
                "-shared",
                "-fPIC",
                str(C_SRC),
                "-o",
                str(C_LIB),
                "-lm",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    lib = ctypes.CDLL(str(C_LIB))
    double_ptr = ctypes.POINTER(ctypes.c_double)
    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int)
    longlong_ptr = ctypes.POINTER(ctypes.c_longlong)
    common_tail = [
        ctypes.c_int,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
    ]
    lib.run_diff_optimizer64.argtypes = [
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_longlong,
        ctypes.c_double,
        ctypes.c_double,
        longlong_ptr,
        *common_tail,
    ]
    lib.run_diff_optimizer64.restype = ctypes.c_int
    lib.run_diff_optimizer32.argtypes = [
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_longlong,
        ctypes.c_float,
        ctypes.c_float,
        longlong_ptr,
        *common_tail,
    ]
    lib.run_diff_optimizer32.restype = ctypes.c_int
    return lib


def run_adam_optimizer_compiled(
    profile: Profile,
    max_iter: int,
    lr: float,
    dtype: str,
) -> dict[str, list[float]]:
    lib = load_optimizer_lib()
    records = record_iters(max_iter)
    n_records = len(records)
    out = {
        "iteration": np.empty(n_records, dtype=np.float64),
        "lambda": np.empty(n_records, dtype=np.float64),
        "profiled_relative_error": np.empty(n_records, dtype=np.float64),
        "separation": np.empty(n_records, dtype=np.float64),
        "amplitude": np.empty(n_records, dtype=np.float64),
        "weight_norm": np.empty(n_records, dtype=np.float64),
        "theta_norm": np.empty(n_records, dtype=np.float64),
        "grad_norm": np.empty(n_records, dtype=np.float64),
        "q": np.empty(n_records, dtype=np.float64),
    }
    record_arr = np.ascontiguousarray(records, dtype=np.int64)

    if dtype == "float64":
        written = lib.run_diff_optimizer64(
            np.ascontiguousarray(profile.b_scaled, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.ascontiguousarray(profile.c_scaled, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.ascontiguousarray(profile.bq_scaled, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            np.ascontiguousarray(profile.cq_scaled, dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(len(profile.q_grid)),
            ctypes.c_int(profile.order),
            ctypes.c_double(profile.coeff_norm_sq),
            ctypes.c_double(profile.offset_sq_sum),
            ctypes.c_double(Q_INIT),
            ctypes.c_double(profile.q_min),
            ctypes.c_double(profile.q_max),
            ctypes.c_double(profile.inv_dq),
            ctypes.c_double(profile.target_norm_sq),
            ctypes.c_longlong(max_iter),
            ctypes.c_double(lr),
            ctypes.c_double(AMPLITUDE_INIT_SCALE),
            record_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            ctypes.c_int(n_records),
            *[out[key].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for key in out],
        )
    elif dtype == "float32":
        q_grid32 = profile.q_grid.astype(np.float32)
        inv_dq32 = np.float32(1.0) / np.float32(q_grid32[1] - q_grid32[0])
        written = lib.run_diff_optimizer32(
            np.ascontiguousarray(profile.b_scaled32, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(profile.c_scaled32, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(profile.bq_scaled32, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(profile.cq_scaled32, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(len(profile.q_grid)),
            ctypes.c_int(profile.order),
            ctypes.c_float(profile.coeff_norm_sq),
            ctypes.c_float(profile.offset_sq_sum),
            ctypes.c_float(Q_INIT),
            ctypes.c_float(q_grid32[0]),
            ctypes.c_float(q_grid32[-1]),
            ctypes.c_float(inv_dq32),
            ctypes.c_float(profile.target_norm_sq32),
            ctypes.c_longlong(max_iter),
            ctypes.c_float(lr),
            ctypes.c_float(AMPLITUDE_INIT_SCALE),
            record_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            ctypes.c_int(n_records),
            *[out[key].ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for key in out],
        )
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    return {key: out[key][:written].tolist() for key in out}


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
        err = relative_l2_error(u, target, weights)
        errors.append(float(err) if np.isfinite(err) else NONFINITE_ERROR)
    return np.asarray(errors)


def stable_profiled_error(profile: Profile, hist: dict[str, list[float]], dtype: str) -> np.ndarray:
    q = np.asarray(hist["q"], dtype=float)
    if dtype == "float32":
        b_grid = profile.b_scaled32.astype(np.float64)
        c_grid = profile.c_scaled32.astype(np.float64)
        target_norm_sq = float(profile.target_norm_sq32)
    elif dtype == "float64":
        b_grid = profile.b_scaled
        c_grid = profile.c_scaled
        target_norm_sq = profile.target_norm_sq
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    b = np.interp(q, profile.q_grid, b_grid)
    c = np.interp(q, profile.q_grid, c_grid)
    ratio = np.zeros_like(q)
    valid = (b > 0.0) & np.isfinite(b) & np.isfinite(c)
    ratio[valid] = (c[valid] * c[valid]) / (b[valid] * target_norm_sq)
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.maximum(np.sqrt(np.maximum(0.0, 1.0 - ratio)), ERROR_FLOOR)


def summarize(hist: dict[str, list[float]], err: np.ndarray, prof: np.ndarray) -> dict[str, float]:
    err_for_min = np.where(np.isfinite(err), err, NONFINITE_ERROR)
    idx = int(np.argmin(err_for_min))
    prof_idx = int(np.argmin(prof))
    return {
        "best_iteration": float(hist["iteration"][idx]),
        "best_error": float(err_for_min[idx]),
        "best_profiled_iteration": float(hist["iteration"][prof_idx]),
        "best_profiled_error": float(prof[prof_idx]),
        "final_iteration": float(hist["iteration"][-1]),
        "final_error": float(err[-1]) if np.isfinite(err[-1]) else NONFINITE_ERROR,
        "final_profiled_error": float(prof[-1]),
        "final_separation": float(hist["separation"][-1]),
        "final_weight_norm": float(hist["weight_norm"][-1]),
        "final_amplitude": float(hist["amplitude"][-1]),
    }


def summarize_error_comparison(
    hist: dict[str, list[float]],
    err64: np.ndarray,
    err32: np.ndarray,
) -> dict[str, float]:
    err64_for_min = np.where(np.isfinite(err64), err64, NONFINITE_ERROR)
    err32_for_min = np.where(np.isfinite(err32), err32, NONFINITE_ERROR)
    idx64 = int(np.argmin(err64_for_min))
    idx32 = int(np.argmin(err32_for_min))
    return {
        "final_iteration": float(hist["iteration"][-1]),
        "final_error_float64": float(err64[-1]) if np.isfinite(err64[-1]) else NONFINITE_ERROR,
        "final_error_float32": float(err32[-1]) if np.isfinite(err32[-1]) else NONFINITE_ERROR,
        "best_iteration_float64": float(hist["iteration"][idx64]),
        "best_error_float64": float(err64_for_min[idx64]),
        "best_iteration_float32": float(hist["iteration"][idx32]),
        "best_error_float32": float(err32_for_min[idx32]),
    }


def run_order(order: int, args: argparse.Namespace) -> dict[str, object]:
    cache = build_profile_cache(order, args.q_min, args.q_max, args.n_profile, args.n_quad)
    profile = Profile(cache)
    hist32 = run_adam_optimizer_compiled(profile, args.float32_iter, args.lr, "float32")
    hist64 = run_adam_optimizer_compiled(profile, args.float64_iter, args.lr, "float64")

    x = np.arange(-10.0, 10.0 + 0.5e-3, 1e-3)
    trap = np.full_like(x, 1e-3)
    trap[0] = trap[-1] = 0.5e-3
    target = target_fn(order, x)

    err32_explicit = observed_error_for_precision(order, hist32, "float32", x, target, trap)
    err64_explicit = observed_error_for_precision(order, hist64, "float64", x, target, trap)
    err_eval32 = observed_error_for_precision(order, hist64, "float32", x, target, trap)
    profiled_err32 = stable_profiled_error(profile, hist32, "float32")
    profiled_err64 = stable_profiled_error(profile, hist64, "float64")
    hist32["adam_relative_error"] = list(hist32["profiled_relative_error"])
    hist64["adam_relative_error"] = list(hist64["profiled_relative_error"])
    hist32["profiled_relative_error"] = profiled_err32.tolist()
    hist64["profiled_relative_error"] = profiled_err64.tolist()
    offsets, coeffs = stencil(order)

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
            "float64_trajectory_evaluation": summarize_error_comparison(hist64, err64_explicit, err_eval32),
            "float32": summarize(hist32, err32_explicit, profiled_err32),
            "float64": summarize(hist64, err64_explicit, profiled_err64),
        },
    }


def plot_error_overlay_panel(ax: plt.Axes, result: dict[str, object], show_legend: bool, blue: str, red: str) -> None:
    hist32 = result["hist32"]
    hist64 = result["hist64"]
    iters32 = np.asarray(hist32["iteration"], dtype=float)
    iters64 = np.asarray(hist64["iteration"], dtype=float)
    err32 = np.asarray(result["err32_explicit"], dtype=float)
    err64 = np.asarray(result["err64_explicit"], dtype=float)
    err32_plot = np.clip(np.maximum(err32, ERROR_FLOOR), ERROR_FLOOR, PANEL_A_ERROR_CAP)
    err64_plot = np.clip(np.maximum(err64, ERROR_FLOOR), ERROR_FLOOR, PANEL_A_ERROR_CAP)
    markevery64 = max(1, len(iters64) // 12)
    markevery32 = max(1, len(iters32) // 12)
    ax.semilogy(
        iters64,
        err64_plot,
        color=blue,
        lw=1.85,
        ls="-",
        marker="o",
        ms=3.0,
        markevery=markevery64,
        label="double-precision error",
    )
    ax.semilogy(
        iters32,
        err32_plot,
        color=red,
        lw=1.85,
        ls="--",
        marker="s",
        ms=2.9,
        markevery=markevery32,
        label="single-precision error",
    )
    ax.set_xlim(0.0, max(float(np.max(iters32)), float(np.max(iters64))))
    ax.set_ylim(1e-8, PANEL_A_ERROR_CAP)
    if show_legend:
        ax.legend(frameon=False, loc="upper left", handlelength=1.8)


def plot_profile_panel(
    ax: plt.Axes,
    hist: dict[str, list[float]],
    error: np.ndarray,
    show_legend: bool,
    error_color: str,
    norm_color: str,
    error_label: str,
) -> None:
    iters = np.asarray(hist["iteration"], dtype=float)
    amplitude_norm = np.abs(np.asarray(hist["amplitude"], dtype=float))
    error_plot = np.clip(np.maximum(np.asarray(error, dtype=float), ERROR_FLOOR), ERROR_FLOOR, PANEL_A_ERROR_CAP)
    is_single = "single" in error_label
    err_line, = ax.semilogy(
        iters,
        error_plot,
        color=error_color,
        lw=1.85,
        ls="--" if is_single else "-",
        marker="s" if is_single else "o",
        ms=2.9 if is_single else 3.0,
        markevery=max(1, len(iters) // 12),
        label=error_label,
    )
    ax2 = ax.twinx()
    norm_line, = ax2.plot(
        iters,
        amplitude_norm,
        color=norm_color,
        lw=1.7,
        ls=":",
        marker="x",
        ms=3.0,
        markevery=max(1, len(iters) // 12),
        label=r"$\|a\|$",
    )
    ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax2.tick_params(axis="y", labelcolor=norm_color, labelsize=8.0)
    ax2.set_ylabel(r"$\|a\|$", color=norm_color, labelpad=4)
    ax.set_ylim(1e-8, PANEL_A_ERROR_CAP)
    if show_legend:
        ax.legend(
            [err_line, norm_line],
            [err_line.get_label(), norm_line.get_label()],
            frameon=False,
            loc="lower right",
            handlelength=1.8,
        )


def plot_figure(results: list[dict[str, object]], iter32: int, iter64: int) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "xtick.major.size": 4.2,
            "ytick.major.size": 4.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.7,
            "legend.fontsize": 8.0,
        }
    )
    blue = "#0000FF"
    red = "#FF0000"
    norm_color = "#555555"
    if len(results) == 1:
        fig, vertical_axes = plt.subplots(3, 1, figsize=(4.685, 6.65))
        axes = np.asarray(vertical_axes, dtype=object).reshape(1, 3)
        fig.subplots_adjust(left=0.17, right=0.82, bottom=0.075, top=0.96, hspace=0.46)
    else:
        fig_height = 2.35 * len(results)
        fig, axes = plt.subplots(len(results), 3, figsize=(12.4, fig_height))
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes[None, :]
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.085, top=0.94, wspace=0.55, hspace=0.5)

    axes[0, 0].set_title(rf"realized error, ${max_iter_latex(max(iter32, iter64))}$ iterations")
    axes[0, 1].set_title(rf"single-precision Adam, ${max_iter_latex(iter32)}$ iterations")
    axes[0, 2].set_title(rf"double-precision Adam, ${max_iter_latex(iter64)}$ iterations")

    for row, result in enumerate(results):
        plot_error_overlay_panel(axes[row, 0], result, row == 0, blue, red)
        plot_profile_panel(
            axes[row, 1],
            result["hist32"],
            result["err32_explicit"],
            False,
            red,
            norm_color,
            r"single-precision error",
        )
        plot_profile_panel(
            axes[row, 2],
            result["hist64"],
            result["err64_explicit"],
            False,
            blue,
            norm_color,
            r"double-precision error",
        )
        for col, color in enumerate(("black", red, blue)):
            axes[row, col].set_ylabel(r"realized rel. $L^2$ error", color=color, labelpad=5)
            axes[row, col].tick_params(axis="y", labelcolor=color)
        for col in range(3):
            axes[row, col].grid(False)
            if row < len(results) - 1:
                axes[row, col].tick_params(labelbottom=False)
            else:
                axes[row, col].set_xlabel("iteration")

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=240)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--float32-iter", type=int, default=10_000)
    parser.add_argument("--float64-iter", type=int, default=1_000_000_000)
    parser.add_argument("--lr", type=float, default=5e-1)
    parser.add_argument("--q-min", type=float, default=-16.0)
    parser.add_argument("--q-max", type=float, default=0.5)
    parser.add_argument("--n-profile", type=int, default=6000)
    parser.add_argument("--n-quad", type=int, default=20001)
    parser.add_argument("--workers", type=int, default=min(4, len(ORDERS)))
    args = parser.parse_args()
    TANH_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    load_optimizer_lib()
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(lambda order: run_order(order, args), ORDERS))
    else:
        results = [run_order(order, args) for order in ORDERS]
    plot_figure(results, args.float32_iter, args.float64_iter)

    payload = {
        "model": "a_h sum_j alpha_j tanh((1+(j-m/2)h)x), m=1,...,4",
        "target": "d_s^m tanh(s*x)|_{s=1}; for m=1 this is x*sech^2(x)",
        "penalty": "none",
        "optimizer": "ordinary Adam on amplitude a and separation q=log(h), with beta1=0.9, beta2=0.999, epsilon=1e-8, and componentwise gradient clipping at 1e3; a is initialized at the least-squares value for q_init and is not profiled during training",
        "learning_rate": args.lr,
        "q_init": Q_INIT,
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
