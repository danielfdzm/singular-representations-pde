"""Diagnostics for the tanh slope-collision sequence.

The profiled model is

    u_h(x) = a [tanh((1+h)x) - tanh((1-h)x)]

which converges to x sech^2(x) as h -> 0 with a ~ (2h)^{-1}.  This is the
tanh analogue of the two-feature Gaussian collapse diagnostic: the represented
state converges while the two coefficients grow and the feature slopes collide.
"""

from __future__ import annotations

import argparse
import ctypes
import json
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


TANH_OUTPUT_DIRECTORY = experiment_output_directory("tanh")
BUILD_DIRECTORY = REPOSITORY_OUTPUT_DIRECTORY / ".build"
C_FLOAT32_SRC = NATIVE_DIRECTORY / "tanh_float32_optimizer.c"
C_FLOAT32_LIB = BUILD_DIRECTORY / "tanh_float32_optimizer.dylib"
CACHE_VERSION = 2
DEFAULT_LAMBDAS = [10.0 ** (-k) for k in range(5, 13)]


def lambda_suffix(lam: float) -> str:
    return f"{lam:.0e}".replace("+0", "").replace("-0", "-")


def lambda_latex(lam: float) -> str:
    exponent = int(np.floor(np.log10(abs(lam))))
    mantissa = lam / (10.0**exponent)
    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:.1g}\times 10^{{{exponent}}}"


def max_iter_suffix(max_iter: int) -> str:
    exponent = int(np.log10(max_iter))
    if 10**exponent == max_iter:
        return f"1e{exponent}"
    return str(max_iter)


def max_iter_latex(max_iter: int) -> str:
    exponent = int(np.log10(max_iter))
    if 10**exponent == max_iter:
        return rf"10^{{{exponent}}}"
    return f"{max_iter:g}"


def output_paths(mode: str, max_iter: int, lambdas: list[float]) -> tuple[Path, Path, Path]:
    iter_part = max_iter_suffix(max_iter)
    if mode == "adaptive":
        stem = (
            "tanh_instability_adaptive_weight_penalty_"
            f"{lambda_suffix(lambdas[0])}_to_{lambda_suffix(lambdas[-1])}_{iter_part}"
        )
    else:
        stem = f"tanh_instability_unpenalized_{iter_part}"
    return (
        TANH_OUTPUT_DIRECTORY / f"{stem}.pdf",
        TANH_OUTPUT_DIRECTORY / f"{stem}.png",
        TANH_OUTPUT_DIRECTORY / f"{stem}.json",
    )


def cache_path(q_min: float, q_max: float, n_profile: int, n_quad: int) -> Path:
    q_min_s = f"{q_min:g}".replace("-", "m").replace(".", "p")
    q_max_s = f"{q_max:g}".replace("-", "m").replace(".", "p")
    return TANH_OUTPUT_DIRECTORY / (
        f"tanh_profile_cache_v{CACHE_VERSION}_q{q_min_s}_{q_max_s}_"
        f"n{n_profile}_N{n_quad}.npz"
    )


def sech2(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def target_fn(x: np.ndarray) -> np.ndarray:
    return x * sech2(x)


def feature_float64(x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    return amplitude * (np.tanh((1.0 + h) * x) - np.tanh((1.0 - h) * x))


def feature_float32(x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    h32 = np.float32(h)
    amp32 = np.float32(amplitude)
    left = np.tanh((np.float32(1.0) + h32) * x32).astype(np.float32)
    right = np.tanh((np.float32(1.0) - h32) * x32).astype(np.float32)
    return (amp32 * (left - right)).astype(np.float64)


def relative_l2_error(u: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * (u - target) ** 2) / np.sum(weights * target * target)))


def build_profile_cache(
    q_min: float,
    q_max: float,
    n_profile: int,
    n_quad: int,
    chunk_size: int = 80,
) -> dict[str, np.ndarray | float]:
    path = cache_path(q_min, q_max, n_profile, n_quad)
    if path.exists():
        data = np.load(path)
        return {
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
    target = target_fn(x)
    target_norm_sq = float(np.sum(trap * target * target))
    x32 = x.astype(np.float32)
    trap32 = trap.astype(np.float32)
    target32 = target.astype(np.float32)
    target_norm_sq32 = float(np.sum(trap32 * target32 * target32, dtype=np.float32))

    q_grid = np.linspace(q_min, q_max, n_profile)
    h_grid = np.exp(q_grid)
    q_grid32 = q_grid.astype(np.float32)
    h_grid32 = np.exp(q_grid32).astype(np.float32)
    b_scaled = np.empty_like(q_grid)
    c_scaled = np.empty_like(q_grid)
    bq_scaled = np.empty_like(q_grid)
    cq_scaled = np.empty_like(q_grid)
    b_scaled32 = np.empty_like(q_grid32)
    c_scaled32 = np.empty_like(q_grid32)
    bq_scaled32 = np.empty_like(q_grid32)
    cq_scaled32 = np.empty_like(q_grid32)

    wx = trap * x
    wx32 = (trap32 * x32).astype(np.float32)
    for start in range(0, n_profile, chunk_size):
        stop = min(start + chunk_size, n_profile)
        h = h_grid[start:stop][:, None]
        z_plus = (1.0 + h) * x[None, :]
        z_minus = (1.0 - h) * x[None, :]
        tanh_plus = np.tanh(z_plus)
        tanh_minus = np.tanh(z_minus)
        phi = tanh_plus - tanh_minus
        dphi_dh = x[None, :] * ((1.0 - tanh_plus * tanh_plus) + (1.0 - tanh_minus * tanh_minus))

        b = np.sum(trap[None, :] * phi * phi, axis=1)
        c = np.sum(trap[None, :] * target[None, :] * phi, axis=1)
        b_h = 2.0 * np.sum(trap[None, :] * phi * dphi_dh, axis=1)
        c_h = np.sum((wx * target)[None, :] * ((1.0 - tanh_plus * tanh_plus) + (1.0 - tanh_minus * tanh_minus)), axis=1)

        hs = h_grid[start:stop]
        b_scaled[start:stop] = b / (hs * hs)
        c_scaled[start:stop] = c / hs
        bq_scaled[start:stop] = b_h / hs
        cq_scaled[start:stop] = c_h

        h32 = h_grid32[start:stop][:, None]
        z_plus32 = ((np.float32(1.0) + h32) * x32[None, :]).astype(np.float32)
        z_minus32 = ((np.float32(1.0) - h32) * x32[None, :]).astype(np.float32)
        tanh_plus32 = np.tanh(z_plus32).astype(np.float32)
        tanh_minus32 = np.tanh(z_minus32).astype(np.float32)
        phi32 = (tanh_plus32 - tanh_minus32).astype(np.float32)
        sech_sum32 = (
            (np.float32(1.0) - tanh_plus32 * tanh_plus32)
            + (np.float32(1.0) - tanh_minus32 * tanh_minus32)
        ).astype(np.float32)
        dphi_dh32 = (x32[None, :] * sech_sum32).astype(np.float32)

        b32 = np.sum(trap32[None, :] * phi32 * phi32, axis=1, dtype=np.float32)
        c32 = np.sum(trap32[None, :] * target32[None, :] * phi32, axis=1, dtype=np.float32)
        b_h32 = np.float32(2.0) * np.sum(
            trap32[None, :] * phi32 * dphi_dh32,
            axis=1,
            dtype=np.float32,
        )
        c_h32 = np.sum(
            (wx32 * target32)[None, :] * sech_sum32,
            axis=1,
            dtype=np.float32,
        )

        hs32 = h_grid32[start:stop]
        b_scaled32[start:stop] = (b32 / (hs32 * hs32)).astype(np.float32)
        c_scaled32[start:stop] = (c32 / hs32).astype(np.float32)
        bq_scaled32[start:stop] = (b_h32 / hs32).astype(np.float32)
        cq_scaled32[start:stop] = c_h32.astype(np.float32)

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
    def __init__(self, cache: dict[str, np.ndarray | float]) -> None:
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
        self.q_grid32 = self.q_grid.astype(np.float32)
        self.q_min32 = np.float32(self.q_grid32[0])
        self.q_max32 = np.float32(self.q_grid32[-1])
        self.dq32 = np.float32(self.q_grid32[1] - self.q_grid32[0])
        self.inv_dq32 = np.float32(1.0) / self.dq32
        self.b_scaled32 = np.asarray(cache["b_scaled32"], dtype=np.float32)
        self.c_scaled32 = np.asarray(cache["c_scaled32"], dtype=np.float32)
        self.bq_scaled32 = np.asarray(cache["bq_scaled32"], dtype=np.float32)
        self.cq_scaled32 = np.asarray(cache["cq_scaled32"], dtype=np.float32)
        self.target_norm_sq32 = np.float32(cache["target_norm_sq32"])

    def _interp(self, arr: np.ndarray, q: float) -> float:
        if q <= self.q_min:
            return float(arr[0])
        if q >= self.q_max:
            return float(arr[-1])
        pos = (q - self.q_min) * self.inv_dq
        idx = int(pos)
        frac = pos - idx
        return float((1.0 - frac) * arr[idx] + frac * arr[idx + 1])

    def _interp32(self, arr: np.ndarray, q: np.float32) -> np.float32:
        if q <= self.q_min32:
            return np.float32(arr[0])
        if q >= self.q_max32:
            return np.float32(arr[-1])
        pos = np.float32((q - self.q_min32) * self.inv_dq32)
        idx = int(pos)
        frac = np.float32(pos - np.float32(idx))
        return np.float32((np.float32(1.0) - frac) * arr[idx] + frac * arr[idx + 1])

    def reduced_loss_and_grad(self, q: float, lam: float = 0.0) -> tuple[float, float, float, float, float]:
        h = float(np.exp(q))
        b_scaled = self._interp(self.b_scaled, q)
        c_scaled = self._interp(self.c_scaled, q)
        bq_scaled = self._interp(self.bq_scaled, q)
        cq_scaled = self._interp(self.cq_scaled, q)

        feature_norm_sq = h * h * b_scaled
        feature_target_ip = h * c_scaled
        d_feature_norm_sq_q = h * h * bq_scaled
        d_feature_target_ip_q = h * cq_scaled
        penalty_slope = lam * np.sqrt(2.0)
        shrink = feature_target_ip - penalty_slope

        if shrink <= 0.0:
            return 0.5 * self.target_norm_sq, 1.0, 0.0, h, 0.0

        amplitude = shrink / feature_norm_sq
        loss = 0.5 * (self.target_norm_sq - (shrink * shrink) / feature_norm_sq)
        grad_q = (
            -shrink * d_feature_target_ip_q / feature_norm_sq
            + 0.5 * shrink * shrink * d_feature_norm_sq_q / (feature_norm_sq * feature_norm_sq)
        )
        rel_error = float(np.sqrt(max(2.0 * loss, 0.0) / self.target_norm_sq))
        return float(loss), rel_error, float(grad_q), h, float(amplitude)

    def reduced_loss_and_grad_float32(
        self,
        q: float | np.float32,
        lam: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        q32 = np.float32(q)
        h = np.float32(np.exp(q32))
        b_scaled = self._interp32(self.b_scaled32, q32)
        c_scaled = self._interp32(self.c_scaled32, q32)
        bq_scaled = self._interp32(self.bq_scaled32, q32)
        cq_scaled = self._interp32(self.cq_scaled32, q32)

        feature_norm_sq = np.float32(h * h * b_scaled)
        feature_target_ip = np.float32(h * c_scaled)
        d_feature_norm_sq_q = np.float32(h * h * bq_scaled)
        d_feature_target_ip_q = np.float32(h * cq_scaled)
        penalty_slope = np.float32(lam) * np.float32(np.sqrt(np.float32(2.0)))
        shrink = np.float32(feature_target_ip - penalty_slope)

        if shrink <= np.float32(0.0):
            loss = np.float32(0.5) * self.target_norm_sq32
            return float(loss), 1.0, 0.0, float(h), 0.0

        amplitude = np.float32(shrink / feature_norm_sq)
        loss = np.float32(
            np.float32(0.5)
            * (self.target_norm_sq32 - np.float32(shrink * shrink / feature_norm_sq))
        )
        grad_q = np.float32(
            np.float32(-shrink * d_feature_target_ip_q / feature_norm_sq)
            + np.float32(
                np.float32(0.5)
                * shrink
                * shrink
                * d_feature_norm_sq_q
                / np.float32(feature_norm_sq * feature_norm_sq)
            )
        )
        rel_arg = np.float32(np.maximum(np.float32(2.0) * loss, np.float32(0.0)) / self.target_norm_sq32)
        rel_error = np.float32(np.sqrt(rel_arg))
        return float(loss), float(rel_error), float(grad_q), float(h), float(amplitude)


def record_iters(max_iter: int) -> set[int]:
    values = np.unique(
        np.concatenate(
            [
                np.arange(0, 10_001, 250, dtype=int),
                np.arange(12_500, 50_001, 2_500, dtype=int),
                np.arange(55_000, max_iter + 1, 5_000, dtype=int),
            ]
        )
    )
    return set(int(i) for i in values if i <= max_iter)


def load_float32_optimizer_lib() -> ctypes.CDLL | None:
    if not C_FLOAT32_SRC.exists():
        return None
    try:
        if not C_FLOAT32_LIB.exists() or C_FLOAT32_LIB.stat().st_mtime < C_FLOAT32_SRC.stat().st_mtime:
            BUILD_DIRECTORY.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "clang",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    str(C_FLOAT32_SRC),
                    "-o",
                    str(C_FLOAT32_LIB),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        lib = ctypes.CDLL(str(C_FLOAT32_LIB))
    except (OSError, subprocess.CalledProcessError):
        return None

    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    lib.run_float32_optimizer.argtypes = [
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        int_ptr,
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
    lib.run_float32_optimizer.restype = ctypes.c_int
    lib.run_adaptive_float32_optimizer.argtypes = [
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        float_ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        int_ptr,
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
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        ctypes.c_int,
    ]
    lib.run_adaptive_float32_optimizer.restype = ctypes.c_int
    return lib


def run_profiled_optimizer(
    profile: Profile,
    max_iter: int,
    lr: float,
    lam: float = 0.0,
) -> dict[str, list[float]]:
    q = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    m = 0.0
    v = 0.0
    records = record_iters(max_iter)

    hist: dict[str, list[float]] = {
        "iteration": [],
        "lambda": [],
        "profiled_relative_error": [],
        "separation": [],
        "amplitude": [],
        "weight_norm": [],
        "theta_norm": [],
        "grad_norm": [],
        "q": [],
    }

    for it in range(max_iter + 1):
        _, rel_error, grad_q, h, amplitude = profile.reduced_loss_and_grad(q, lam)

        if it in records or it == max_iter:
            hist["iteration"].append(float(it))
            hist["lambda"].append(float(lam))
            hist["profiled_relative_error"].append(float(rel_error))
            hist["separation"].append(float(2.0 * h))
            hist["amplitude"].append(float(amplitude))
            hist["weight_norm"].append(float(np.sqrt(2.0) * abs(amplitude)))
            hist["theta_norm"].append(float(np.sqrt(2.0 * amplitude * amplitude + 2.0 * h * h)))
            hist["grad_norm"].append(float(abs(grad_q)))
            hist["q"].append(float(q))

        if it == max_iter:
            break

        m = beta1 * m + (1.0 - beta1) * grad_q
        v = beta2 * v + (1.0 - beta2) * grad_q * grad_q
        m_hat = m / (1.0 - beta1 ** (it + 1))
        v_hat = v / (1.0 - beta2 ** (it + 1))
        q -= float(lr * m_hat / (np.sqrt(v_hat) + eps_adam))

    return hist


def run_profiled_optimizer_float32(
    profile: Profile,
    max_iter: int,
    lr: float,
    lam: float = 0.0,
) -> dict[str, list[float]]:
    q = np.float32(0.0)
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps_adam = np.float32(1e-8)
    lr32 = np.float32(lr)
    lam32 = np.float32(lam)
    m = np.float32(0.0)
    v = np.float32(0.0)
    records = record_iters(max_iter)

    hist: dict[str, list[float]] = {
        "iteration": [],
        "lambda": [],
        "profiled_relative_error": [],
        "separation": [],
        "amplitude": [],
        "weight_norm": [],
        "theta_norm": [],
        "grad_norm": [],
        "q": [],
    }

    for it in range(max_iter + 1):
        _, rel_error, grad_q, h, amplitude = profile.reduced_loss_and_grad_float32(q, float(lam32))
        grad_q32 = np.float32(grad_q)
        h32 = np.float32(h)
        amplitude32 = np.float32(amplitude)

        if it in records or it == max_iter:
            weight_norm32 = np.float32(np.sqrt(np.float32(2.0)) * np.abs(amplitude32))
            theta_norm32 = np.float32(
                np.sqrt(
                    np.float32(2.0) * amplitude32 * amplitude32
                    + np.float32(2.0) * h32 * h32
                )
            )
            hist["iteration"].append(float(it))
            hist["lambda"].append(float(lam32))
            hist["profiled_relative_error"].append(float(np.float32(rel_error)))
            hist["separation"].append(float(np.float32(2.0) * h32))
            hist["amplitude"].append(float(amplitude32))
            hist["weight_norm"].append(float(weight_norm32))
            hist["theta_norm"].append(float(theta_norm32))
            hist["grad_norm"].append(float(np.abs(grad_q32)))
            hist["q"].append(float(q))

        if it == max_iter:
            break

        m = np.float32(beta1 * m + (np.float32(1.0) - beta1) * grad_q32)
        v = np.float32(beta2 * v + (np.float32(1.0) - beta2) * grad_q32 * grad_q32)
        m_hat = np.float32(m / (np.float32(1.0) - np.float32(float(beta1) ** (it + 1))))
        v_hat = np.float32(v / (np.float32(1.0) - np.float32(float(beta2) ** (it + 1))))
        q = np.float32(q - np.float32(lr32 * m_hat / (np.float32(np.sqrt(v_hat)) + eps_adam)))

    return hist


def run_profiled_optimizer_float32_compiled(
    profile: Profile,
    max_iter: int,
    lr: float,
    lam: float = 0.0,
) -> dict[str, list[float]] | None:
    lib = load_float32_optimizer_lib()
    if lib is None:
        return None

    records = np.asarray(sorted(record_iters(max_iter) | {max_iter}), dtype=np.int32)
    n_records = int(records.size)
    columns = {
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

    b_scaled32 = np.ascontiguousarray(profile.b_scaled32, dtype=np.float32)
    c_scaled32 = np.ascontiguousarray(profile.c_scaled32, dtype=np.float32)
    bq_scaled32 = np.ascontiguousarray(profile.bq_scaled32, dtype=np.float32)
    cq_scaled32 = np.ascontiguousarray(profile.cq_scaled32, dtype=np.float32)
    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int)
    double_ptr = ctypes.POINTER(ctypes.c_double)

    written = lib.run_float32_optimizer(
        b_scaled32.ctypes.data_as(float_ptr),
        c_scaled32.ctypes.data_as(float_ptr),
        bq_scaled32.ctypes.data_as(float_ptr),
        cq_scaled32.ctypes.data_as(float_ptr),
        ctypes.c_int(b_scaled32.size),
        ctypes.c_float(float(profile.q_min32)),
        ctypes.c_float(float(profile.q_max32)),
        ctypes.c_float(float(profile.inv_dq32)),
        ctypes.c_float(float(profile.target_norm_sq32)),
        ctypes.c_int(max_iter),
        ctypes.c_float(lr),
        ctypes.c_float(lam),
        records.ctypes.data_as(int_ptr),
        ctypes.c_int(n_records),
        columns["iteration"].ctypes.data_as(double_ptr),
        columns["lambda"].ctypes.data_as(double_ptr),
        columns["profiled_relative_error"].ctypes.data_as(double_ptr),
        columns["separation"].ctypes.data_as(double_ptr),
        columns["amplitude"].ctypes.data_as(double_ptr),
        columns["weight_norm"].ctypes.data_as(double_ptr),
        columns["theta_norm"].ctypes.data_as(double_ptr),
        columns["grad_norm"].ctypes.data_as(double_ptr),
        columns["q"].ctypes.data_as(double_ptr),
    )
    if written != n_records:
        return None
    return {key: value.tolist() for key, value in columns.items()}


def run_adaptive_profiled_optimizer_float32_compiled(
    profile: Profile,
    lambdas: list[float],
    max_iter: int,
    lr: float,
    plateau_window: int,
    plateau_rtol: float,
) -> tuple[dict[str, list[float]], list[dict[str, float]]] | None:
    lib = load_float32_optimizer_lib()
    if lib is None:
        return None

    records = np.asarray(sorted(record_iters(max_iter) | {max_iter}), dtype=np.int32)
    n_records = int(records.size)
    columns = {
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
    max_switches = max(len(lambdas) - 1, 0)
    switch_columns = {
        "iteration": np.empty(max_switches, dtype=np.float64),
        "from_lambda": np.empty(max_switches, dtype=np.float64),
        "to_lambda": np.empty(max_switches, dtype=np.float64),
        "profiled_relative_error": np.empty(max_switches, dtype=np.float64),
        "separation": np.empty(max_switches, dtype=np.float64),
        "weight_norm": np.empty(max_switches, dtype=np.float64),
    }

    b_scaled32 = np.ascontiguousarray(profile.b_scaled32, dtype=np.float32)
    c_scaled32 = np.ascontiguousarray(profile.c_scaled32, dtype=np.float32)
    bq_scaled32 = np.ascontiguousarray(profile.bq_scaled32, dtype=np.float32)
    cq_scaled32 = np.ascontiguousarray(profile.cq_scaled32, dtype=np.float32)
    lambdas32 = np.ascontiguousarray(lambdas, dtype=np.float32)
    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int)
    double_ptr = ctypes.POINTER(ctypes.c_double)

    switch_count = lib.run_adaptive_float32_optimizer(
        b_scaled32.ctypes.data_as(float_ptr),
        c_scaled32.ctypes.data_as(float_ptr),
        bq_scaled32.ctypes.data_as(float_ptr),
        cq_scaled32.ctypes.data_as(float_ptr),
        ctypes.c_int(b_scaled32.size),
        ctypes.c_float(float(profile.q_min32)),
        ctypes.c_float(float(profile.q_max32)),
        ctypes.c_float(float(profile.inv_dq32)),
        ctypes.c_float(float(profile.target_norm_sq32)),
        lambdas32.ctypes.data_as(float_ptr),
        ctypes.c_int(lambdas32.size),
        ctypes.c_int(max_iter),
        ctypes.c_float(lr),
        ctypes.c_int(plateau_window),
        ctypes.c_float(plateau_rtol),
        records.ctypes.data_as(int_ptr),
        ctypes.c_int(n_records),
        columns["iteration"].ctypes.data_as(double_ptr),
        columns["lambda"].ctypes.data_as(double_ptr),
        columns["profiled_relative_error"].ctypes.data_as(double_ptr),
        columns["separation"].ctypes.data_as(double_ptr),
        columns["amplitude"].ctypes.data_as(double_ptr),
        columns["weight_norm"].ctypes.data_as(double_ptr),
        columns["theta_norm"].ctypes.data_as(double_ptr),
        columns["grad_norm"].ctypes.data_as(double_ptr),
        columns["q"].ctypes.data_as(double_ptr),
        switch_columns["iteration"].ctypes.data_as(double_ptr),
        switch_columns["from_lambda"].ctypes.data_as(double_ptr),
        switch_columns["to_lambda"].ctypes.data_as(double_ptr),
        switch_columns["profiled_relative_error"].ctypes.data_as(double_ptr),
        switch_columns["separation"].ctypes.data_as(double_ptr),
        switch_columns["weight_norm"].ctypes.data_as(double_ptr),
        ctypes.c_int(max_switches),
    )
    if switch_count < 0 or switch_count > max_switches:
        return None

    hist = {key: value.tolist() for key, value in columns.items()}
    switches = [
        {
            "iteration": float(switch_columns["iteration"][i]),
            "from_lambda": float(switch_columns["from_lambda"][i]),
            "to_lambda": float(switch_columns["to_lambda"][i]),
            "profiled_relative_error": float(switch_columns["profiled_relative_error"][i]),
            "separation": float(switch_columns["separation"][i]),
            "weight_norm": float(switch_columns["weight_norm"][i]),
        }
        for i in range(switch_count)
    ]
    return hist, switches


def run_adaptive_profiled_optimizer(
    profile: Profile,
    lambdas: list[float],
    max_iter: int,
    lr: float,
    plateau_window: int,
    plateau_rtol: float,
) -> tuple[dict[str, list[float]], list[dict[str, float]]]:
    q = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    m = 0.0
    v = 0.0
    stage = 0
    best_stage_error = np.inf
    last_improvement_iter = 0
    records = record_iters(max_iter)

    hist: dict[str, list[float]] = {
        "iteration": [],
        "lambda": [],
        "profiled_relative_error": [],
        "separation": [],
        "amplitude": [],
        "weight_norm": [],
        "theta_norm": [],
        "grad_norm": [],
        "q": [],
    }
    switches: list[dict[str, float]] = []

    for it in range(max_iter + 1):
        lam = lambdas[stage]
        _, rel_error, grad_q, h, amplitude = profile.reduced_loss_and_grad(q, lam)

        if rel_error < best_stage_error * (1.0 - plateau_rtol):
            best_stage_error = rel_error
            last_improvement_iter = it

        if it in records or it == max_iter:
            hist["iteration"].append(float(it))
            hist["lambda"].append(float(lam))
            hist["profiled_relative_error"].append(float(rel_error))
            hist["separation"].append(float(2.0 * h))
            hist["amplitude"].append(float(amplitude))
            hist["weight_norm"].append(float(np.sqrt(2.0) * abs(amplitude)))
            hist["theta_norm"].append(float(np.sqrt(2.0 * amplitude * amplitude + 2.0 * h * h)))
            hist["grad_norm"].append(float(abs(grad_q)))
            hist["q"].append(float(q))

        should_switch = stage < len(lambdas) - 1 and it - last_improvement_iter >= plateau_window
        if should_switch:
            switches.append(
                {
                    "iteration": float(it),
                    "from_lambda": float(lam),
                    "to_lambda": float(lambdas[stage + 1]),
                    "profiled_relative_error": float(rel_error),
                    "separation": float(2.0 * h),
                    "weight_norm": float(np.sqrt(2.0) * abs(amplitude)),
                }
            )
            stage += 1
            best_stage_error = np.inf
            last_improvement_iter = it
            m = 0.0
            v = 0.0
            continue

        if it == max_iter:
            break

        m = beta1 * m + (1.0 - beta1) * grad_q
        v = beta2 * v + (1.0 - beta2) * grad_q * grad_q
        m_hat = m / (1.0 - beta1 ** (it + 1))
        v_hat = v / (1.0 - beta2 ** (it + 1))
        q -= float(lr * m_hat / (np.sqrt(v_hat) + eps_adam))

    return hist, switches


def observed_training_errors(
    hist: dict[str, list[float]],
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    err64 = []
    err32 = []
    for separation, amplitude in zip(hist["separation"], hist["amplitude"]):
        h = 0.5 * float(separation)
        a = float(amplitude)
        err64.append(relative_l2_error(feature_float64(x, h, a), target, weights))
        err32.append(relative_l2_error(feature_float32(x, h, a), target, weights))
    return np.asarray(err64), np.asarray(err32)


def plot_optimizer_panel(
    ax: plt.Axes,
    iters: np.ndarray,
    error: np.ndarray,
    separation: np.ndarray,
    weight_norm: np.ndarray,
    marker_iter: float,
    switches: list[dict[str, float]],
    title: str,
    blue: str,
    red: str,
    gray: str,
    marker_gray: str,
    show_separation: bool = True,
    show_legend: bool = True,
) -> None:
    err_line, = ax.semilogy(iters, error, color=blue, lw=1.7, label=r"relative $L^2$ error")
    handles = [err_line]
    labels = [err_line.get_label()]
    if show_separation:
        sep_line, = ax.semilogy(
            iters,
            separation,
            color=gray,
            lw=1.5,
            ls=(0, (3, 2)),
            label=r"slope separation $\delta$",
        )
        handles.append(sep_line)
        labels.append(sep_line.get_label())
    ax.axvline(marker_iter, color=marker_gray, lw=0.8, ls=(0, (1.5, 1.5)))
    for switch in switches:
        ax.axvline(switch["iteration"], color=marker_gray, lw=0.45, ls=(0, (1.5, 1.5)), alpha=0.45)
    ax.set_xlabel("iteration")
    if show_separation:
        ax.set_ylabel(r"relative $L^2$ error and separation", labelpad=9)
    else:
        ax.set_ylabel(r"relative $L^2$ error", labelpad=9)
    ax.set_title(title)
    ax2 = ax.twinx()
    norm_line, = ax2.plot(iters, weight_norm, color=red, lw=1.7, label=r"$\|c\|_2$")
    ax2.set_ylabel(r"$\|c\|_2$", color=red, labelpad=7)
    ax2.tick_params(axis="y", labelcolor=red)
    handles.append(norm_line)
    labels.append(norm_line.get_label())
    if show_legend:
        ax.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.33),
            ncol=len(handles),
            columnspacing=1.0,
            handlelength=2.2,
        )


def plot_figure(
    hist: dict[str, list[float]],
    switches: list[dict[str, float]],
    train_err64: np.ndarray,
    train_err32: np.ndarray,
    mode: str,
    max_iter: int,
    lambdas: list[float],
    out_fig: Path,
    out_png: Path,
    hist_float32_optimizer: dict[str, list[float]] | None = None,
    float32_optimizer_error: np.ndarray | None = None,
    switches_float32_optimizer: list[dict[str, float]] | None = None,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.7,
        }
    )

    iters = np.asarray(hist["iteration"])
    opt_delta = np.asarray(hist["separation"])
    opt_weight = np.asarray(hist["weight_norm"])
    best32_idx = int(np.argmin(train_err32))
    best64_idx = int(np.argmin(train_err64))
    marker_iter = iters[best32_idx] if mode == "unpenalized" else iters[best64_idx]

    blue = "#0000FF"
    red = "#FF0000"
    gray = "0.25"
    marker_gray = "0.45"

    has_float32_optimizer = (
        hist_float32_optimizer is not None
        and float32_optimizer_error is not None
    )
    figure6_style = mode == "adaptive" and has_float32_optimizer
    ncols = 3 if has_float32_optimizer else 2
    figsize = (10.9, 2.85) if has_float32_optimizer else (7.4, 2.85)
    fig, axes_obj = plt.subplots(1, ncols, figsize=figsize)
    axes = np.asarray(axes_obj).ravel()
    if has_float32_optimizer:
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.31, top=0.84, wspace=0.58)
    else:
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.28, top=0.84, wspace=0.48)

    ax = axes[0]
    ax.semilogy(iters, train_err64, color=blue, lw=1.7, label="float64")
    ax.semilogy(iters, train_err32, color=red, lw=1.7, label="float32")
    ax.axvline(marker_iter, color=marker_gray, lw=0.8, ls=(0, (1.5, 1.5)))
    for switch in switches:
        ax.axvline(switch["iteration"], color=marker_gray, lw=0.45, ls=(0, (1.5, 1.5)), alpha=0.45)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"relative $L^2$ error", labelpad=9)
    if mode == "adaptive":
        ax.set_title(r"(a) adaptive tanh states")
    else:
        ax.set_title(r"(a) tanh states, finite precision")
    if not figure6_style:
        ax.legend(frameon=False, loc="upper left")

    if has_float32_optimizer:
        hist32 = hist_float32_optimizer
        switches32 = [] if switches_float32_optimizer is None else switches_float32_optimizer
        iters32 = np.asarray(hist32["iteration"])
        opt_delta32 = np.asarray(hist32["separation"])
        opt_weight32 = np.asarray(hist32["weight_norm"])
        marker_iter32 = iters32[int(np.argmin(float32_optimizer_error))]
        if mode == "adaptive":
            title32 = r"(b) float32 adaptive"
            title64 = r"(c) float64 adaptive"
        else:
            title32 = rf"(b) float32 no penalty, ${max_iter_latex(max_iter)}$ iterations"
            title64 = rf"(c) float64 no penalty, ${max_iter_latex(max_iter)}$ iterations"
        plot_optimizer_panel(
            axes[1],
            iters32,
            float32_optimizer_error,
            opt_delta32,
            opt_weight32,
            marker_iter32,
            switches32,
            title32,
            blue,
            red,
            gray,
            marker_gray,
            show_separation=not figure6_style,
            show_legend=True,
        )
        plot_optimizer_panel(
            axes[2],
            iters,
            train_err64,
            opt_delta,
            opt_weight,
            marker_iter,
            switches,
            title64,
            blue,
            red,
            gray,
            marker_gray,
            show_separation=not figure6_style,
            show_legend=not figure6_style,
        )
    else:
        if mode == "adaptive":
            title = (
                rf"(b) adaptive $\lambda\|c\|_2$, "
                rf"${lambda_latex(lambdas[0])}\to {lambda_latex(lambdas[-1])}$"
            )
        else:
            title = rf"(b) no penalty, ${max_iter_latex(max_iter)}$ iterations"
        plot_optimizer_panel(
            axes[1],
            iters,
            train_err64,
            opt_delta,
            opt_weight,
            marker_iter,
            switches,
            title,
            blue,
            red,
            gray,
            marker_gray,
        )

    fig.savefig(out_fig, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")


def write_payload(
    hist: dict[str, list[float]],
    switches: list[dict[str, float]],
    train_err64: np.ndarray,
    train_err32: np.ndarray,
    mode: str,
    max_iter: int,
    lr: float,
    lambdas: list[float],
    plateau_window: int,
    plateau_rtol: float,
    out_json: Path,
    hist_float32_optimizer: dict[str, list[float]] | None = None,
    train_err64_float32_optimizer: np.ndarray | None = None,
    train_err32_float32_optimizer: np.ndarray | None = None,
    switches_float32_optimizer: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    iters = np.asarray(hist["iteration"])
    opt_delta = np.asarray(hist["separation"])
    opt_amp = np.asarray(hist["amplitude"])
    opt_weight = np.asarray(hist["weight_norm"])
    opt_grad = np.asarray(hist["grad_norm"])
    lam_hist = np.asarray(hist["lambda"])
    best64_idx = int(np.argmin(train_err64))
    best32_idx = int(np.argmin(train_err32))

    payload = {
        "mode": mode,
        "model": "a*(tanh((1+h)x)-tanh((1-h)x))",
        "target": "x sech^2(x)",
        "max_iter": max_iter,
        "lr": lr,
        "penalty": "adaptive lambda * ||c||_2" if mode == "adaptive" else "none",
        "lambdas": lambdas if mode == "adaptive" else [],
        "plateau_window": plateau_window if mode == "adaptive" else None,
        "plateau_rtol": plateau_rtol if mode == "adaptive" else None,
        "switches": switches,
        "optimizer_trace": hist,
        "training_precision_trace": [
            {
                "iteration": float(it),
                "lambda": float(lam),
                "rel_error_float64": float(e64),
                "rel_error_float32": float(e32),
                "separation": float(sep),
                "amplitude": float(amp),
                "weight_norm": float(wn),
                "grad_norm": float(gn),
            }
            for it, lam, e64, e32, sep, amp, wn, gn in zip(
                iters,
                lam_hist,
                train_err64,
                train_err32,
                opt_delta,
                opt_amp,
                opt_weight,
                opt_grad,
            )
        ],
        "best_float64": {
            "iteration": float(iters[best64_idx]),
            "lambda": float(lam_hist[best64_idx]),
            "rel_error_float64": float(train_err64[best64_idx]),
            "rel_error_float32": float(train_err32[best64_idx]),
            "separation": float(opt_delta[best64_idx]),
            "amplitude": float(opt_amp[best64_idx]),
            "weight_norm": float(opt_weight[best64_idx]),
        },
        "best_float32": {
            "iteration": float(iters[best32_idx]),
            "lambda": float(lam_hist[best32_idx]),
            "rel_error_float64": float(train_err64[best32_idx]),
            "rel_error_float32": float(train_err32[best32_idx]),
            "separation": float(opt_delta[best32_idx]),
            "amplitude": float(opt_amp[best32_idx]),
            "weight_norm": float(opt_weight[best32_idx]),
        },
        "final": {
            "iteration": float(iters[-1]),
            "lambda": float(lam_hist[-1]),
            "rel_error_float64": float(train_err64[-1]),
            "rel_error_float32": float(train_err32[-1]),
            "separation": float(opt_delta[-1]),
            "amplitude": float(opt_amp[-1]),
            "weight_norm": float(opt_weight[-1]),
            "grad_norm": float(opt_grad[-1]),
        },
    }
    if (
        hist_float32_optimizer is not None
        and train_err64_float32_optimizer is not None
        and train_err32_float32_optimizer is not None
    ):
        iters32 = np.asarray(hist_float32_optimizer["iteration"])
        opt_delta32 = np.asarray(hist_float32_optimizer["separation"])
        opt_amp32 = np.asarray(hist_float32_optimizer["amplitude"])
        opt_weight32 = np.asarray(hist_float32_optimizer["weight_norm"])
        opt_grad32 = np.asarray(hist_float32_optimizer["grad_norm"])
        lam_hist32 = np.asarray(hist_float32_optimizer["lambda"])
        profiled_error32 = np.asarray(hist_float32_optimizer["profiled_relative_error"])
        best_profile32_idx = int(np.argmin(profiled_error32))
        best_eval32_idx = int(np.argmin(train_err32_float32_optimizer))
        payload["optimizer_trace_float32_throughout"] = hist_float32_optimizer
        payload["switches_float32_throughout"] = [] if switches_float32_optimizer is None else switches_float32_optimizer
        payload["float32_throughout_training_trace"] = [
            {
                "iteration": float(it),
                "lambda": float(lam),
                "profiled_rel_error_float32": float(e_profile),
                "rel_error_evaluated_float64": float(e64),
                "rel_error_evaluated_float32": float(e32),
                "separation": float(sep),
                "amplitude": float(amp),
                "weight_norm": float(wn),
                "grad_norm": float(gn),
            }
            for it, lam, e_profile, e64, e32, sep, amp, wn, gn in zip(
                iters32,
                lam_hist32,
                profiled_error32,
                train_err64_float32_optimizer,
                train_err32_float32_optimizer,
                opt_delta32,
                opt_amp32,
                opt_weight32,
                opt_grad32,
            )
        ]
        payload["best_float32_throughout_profiled"] = {
            "iteration": float(iters32[best_profile32_idx]),
            "lambda": float(lam_hist32[best_profile32_idx]),
            "profiled_rel_error_float32": float(profiled_error32[best_profile32_idx]),
            "rel_error_evaluated_float64": float(train_err64_float32_optimizer[best_profile32_idx]),
            "rel_error_evaluated_float32": float(train_err32_float32_optimizer[best_profile32_idx]),
            "separation": float(opt_delta32[best_profile32_idx]),
            "amplitude": float(opt_amp32[best_profile32_idx]),
            "weight_norm": float(opt_weight32[best_profile32_idx]),
        }
        payload["best_float32_throughout_evaluated"] = {
            "iteration": float(iters32[best_eval32_idx]),
            "lambda": float(lam_hist32[best_eval32_idx]),
            "profiled_rel_error_float32": float(profiled_error32[best_eval32_idx]),
            "rel_error_evaluated_float64": float(train_err64_float32_optimizer[best_eval32_idx]),
            "rel_error_evaluated_float32": float(train_err32_float32_optimizer[best_eval32_idx]),
            "separation": float(opt_delta32[best_eval32_idx]),
            "amplitude": float(opt_amp32[best_eval32_idx]),
            "weight_norm": float(opt_weight32[best_eval32_idx]),
        }
        payload["final_float32_throughout"] = {
            "iteration": float(iters32[-1]),
            "lambda": float(lam_hist32[-1]),
            "profiled_rel_error_float32": float(profiled_error32[-1]),
            "rel_error_evaluated_float64": float(train_err64_float32_optimizer[-1]),
            "rel_error_evaluated_float32": float(train_err32_float32_optimizer[-1]),
            "separation": float(opt_delta32[-1]),
            "amplitude": float(opt_amp32[-1]),
            "weight_norm": float(opt_weight32[-1]),
            "grad_norm": float(opt_grad32[-1]),
        }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        "best64_iter": float(iters[best64_idx]),
        "best64_error": float(train_err64[best64_idx]),
        "best32_iter": float(iters[best32_idx]),
        "best32_error": float(train_err32[best32_idx]),
        "final64_error": float(train_err64[-1]),
        "final32_error": float(train_err32[-1]),
        "final_delta": float(opt_delta[-1]),
        "final_weight_norm": float(opt_weight[-1]),
        "final_lambda": float(lam_hist[-1]),
        "switch_count": float(len(switches)),
    }
    if (
        hist_float32_optimizer is not None
        and train_err64_float32_optimizer is not None
        and train_err32_float32_optimizer is not None
    ):
        summary.update(
            {
                "float32_throughout_best_profile_iter": float(iters32[best_profile32_idx]),
                "float32_throughout_best_profile_error": float(profiled_error32[best_profile32_idx]),
                "float32_throughout_best_eval_iter": float(iters32[best_eval32_idx]),
                "float32_throughout_best_eval_error": float(train_err32_float32_optimizer[best_eval32_idx]),
                "float32_throughout_final_profile_error": float(profiled_error32[-1]),
                "float32_throughout_final_eval_error": float(train_err32_float32_optimizer[-1]),
                "float32_throughout_final_delta": float(opt_delta32[-1]),
                "float32_throughout_final_weight_norm": float(opt_weight32[-1]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["unpenalized", "adaptive"], required=True)
    parser.add_argument("--max-iter", type=int, default=10_000_000)
    parser.add_argument("--lr", type=float, default=5e-1)
    parser.add_argument("--start-exp", type=int, default=5)
    parser.add_argument("--end-exp", type=int, default=12)
    parser.add_argument("--plateau-window", type=int, default=50_000)
    parser.add_argument("--plateau-rtol", type=float, default=1e-3)
    parser.add_argument("--q-min", type=float, default=-18.0)
    parser.add_argument("--q-max", type=float, default=0.5)
    parser.add_argument("--n-profile", type=int, default=6000)
    parser.add_argument("--n-quad", type=int, default=20001)
    args = parser.parse_args()
    TANH_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    lambdas = [10.0 ** (-k) for k in range(args.start_exp, args.end_exp + 1)]
    out_fig, out_png, out_json = output_paths(args.mode, args.max_iter, lambdas)
    cache = build_profile_cache(args.q_min, args.q_max, args.n_profile, args.n_quad)
    profile = Profile(cache)
    hist_float32_optimizer = None
    switches_float32_optimizer = None
    train_err64_float32_optimizer = None
    train_err32_float32_optimizer = None

    if args.mode == "adaptive":
        hist, switches = run_adaptive_profiled_optimizer(
            profile,
            lambdas=lambdas,
            max_iter=args.max_iter,
            lr=args.lr,
            plateau_window=args.plateau_window,
            plateau_rtol=args.plateau_rtol,
        )
        adaptive_float32 = run_adaptive_profiled_optimizer_float32_compiled(
            profile,
            lambdas=lambdas,
            max_iter=args.max_iter,
            lr=args.lr,
            plateau_window=args.plateau_window,
            plateau_rtol=args.plateau_rtol,
        )
        if adaptive_float32 is not None:
            hist_float32_optimizer, switches_float32_optimizer = adaptive_float32
    else:
        hist = run_profiled_optimizer(profile, max_iter=args.max_iter, lr=args.lr)
        hist_float32_optimizer = run_profiled_optimizer_float32_compiled(
            profile,
            max_iter=args.max_iter,
            lr=args.lr,
        )
        if hist_float32_optimizer is None:
            hist_float32_optimizer = run_profiled_optimizer_float32(
                profile,
                max_iter=args.max_iter,
                lr=args.lr,
            )
        switches = []

    x = np.arange(-10.0, 10.0 + 0.5e-3, 1e-3)
    trap = np.full_like(x, 1e-3)
    trap[0] = trap[-1] = 0.5e-3
    target = target_fn(x)
    train_err64, train_err32 = observed_training_errors(hist, x, target, trap)
    if hist_float32_optimizer is not None:
        train_err64_float32_optimizer, train_err32_float32_optimizer = observed_training_errors(
            hist_float32_optimizer,
            x,
            target,
            trap,
        )
        float32_optimizer_error = train_err32_float32_optimizer
    else:
        float32_optimizer_error = None

    plot_figure(
        hist,
        switches,
        train_err64,
        train_err32,
        args.mode,
        args.max_iter,
        lambdas,
        out_fig,
        out_png,
        hist_float32_optimizer=hist_float32_optimizer,
        float32_optimizer_error=float32_optimizer_error,
        switches_float32_optimizer=switches_float32_optimizer,
    )
    summary = write_payload(
        hist,
        switches,
        train_err64,
        train_err32,
        args.mode,
        args.max_iter,
        args.lr,
        lambdas,
        args.plateau_window,
        args.plateau_rtol,
        out_json,
        hist_float32_optimizer=hist_float32_optimizer,
        train_err64_float32_optimizer=train_err64_float32_optimizer,
        train_err32_float32_optimizer=train_err32_float32_optimizer,
        switches_float32_optimizer=switches_float32_optimizer,
    )

    print(f"Wrote {out_fig}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")
    if switches:
        print("Switches:")
        for switch in switches:
            print(
                f"  iter={switch['iteration']:.0f}: "
                f"{switch['from_lambda']:.0e} -> {switch['to_lambda']:.0e}, "
                f"rel_error={switch['profiled_relative_error']:.3e}, "
                f"||c||={switch['weight_norm']:.3e}"
            )
    print(
        "Summary: "
        f"best64_iter={summary['best64_iter']:.0f}, "
        f"best64={summary['best64_error']:.3e}, "
        f"best32_iter={summary['best32_iter']:.0f}, "
        f"best32={summary['best32_error']:.3e}, "
        f"final64={summary['final64_error']:.3e}, "
        f"final32={summary['final32_error']:.3e}, "
        f"delta={summary['final_delta']:.3e}, "
        f"||c||={summary['final_weight_norm']:.3e}, "
        f"lambda={summary['final_lambda']:.0e}, "
        f"switches={summary['switch_count']:.0f}"
    )
    if "float32_throughout_final_profile_error" in summary:
        print(
            "Float32-throughout summary: "
            f"best_profile_iter={summary['float32_throughout_best_profile_iter']:.0f}, "
            f"best_profile={summary['float32_throughout_best_profile_error']:.3e}, "
            f"best_eval_iter={summary['float32_throughout_best_eval_iter']:.0f}, "
            f"best_eval={summary['float32_throughout_best_eval_error']:.3e}, "
            f"final_profile={summary['float32_throughout_final_profile_error']:.3e}, "
            f"final_eval={summary['float32_throughout_final_eval_error']:.3e}, "
            f"delta={summary['float32_throughout_final_delta']:.3e}, "
            f"||c||={summary['float32_throughout_final_weight_norm']:.3e}"
        )


if __name__ == "__main__":
    main()
