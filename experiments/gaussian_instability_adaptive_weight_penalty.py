"""Adaptive weight-penalty variant of the Gaussian RBF collision diagnostic.

This uses the same two-neuron first-difference collision problem,

    u_{a,h}(x) = a (K(x-h/2) - K(x+h/2)),     u*(x)=xK(x),

with h equal to the plotted center separation.  The run starts with
lambda = 1e-5 in lambda * ||a||, where a denotes the signed amplitude
vector.  When the
unregularized relative L2 error has not improved by a relative factor of
plateau_rtol over plateau_window iterations, lambda is decreased by one
decade and Adam's moments are reset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from artifact_paths import GAUSSIAN_DATA_DIRECTORY, experiment_output_directory
from gaussian_instability_experiment import (
    GRAM_C,
    TARGET_NORM_SQ,
    gaussian,
    relative_l2_error,
)
from gaussian_instability_weight_penalty import lambda_latex, lambda_suffix


DEFAULT_LAMBDAS = [10.0 ** (-k) for k in range(5, 13)]
Q_INIT = 2.0
ERROR_FLOOR = 1e-16
ERROR_CAP = 1e2
GAUSSIAN_OUTPUT_DIRECTORY = experiment_output_directory("gaussian")


def output_paths(lambdas: list[float]) -> tuple[Path, Path, Path]:
    stem = f"gaussian_instability_weight_penalty_adaptive_{lambda_suffix(lambdas[0])}_to_{lambda_suffix(lambdas[-1])}"
    return (
        GAUSSIAN_OUTPUT_DIRECTORY / f"{stem}.pdf",
        GAUSSIAN_OUTPUT_DIRECTORY / f"{stem}.png",
        GAUSSIAN_DATA_DIRECTORY / f"{stem}.json",
    )


def target_fn(x: np.ndarray) -> np.ndarray:
    return x * gaussian(x)


def first_difference_model_float64(x: np.ndarray, separation: float, amplitude: float) -> np.ndarray:
    half_sep = 0.5 * separation
    return amplitude * (gaussian(x - half_sep) - gaussian(x + half_sep))


def first_difference_model_float32(x: np.ndarray, separation: float, amplitude: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    half_sep32 = np.float32(0.5) * np.float32(separation)
    amp32 = np.float32(amplitude)
    c32 = np.float32(1.0 / np.sqrt(np.float32(2.0) * np.float32(np.pi)))
    right_center = x32 - half_sep32
    left_center = x32 + half_sep32
    right = np.exp(np.float32(-0.5) * right_center * right_center, dtype=np.float32) * c32
    left = np.exp(np.float32(-0.5) * left_center * left_center, dtype=np.float32) * c32
    return (amp32 * (right - left)).astype(np.float64)


def feature_terms(separation: float) -> tuple[float, float]:
    feature_norm_sq = 2.0 * GRAM_C * (-np.expm1(-(separation * separation) / 4.0))
    feature_target_ip = 0.5 * separation * GRAM_C * np.exp(-(separation * separation) / 16.0)
    return feature_norm_sq, feature_target_ip


def analytic_relative_l2_error(separation: float, amplitude: float) -> float:
    feature_norm_sq, feature_target_ip = feature_terms(separation)
    err_sq = (
        amplitude * amplitude * feature_norm_sq
        - 2.0 * amplitude * feature_target_ip
        + TARGET_NORM_SQ
    )
    return float(np.sqrt(max(err_sq, 0.0) / TARGET_NORM_SQ))


def penalized_reduced_loss_and_grad_fig5(
    q: float,
    lam: float,
    precision: str = "float64",
) -> tuple[float, float, float, float, float]:
    dtype = np.float32 if precision == "float32" else np.float64
    qv = dtype(q)
    lamv = dtype(lam)
    separation_v = dtype(np.exp(qv))
    separation_sq = dtype(separation_v * separation_v)
    gram_c = dtype(GRAM_C)
    target_norm_sq = dtype(TARGET_NORM_SQ)

    feature_norm_sq = dtype(
        dtype(2.0)
        * gram_c
        * dtype(-np.expm1(dtype(-separation_sq / dtype(4.0))))
    )
    feature_target_ip = dtype(
        dtype(0.5)
        * separation_v
        * gram_c
        * dtype(np.exp(dtype(-separation_sq / dtype(16.0))))
    )
    penalty_slope = dtype(lamv * dtype(np.sqrt(dtype(2.0))))
    shrink = feature_target_ip - penalty_slope

    if shrink <= 0.0:
        return float(dtype(0.5) * target_norm_sq), 1.0, 0.0, float(separation_v), 0.0

    d_feature_norm_sq = dtype(
        gram_c * separation_v * dtype(np.exp(dtype(-separation_sq / dtype(4.0))))
    )
    d_feature_target_ip = dtype(
        dtype(0.5)
        * gram_c
        * dtype(np.exp(dtype(-separation_sq / dtype(16.0))))
        * dtype(dtype(1.0) - separation_sq / dtype(8.0))
    )

    amplitude = dtype(shrink / feature_norm_sq)
    penalized_loss = dtype(
        dtype(0.5) * dtype(target_norm_sq - dtype(shrink * shrink) / feature_norm_sq)
    )
    grad_sep = dtype(
        dtype(-shrink * d_feature_target_ip / feature_norm_sq)
        + dtype(
            dtype(0.5)
            * dtype(shrink * shrink)
            * d_feature_norm_sq
            / dtype(feature_norm_sq * feature_norm_sq)
        )
    )
    grad_q = dtype(separation_v * grad_sep)
    rel_error = analytic_relative_l2_error(float(separation_v), float(amplitude))
    return (
        float(penalized_loss),
        float(rel_error),
        float(grad_q),
        float(separation_v),
        float(amplitude),
    )


def run_adaptive_profiled_optimizer(
    lambdas: list[float],
    max_iter: int = 1_000_000,
    lr: float = 5e-1,
    plateau_window: int = 50_000,
    plateau_rtol: float = 1e-3,
    precision: str = "float64",
) -> tuple[dict[str, list[float]], list[dict[str, float]]]:
    dtype = np.float32 if precision == "float32" else np.float64
    q = dtype(Q_INIT)
    beta1 = dtype(0.9)
    beta2 = dtype(0.999)
    eps_adam = dtype(1e-8)
    lr_v = dtype(lr)
    one = dtype(1.0)
    m = dtype(0.0)
    v = dtype(0.0)
    stage = 0
    best_stage_error = np.inf
    last_improvement_iter = 0

    record_iters = np.unique(
        np.concatenate(
            [
                np.arange(0, 10_001, 250, dtype=int),
                np.arange(12_500, 50_001, 2_500, dtype=int),
                np.arange(55_000, max_iter + 1, 5_000, dtype=int),
            ]
        )
    )
    record_iters = set(int(i) for i in record_iters if i <= max_iter)

    hist: dict[str, list[float]] = {
        "iteration": [],
        "lambda": [],
        "relative_error": [],
        "separation": [],
        "amplitude": [],
        "weight_norm": [],
        "theta_norm": [],
        "grad_norm": [],
    }
    switches: list[dict[str, float]] = []

    for it in range(max_iter + 1):
        lam = lambdas[stage]
        _, rel_error, grad_q, separation, amplitude = penalized_reduced_loss_and_grad_fig5(
            float(q),
            lam,
            precision=precision,
        )

        if rel_error < best_stage_error * (1.0 - plateau_rtol):
            best_stage_error = rel_error
            last_improvement_iter = it

        if it in record_iters or it == max_iter:
            hist["iteration"].append(float(it))
            hist["lambda"].append(float(lam))
            hist["relative_error"].append(float(rel_error))
            hist["separation"].append(float(separation))
            hist["amplitude"].append(float(amplitude))
            hist["weight_norm"].append(float(np.sqrt(2.0) * abs(amplitude)))
            hist["theta_norm"].append(float(np.sqrt(2.0 * amplitude * amplitude + 0.5 * separation * separation)))
            hist["grad_norm"].append(float(abs(grad_q)))

        should_switch = (
            stage < len(lambdas) - 1
            and it - last_improvement_iter >= plateau_window
        )
        if should_switch:
            switches.append(
                {
                    "iteration": float(it),
                    "from_lambda": float(lam),
                    "to_lambda": float(lambdas[stage + 1]),
                    "relative_error": float(rel_error),
                    "separation": float(separation),
                    "weight_norm": float(np.sqrt(2.0) * abs(amplitude)),
                }
            )
            stage += 1
            best_stage_error = np.inf
            last_improvement_iter = it
            m = dtype(0.0)
            v = dtype(0.0)
            continue

        if it == max_iter:
            break

        grad_v = dtype(grad_q)
        m = dtype(beta1 * m + dtype(one - beta1) * grad_v)
        v = dtype(beta2 * v + dtype(one - beta2) * dtype(grad_v * grad_v))
        beta1_pow = dtype(beta1 ** dtype(it + 1))
        beta2_pow = dtype(beta2 ** dtype(it + 1))
        m_hat = dtype(m / dtype(one - beta1_pow))
        v_hat = dtype(v / dtype(one - beta2_pow))
        q = dtype(q - dtype(lr_v * m_hat / dtype(dtype(np.sqrt(v_hat)) + eps_adam)))

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
        a = float(amplitude)
        err64.append(relative_l2_error(first_difference_model_float64(x, float(separation), a), target, weights))
        err32.append(relative_l2_error(first_difference_model_float32(x, float(separation), a), target, weights))
    return np.asarray(err64), np.asarray(err32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-exp", type=int, default=5)
    parser.add_argument("--end-exp", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=5e-1)
    parser.add_argument("--plateau-window", type=int, default=50_000)
    parser.add_argument("--plateau-rtol", type=float, default=1e-3)
    parser.add_argument(
        "--plot-from-json",
        type=Path,
        help="redraw the PDF and PNG from an existing JSON trace without optimization",
    )
    args = parser.parse_args()
    GAUSSIAN_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    GAUSSIAN_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    cached_payload = None
    if args.plot_from_json is not None:
        cached_payload = json.loads(args.plot_from_json.read_text())
        lambdas = [float(value) for value in cached_payload["lambdas"]]
        args.plateau_window = int(cached_payload["plateau_window"])
        args.plateau_rtol = float(cached_payload["plateau_rtol"])
    else:
        lambdas = [10.0 ** (-k) for k in range(args.start_exp, args.end_exp + 1)]
    out_fig, out_png, out_json = output_paths(lambdas)

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

    x = np.arange(-10.0, 10.0 + 0.5e-3, 1e-3)
    trap = np.full_like(x, 1e-3)
    trap[0] = trap[-1] = 0.5e-3
    target = target_fn(x)

    if cached_payload is None:
        hist32, switches32 = run_adaptive_profiled_optimizer(
            lambdas,
            max_iter=args.max_iter,
            lr=args.lr,
            plateau_window=args.plateau_window,
            plateau_rtol=args.plateau_rtol,
            precision="float32",
        )
        hist64, switches64 = run_adaptive_profiled_optimizer(
            lambdas,
            max_iter=args.max_iter,
            lr=args.lr,
            plateau_window=args.plateau_window,
            plateau_rtol=args.plateau_rtol,
            precision="float64",
        )
    else:
        hist32 = cached_payload["runs"]["float32"]["optimizer_trace"]
        hist64 = cached_payload["runs"]["float64"]["optimizer_trace"]
        switches32 = cached_payload["runs"]["float32"]["switches"]
        switches64 = cached_payload["runs"]["float64"]["switches"]

    def unpack(hist: dict[str, list[float]]) -> tuple[np.ndarray, ...]:
        return (
            np.asarray(hist["iteration"], dtype=float),
            np.asarray(hist["lambda"], dtype=float),
            np.asarray(hist["separation"], dtype=float),
            np.asarray(hist["amplitude"], dtype=float),
            np.asarray(hist["weight_norm"], dtype=float),
            np.asarray(hist["grad_norm"], dtype=float),
        )

    iters32, lam32, sep32, amp32, weight32, grad32 = unpack(hist32)
    iters64, lam64, sep64, amp64, weight64, grad64 = unpack(hist64)
    err64_on32, err32_on32 = observed_training_errors(hist32, x, target, trap)
    err64_on64, err32_on64 = observed_training_errors(hist64, x, target, trap)
    best32_idx = int(np.argmin(err32_on32))
    best64_idx = int(np.argmin(err64_on64))

    blue = "#0000FF"
    red = "#FF0000"
    norm_color = "0.25"
    marker_gray = "0.45"

    def clipped(error: np.ndarray) -> np.ndarray:
        return np.clip(np.maximum(np.asarray(error, dtype=float), ERROR_FLOOR), ERROR_FLOOR, ERROR_CAP)

    def draw_switches(ax: plt.Axes, switches: list[dict[str, float]], alpha: float = 0.55) -> None:
        for switch in switches:
            ax.axvline(
                switch["iteration"],
                color=marker_gray,
                lw=0.55,
                ls=(0, (1.5, 1.5)),
                alpha=alpha,
            )

    def plot_precision_panel(
        ax: plt.Axes,
        iters: np.ndarray,
        error: np.ndarray,
        weight: np.ndarray,
        switches: list[dict[str, float]],
        best_idx: int,
        error_color: str,
        title: str,
        error_label: str,
        show_legend: bool,
    ) -> None:
        err_line, = ax.semilogy(
            iters,
            clipped(error),
            color=error_color,
            lw=2.0,
            ls="--" if "single" in error_label else "-",
            marker="s" if "single" in error_label else "o",
            ms=3.0,
            markevery=max(1, len(iters) // 12),
            label=error_label,
        )
        draw_switches(ax, switches)
        ax.axvline(iters[best_idx], color=marker_gray, lw=0.8, ls=(0, (1.5, 1.5)))
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"relative $L^2$ error", color=error_color, labelpad=4)
        ax.tick_params(axis="y", labelcolor=error_color)
        ax.set_ylim(1e-8, ERROR_CAP)
        ax2 = ax.twinx()
        norm_line, = ax2.plot(
            iters,
            weight,
            color=norm_color,
            lw=1.8,
            ls=":",
            marker="x",
            ms=3.0,
            markevery=max(1, len(iters) // 12),
            label=r"$\|w\|_2$",
        )
        ax2.set_ylabel(r"$\|w\|_2$", color=norm_color, labelpad=2)
        ax2.tick_params(axis="y", labelcolor=norm_color)
        ax.set_title(title)
        if show_legend:
            ax.legend(
                [err_line, norm_line],
                [err_line.get_label(), norm_line.get_label()],
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.03),
                ncol=2,
                columnspacing=1.0,
                handlelength=2.2,
            )

    fig, axes = plt.subplots(3, 1, figsize=(4.685, 6.50))
    fig.subplots_adjust(left=0.17, right=0.82, bottom=0.075, top=0.96, hspace=0.46)

    ax = axes[0]
    ax.semilogy(
        iters64,
        clipped(err64_on64),
        color=blue,
        lw=2.0,
        ls="-",
        marker="o",
        ms=3.0,
        markevery=max(1, len(iters64) // 12),
        label="double-precision error",
    )
    ax.semilogy(
        iters32,
        clipped(err32_on32),
        color=red,
        lw=2.0,
        ls="--",
        marker="s",
        ms=3.0,
        markevery=max(1, len(iters32) // 12),
        label="single-precision error",
    )
    for iteration in sorted({s["iteration"] for s in switches32 + switches64}):
        ax.axvline(iteration, color=marker_gray, lw=0.55, ls=(0, (1.5, 1.5)), alpha=0.45)
    ax.set_xlim(0.0, max(float(np.max(iters32)), float(np.max(iters64))))
    ax.set_ylim(1e-8, ERROR_CAP)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(r"(a) realized error")
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)

    plot_precision_panel(
        axes[1],
        iters32,
        err32_on32,
        weight32,
        switches32,
        best32_idx,
        red,
        r"(b) single-precision Adam",
        r"single-precision error",
        False,
    )
    plot_precision_panel(
        axes[2],
        iters64,
        err64_on64,
        weight64,
        switches64,
        best64_idx,
        blue,
        r"(c) double-precision Adam",
        r"double-precision error",
        False,
    )

    fig.savefig(out_fig)
    fig.savefig(out_png, dpi=240)

    def training_trace(
        iters: np.ndarray,
        lam_hist: np.ndarray,
        err64: np.ndarray,
        err32: np.ndarray,
        separation: np.ndarray,
        amplitude: np.ndarray,
        weight_norm: np.ndarray,
        grad_norm: np.ndarray,
    ) -> list[dict[str, float]]:
        return [
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
                err64,
                err32,
                separation,
                amplitude,
                weight_norm,
                grad_norm,
            )
        ]

    def state_summary(
        idx: int,
        iters: np.ndarray,
        lam_hist: np.ndarray,
        err64: np.ndarray,
        err32: np.ndarray,
        separation: np.ndarray,
        amplitude: np.ndarray,
        weight_norm: np.ndarray,
        grad_norm: np.ndarray,
    ) -> dict[str, float]:
        return {
            "iteration": float(iters[idx]),
            "lambda": float(lam_hist[idx]),
            "rel_error_float64": float(err64[idx]),
            "rel_error_float32": float(err32[idx]),
            "separation": float(separation[idx]),
            "amplitude": float(amplitude[idx]),
            "weight_norm": float(weight_norm[idx]),
            "grad_norm": float(grad_norm[idx]),
        }

    trace32 = training_trace(iters32, lam32, err64_on32, err32_on32, sep32, amp32, weight32, grad32)
    trace64 = training_trace(iters64, lam64, err64_on64, err32_on64, sep64, amp64, weight64, grad64)
    run32_payload = {
        "precision": "float32",
        "switches": switches32,
        "optimizer_trace": hist32,
        "training_precision_trace": trace32,
        "best_displayed": state_summary(
            best32_idx, iters32, lam32, err64_on32, err32_on32, sep32, amp32, weight32, grad32
        ),
        "best_float64": state_summary(
            int(np.argmin(err64_on32)), iters32, lam32, err64_on32, err32_on32, sep32, amp32, weight32, grad32
        ),
        "best_float32": state_summary(
            best32_idx, iters32, lam32, err64_on32, err32_on32, sep32, amp32, weight32, grad32
        ),
        "final": state_summary(-1, iters32, lam32, err64_on32, err32_on32, sep32, amp32, weight32, grad32),
    }
    run64_payload = {
        "precision": "float64",
        "switches": switches64,
        "optimizer_trace": hist64,
        "training_precision_trace": trace64,
        "best_displayed": state_summary(
            best64_idx, iters64, lam64, err64_on64, err32_on64, sep64, amp64, weight64, grad64
        ),
        "best_float64": state_summary(
            best64_idx, iters64, lam64, err64_on64, err32_on64, sep64, amp64, weight64, grad64
        ),
        "best_float32": state_summary(
            int(np.argmin(err32_on64)), iters64, lam64, err64_on64, err32_on64, sep64, amp64, weight64, grad64
        ),
        "final": state_summary(-1, iters64, lam64, err64_on64, err32_on64, sep64, amp64, weight64, grad64),
    }
    payload = {
        "model": "a * (K(x-h/2)-K(x+h/2)), K(x)=(2pi)^(-1/2) exp(-x^2/2)",
        "target": "x K(x) = d_c K(x-c)|_{c=0}",
        "q_init": Q_INIT,
        "penalty": "adaptive lambda * ||a||, where a is the signed amplitude vector",
        "lambdas": lambdas,
        "plateau_window": args.plateau_window,
        "plateau_rtol": args.plateau_rtol,
        "switches": switches64,
        "optimizer_trace": hist64,
        "training_precision_trace": trace64,
        "best_float64": run64_payload["best_float64"],
        "final": run64_payload["final"],
        "runs": {
            "float32": run32_payload,
            "float64": run64_payload,
        },
    }
    if cached_payload is None:
        out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {out_fig}")
    print(f"Wrote {out_png}")
    if cached_payload is None:
        print(f"Wrote {out_json}")
    else:
        print(f"Redrew from {args.plot_from_json}; preserved cached JSON")
    for precision, switches in (("float32", switches32), ("float64", switches64)):
        print(f"Switches ({precision}):")
        for switch in switches:
            print(
                f"  iter={switch['iteration']:.0f}: "
                f"{switch['from_lambda']:.0e} -> {switch['to_lambda']:.0e}, "
                f"rel_error={switch['relative_error']:.3e}, "
                f"||a||={switch['weight_norm']:.3e}"
            )
    print(
        "Best float32 state: "
        f"iter={iters32[best32_idx]:.0f}, lambda={lam32[best32_idx]:.0e}, "
        f"rel_error={err32_on32[best32_idx]:.3e}, "
        f"delta={sep32[best32_idx]:.3e}, |a|={amp32[best32_idx]:.3e}, "
        f"||a||={weight32[best32_idx]:.3e}"
    )
    print(
        "Best float64 state: "
        f"iter={iters64[best64_idx]:.0f}, lambda={lam64[best64_idx]:.0e}, "
        f"rel_error={err64_on64[best64_idx]:.3e}, "
        f"delta={sep64[best64_idx]:.3e}, |a|={amp64[best64_idx]:.3e}, "
        f"||a||={weight64[best64_idx]:.3e}"
    )
    print(
        "Final float32 optimizer state: "
        f"lambda={lam32[-1]:.0e}, rel_error={err32_on32[-1]:.3e}, "
        f"delta={sep32[-1]:.3e}, |a|={amp32[-1]:.3e}, "
        f"||a||={weight32[-1]:.3e}, grad={grad32[-1]:.3e}"
    )
    print(
        "Final float64 optimizer state: "
        f"lambda={lam64[-1]:.0e}, rel_error={err64_on64[-1]:.3e}, "
        f"delta={sep64[-1]:.3e}, |a|={amp64[-1]:.3e}, "
        f"||a||={weight64[-1]:.3e}, grad={grad64[-1]:.3e}"
    )


if __name__ == "__main__":
    main()
