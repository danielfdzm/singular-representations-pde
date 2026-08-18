"""Penalized variant of the Gaussian cancellation diagnostic.

This runs the same profiled two-Gaussian experiment as
gaussian_instability_experiment.py, but adds lambda * ||w||_2 to the profiled
objective with lambda = 1e-3.  In the antisymmetric model the weights are
(a, -a), so ||w||_2 = sqrt(2) |a|.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from artifact_paths import experiment_output_directory
from gaussian_instability_experiment import (
    GRAM_C,
    TARGET_NORM_SQ,
    antisymmetric_model_float32,
    antisymmetric_model_float64,
    gaussian_prime,
    relative_l2_error,
)


DEFAULT_LAMBDA = 1e-3
GAUSSIAN_OUTPUT_DIRECTORY = experiment_output_directory("gaussian")


def lambda_suffix(lam: float) -> str:
    return f"{lam:.0e}".replace("+0", "").replace("-0", "-")


def lambda_latex(lam: float) -> str:
    exponent = int(np.floor(np.log10(abs(lam))))
    mantissa = lam / (10.0**exponent)
    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:.1g}\times 10^{{{exponent}}}"


def output_paths(lam: float, threshold_weight_norm: float | None = None) -> tuple[Path, Path, Path]:
    suffix = lambda_suffix(lam)
    stem = f"gaussian_instability_weight_penalty_lambda_{suffix}"
    if threshold_weight_norm is not None:
        threshold_suffix = f"{threshold_weight_norm:g}".replace(".", "p")
        stem += f"_threshold_{threshold_suffix}"
    return (
        GAUSSIAN_OUTPUT_DIRECTORY / f"{stem}.pdf",
        GAUSSIAN_OUTPUT_DIRECTORY / f"{stem}.png",
        GAUSSIAN_OUTPUT_DIRECTORY / f"{stem}.json",
    )


def penalized_reduced_loss_and_grad(
    q: float,
    lam: float = DEFAULT_LAMBDA,
    threshold_weight_norm: float | None = None,
) -> tuple[float, float, float, float, float]:
    h = float(np.exp(q))
    feature_norm_sq = 2.0 * GRAM_C * (-np.expm1(-(h * h)))
    feature_target_ip = h * GRAM_C * np.exp(-(h * h) / 4.0)
    penalty_slope = lam * np.sqrt(2.0)
    unpenalized_amplitude = feature_target_ip / feature_norm_sq

    if threshold_weight_norm is not None:
        threshold_amplitude = threshold_weight_norm / np.sqrt(2.0)
        if unpenalized_amplitude <= threshold_amplitude:
            amplitude = unpenalized_amplitude
            d_feature_norm_sq = 4.0 * GRAM_C * h * np.exp(-(h * h))
            d_feature_target_ip = GRAM_C * np.exp(-(h * h) / 4.0) * (1.0 - 0.5 * h * h)
            loss = 0.5 * (TARGET_NORM_SQ - (feature_target_ip * feature_target_ip) / feature_norm_sq)
            grad_h = (
                -feature_target_ip * d_feature_target_ip / feature_norm_sq
                + 0.5
                * feature_target_ip
                * feature_target_ip
                * d_feature_norm_sq
                / (feature_norm_sq * feature_norm_sq)
            )
            grad_q = h * grad_h
            rel_error = np.sqrt(max(2.0 * loss, 0.0) / TARGET_NORM_SQ)
            return loss, rel_error, grad_q, h, amplitude

        active_amplitude = (feature_target_ip - penalty_slope) / feature_norm_sq
        if active_amplitude < threshold_amplitude:
            amplitude = threshold_amplitude
            d_feature_norm_sq = 4.0 * GRAM_C * h * np.exp(-(h * h))
            d_feature_target_ip = GRAM_C * np.exp(-(h * h) / 4.0) * (1.0 - 0.5 * h * h)
            loss = 0.5 * (
                amplitude * amplitude * feature_norm_sq
                - 2.0 * amplitude * feature_target_ip
                + TARGET_NORM_SQ
            )
            grad_h = 0.5 * amplitude * amplitude * d_feature_norm_sq - amplitude * d_feature_target_ip
            grad_q = h * grad_h
            rel_error = np.sqrt(max(2.0 * loss, 0.0) / TARGET_NORM_SQ)
            return loss, rel_error, grad_q, h, amplitude

    shrink = feature_target_ip - penalty_slope

    if shrink <= 0.0:
        return 0.5 * TARGET_NORM_SQ, 1.0, 0.0, h, 0.0

    d_feature_norm_sq = 4.0 * GRAM_C * h * np.exp(-(h * h))
    d_feature_target_ip = GRAM_C * np.exp(-(h * h) / 4.0) * (1.0 - 0.5 * h * h)

    amplitude = shrink / feature_norm_sq
    loss = 0.5 * (TARGET_NORM_SQ - (shrink * shrink) / feature_norm_sq)
    grad_h = (
        -shrink * d_feature_target_ip / feature_norm_sq
        + 0.5 * shrink * shrink * d_feature_norm_sq / (feature_norm_sq * feature_norm_sq)
    )
    grad_q = h * grad_h
    rel_error = np.sqrt(max(2.0 * loss, 0.0) / TARGET_NORM_SQ)
    return loss, rel_error, grad_q, h, amplitude


def run_penalized_profiled_optimizer(
    max_iter: int = 1_000_000,
    lr: float = 5e-1,
    lam: float = DEFAULT_LAMBDA,
    threshold_weight_norm: float | None = None,
) -> dict[str, list[float]]:
    q = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    m = 0.0
    v = 0.0

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
        "penalized_relative_error": [],
        "separation": [],
        "amplitude": [],
        "weight_norm": [],
        "theta_norm": [],
        "grad_norm": [],
        "penalty_active": [],
    }

    for it in range(max_iter + 1):
        _, penalized_rel_error, grad_q, h, amplitude = penalized_reduced_loss_and_grad(
            q,
            lam,
            threshold_weight_norm,
        )

        if it in record_iters or it == max_iter:
            current_weight_norm = float(np.sqrt(2.0) * abs(amplitude))
            hist["iteration"].append(float(it))
            hist["penalized_relative_error"].append(float(penalized_rel_error))
            hist["separation"].append(float(2.0 * h))
            hist["amplitude"].append(float(amplitude))
            hist["weight_norm"].append(current_weight_norm)
            hist["theta_norm"].append(float(np.sqrt(2.0 * amplitude * amplitude + 2.0 * h * h)))
            hist["grad_norm"].append(float(abs(grad_q)))
            hist["penalty_active"].append(
                bool(threshold_weight_norm is None or current_weight_norm > threshold_weight_norm)
            )

        if it == max_iter:
            break

        m = beta1 * m + (1.0 - beta1) * grad_q
        v = beta2 * v + (1.0 - beta2) * grad_q * grad_q
        m_hat = m / (1.0 - beta1 ** (it + 1))
        v_hat = v / (1.0 - beta2 ** (it + 1))
        q -= float(lr * m_hat / (np.sqrt(v_hat) + eps_adam))

    return hist


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
        err64.append(relative_l2_error(antisymmetric_model_float64(x, h, a), target, weights))
        err32.append(relative_l2_error(antisymmetric_model_float32(x, h, a), target, weights))
    return np.asarray(err64), np.asarray(err32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA, help="Weight on ||w||_2 penalty.")
    parser.add_argument(
        "--threshold-weight-norm",
        type=float,
        default=None,
        help="Use lambda * max(||w||_2 - threshold, 0) instead of lambda * ||w||_2.",
    )
    parser.add_argument("--max-iter", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=5e-1)
    args = parser.parse_args()
    GAUSSIAN_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    lam = float(args.lam)
    threshold_weight_norm = None if args.threshold_weight_norm is None else float(args.threshold_weight_norm)
    out_fig, out_png, out_json = output_paths(lam, threshold_weight_norm)

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

    x = np.arange(-10.0, 10.0 + 0.5e-3, 1e-3)
    trap = np.full_like(x, 1e-3)
    trap[0] = trap[-1] = 0.5e-3
    target = gaussian_prime(x)

    opt_hist = run_penalized_profiled_optimizer(
        max_iter=args.max_iter,
        lr=args.lr,
        lam=lam,
        threshold_weight_norm=threshold_weight_norm,
    )
    iters = np.asarray(opt_hist["iteration"])
    train_err64, train_err32 = observed_training_errors(opt_hist, x, target, trap)
    opt_weight = np.asarray(opt_hist["weight_norm"])
    opt_delta = np.asarray(opt_hist["separation"])
    opt_amp = np.asarray(opt_hist["amplitude"])
    opt_grad = np.asarray(opt_hist["grad_norm"])
    best64_idx = int(np.argmin(train_err64))
    best32_idx = int(np.argmin(train_err32))

    blue = "#0000FF"
    red = "#FF0000"
    gray = "0.25"
    marker_gray = "0.45"

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.28, top=0.84, wspace=0.48)

    ax = axes[0]
    ax.semilogy(iters, train_err64, color=blue, lw=1.7, label="float64")
    ax.semilogy(iters, train_err32, color=red, lw=1.7, label="float32")
    ax.axvline(iters[best32_idx], color=marker_gray, lw=0.8, ls=(0, (1.5, 1.5)))
    ax.set_xlabel("profiled Adam iteration")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(r"(a) weight-penalized trained states")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    err_line, = ax.semilogy(iters, train_err64, color=blue, lw=1.7, label=r"relative $L^2$ error")
    sep_line, = ax.semilogy(iters, opt_delta, color=gray, lw=1.5, ls=(0, (3, 2)), label=r"separation $\delta$")
    ax.axvline(iters[best64_idx], color=marker_gray, lw=0.8, ls=(0, (1.5, 1.5)))
    ax.set_xlabel("profiled Adam iteration")
    ax.set_ylabel("")
    ax.text(
        -0.19,
        0.5,
        r"relative $L^2$ error",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=mpl.rcParams["axes.labelsize"],
        color=blue,
    )
    ax.text(
        -0.13,
        0.5,
        "and separation",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=mpl.rcParams["axes.labelsize"],
        color="black",
    )
    ax2 = ax.twinx()
    norm_line, = ax2.plot(iters, opt_weight, color=red, lw=1.7, label=r"$\|w\|_2$")
    ax2.set_ylabel(r"$\|w\|_2$", color=red)
    ax2.tick_params(axis="y", labelcolor=red)
    if threshold_weight_norm is None:
        title = rf"(b) $\lambda\|w\|_2$, $\lambda={lambda_latex(lam)}$"
    else:
        title = (
            rf"(b) $\lambda(\|w\|_2-{threshold_weight_norm:g})_+$, "
            rf"$\lambda={lambda_latex(lam)}$"
        )
    ax.set_title(title)
    ax.legend(
        [err_line, sep_line, norm_line],
        [err_line.get_label(), sep_line.get_label(), norm_line.get_label()],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
        ncol=3,
        columnspacing=1.0,
        handlelength=2.2,
    )

    fig.savefig(out_fig, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")

    payload = {
        "lambda": lam,
        "penalty": (
            "lambda * ||w||_2"
            if threshold_weight_norm is None
            else "lambda * max(||w||_2 - threshold_weight_norm, 0)"
        ),
        "threshold_weight_norm": threshold_weight_norm,
        "optimizer_trace": opt_hist,
        "training_precision_trace": [
            {
                "iteration": float(it),
                "rel_error_float64": float(e64),
                "rel_error_float32": float(e32),
                "separation": float(sep),
                "amplitude": float(amp),
                "weight_norm": float(wn),
                "grad_norm": float(gn),
                "penalty_active": bool(active),
            }
            for it, e64, e32, sep, amp, wn, gn, active in zip(
                iters,
                train_err64,
                train_err32,
                opt_delta,
                opt_amp,
                opt_weight,
                opt_grad,
                opt_hist["penalty_active"],
            )
        ],
        "best_float64": {
            "iteration": float(iters[best64_idx]),
            "rel_error_float64": float(train_err64[best64_idx]),
            "rel_error_float32": float(train_err32[best64_idx]),
            "separation": float(opt_delta[best64_idx]),
            "amplitude": float(opt_amp[best64_idx]),
            "weight_norm": float(opt_weight[best64_idx]),
            "penalty_active": bool(opt_hist["penalty_active"][best64_idx]),
        },
        "final": {
            "iteration": float(iters[-1]),
            "rel_error_float64": float(train_err64[-1]),
            "rel_error_float32": float(train_err32[-1]),
            "separation": float(opt_delta[-1]),
            "amplitude": float(opt_amp[-1]),
            "weight_norm": float(opt_weight[-1]),
            "grad_norm": float(opt_grad[-1]),
            "penalty_active": bool(opt_hist["penalty_active"][-1]),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {out_fig}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")
    print(
        "Best float64 state: "
        f"iter={iters[best64_idx]:.0f}, rel_error={train_err64[best64_idx]:.3e}, "
        f"delta={opt_delta[best64_idx]:.3e}, |w|={opt_amp[best64_idx]:.3e}, "
        f"||w||={opt_weight[best64_idx]:.3e}"
    )
    print(
        "Final optimizer state: "
        f"rel_error={train_err64[-1]:.3e}, delta={opt_delta[-1]:.3e}, "
        f"|w|={opt_amp[-1]:.3e}, ||w||={opt_weight[-1]:.3e}, "
        f"grad={opt_grad[-1]:.3e}"
    )


if __name__ == "__main__":
    main()
