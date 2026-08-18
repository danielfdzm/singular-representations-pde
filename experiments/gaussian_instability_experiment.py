"""Gaussian cancellation diagnostics for parameter blow-up.

This experiment quantifies two practical consequences of the two-center
Gaussian collapse sequence:

1. Finite precision: the exact functions converge to G', but the computation
   becomes a subtraction of two large nearly equal terms.
2. Optimization dynamics: gradient-based training moves in parameter space and
   can keep drifting along the non-attained valley after the represented
   function has already stabilized.
"""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from artifact_paths import experiment_output_directory


GAUSSIAN_OUTPUT_DIRECTORY = experiment_output_directory("gaussian")
OUT_FIG = GAUSSIAN_OUTPUT_DIRECTORY / "gaussian_instability_diagnostics.pdf"
OUT_PNG = GAUSSIAN_OUTPUT_DIRECTORY / "gaussian_instability_diagnostics.png"
OUT_JSON = GAUSSIAN_OUTPUT_DIRECTORY / "gaussian_instability_diagnostics.json"
OUT_TABLE = GAUSSIAN_OUTPUT_DIRECTORY / "gaussian_instability_table.tex"

SQRT_2PI = np.sqrt(2.0 * np.pi)
GRAM_C = 1.0 / (2.0 * np.sqrt(np.pi))
TARGET_NORM_SQ = GRAM_C / 2.0


def gaussian(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / SQRT_2PI


def gaussian_prime(x: np.ndarray) -> np.ndarray:
    return -x * gaussian(x)


def u_eps_float64(x: np.ndarray, eps: float) -> np.ndarray:
    return (gaussian(x + eps) - gaussian(x - eps)) / (2.0 * eps)


def u_eps_float32(x: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    eps32 = np.float32(eps)
    c32 = np.float32(1.0 / np.sqrt(2.0 * np.pi))
    left = np.exp(np.float32(-0.5) * (x32 + eps32) * (x32 + eps32), dtype=np.float32) * c32
    right = np.exp(np.float32(-0.5) * (x32 - eps32) * (x32 - eps32), dtype=np.float32) * c32
    return ((left - right) / (np.float32(2.0) * eps32)).astype(np.float64)


def antisymmetric_model_float64(x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    return amplitude * (gaussian(x + h) - gaussian(x - h))


def antisymmetric_model_float32(x: np.ndarray, h: float, amplitude: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    h32 = np.float32(h)
    amp32 = np.float32(amplitude)
    c32 = np.float32(1.0 / np.sqrt(2.0 * np.pi))
    left = np.exp(np.float32(-0.5) * (x32 + h32) * (x32 + h32), dtype=np.float32) * c32
    right = np.exp(np.float32(-0.5) * (x32 - h32) * (x32 - h32), dtype=np.float32) * c32
    return (amp32 * (left - right)).astype(np.float64)


def relative_l2_error(u: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    num = np.sqrt(float(np.sum(weights * (u - target) ** 2)))
    den = np.sqrt(float(np.sum(weights * target**2)))
    return num / den


def gram_condition(eps: np.ndarray | float) -> np.ndarray | float:
    eps_arr = np.asarray(eps, dtype=np.float64)
    rho = np.exp(-(eps_arr**2))
    denom = -np.expm1(-(eps_arr**2))
    value = (1.0 + rho) / denom
    if np.isscalar(eps):
        return float(value)
    return value


def best_amplitude(h: np.ndarray | float) -> np.ndarray | float:
    h_arr = np.asarray(h, dtype=np.float64)
    a = h_arr * np.exp(-(h_arr**2) / 4.0) / (2.0 * (-np.expm1(-(h_arr**2))))
    if np.isscalar(h):
        return float(a)
    return a


def analytic_loss_and_grad(a: float, q: float) -> tuple[float, float, np.ndarray, float, float, float]:
    h = float(np.exp(q))
    e_h2 = float(np.exp(-(h * h)))
    e_h2_quarter = float(np.exp(-(h * h) / 4.0))

    feature_norm_sq = 2.0 * GRAM_C * (-np.expm1(-(h * h)))
    feature_target_ip = h * GRAM_C * e_h2_quarter
    loss = 0.5 * (a * a * feature_norm_sq - 2.0 * a * feature_target_ip + TARGET_NORM_SQ)

    d_feature_norm_sq = 4.0 * GRAM_C * h * e_h2
    d_feature_target_ip = GRAM_C * e_h2_quarter * (1.0 - 0.5 * h * h)

    grad_a = a * feature_norm_sq - feature_target_ip
    grad_h = 0.5 * a * a * d_feature_norm_sq - a * d_feature_target_ip
    grad_q = h * grad_h
    rel_error = np.sqrt(max(2.0 * loss, 0.0) / TARGET_NORM_SQ)
    return loss, rel_error, np.array([grad_a, grad_q], dtype=np.float64), h, feature_norm_sq, feature_target_ip


def reduced_loss_and_grad(q: float) -> tuple[float, float, float, float, float]:
    h = float(np.exp(q))
    feature_norm_sq = 2.0 * GRAM_C * (-np.expm1(-(h * h)))
    feature_target_ip = h * GRAM_C * np.exp(-(h * h) / 4.0)
    amplitude = feature_target_ip / feature_norm_sq

    d_feature_norm_sq = 4.0 * GRAM_C * h * np.exp(-(h * h))
    d_feature_target_ip = GRAM_C * np.exp(-(h * h) / 4.0) * (1.0 - 0.5 * h * h)

    loss = 0.5 * (TARGET_NORM_SQ - (feature_target_ip * feature_target_ip) / feature_norm_sq)
    grad_h = (
        -feature_target_ip * d_feature_target_ip / feature_norm_sq
        + 0.5 * feature_target_ip * feature_target_ip * d_feature_norm_sq / (feature_norm_sq * feature_norm_sq)
    )
    grad_q = h * grad_h
    rel_error = np.sqrt(max(2.0 * loss, 0.0) / TARGET_NORM_SQ)
    return loss, rel_error, grad_q, h, amplitude


def run_optimizer(max_iter: int = 1_000_000, lr: float = 5e-2) -> dict[str, list[float]]:
    a = 0.5
    q = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    m = np.zeros(2, dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)

    record_iters = np.unique(
        np.concatenate(
            [
                np.arange(0, 20_001, 500, dtype=int),
                np.arange(25_000, 100_001, 5_000, dtype=int),
                np.arange(125_000, max_iter + 1, 25_000, dtype=int),
            ]
        )
    )
    record_iters = set(int(i) for i in record_iters if i <= max_iter)

    hist: dict[str, list[float]] = {
        "iteration": [],
        "relative_error": [],
        "separation": [],
        "amplitude": [],
        "weight_norm": [],
        "theta_norm": [],
        "grad_norm": [],
    }

    for it in range(max_iter + 1):
        loss, rel_error, grad, h, _, _ = analytic_loss_and_grad(a, q)

        if it in record_iters or it == max_iter:
            hist["iteration"].append(float(it))
            hist["relative_error"].append(float(rel_error))
            hist["separation"].append(float(2.0 * h))
            hist["amplitude"].append(float(a))
            hist["weight_norm"].append(float(np.sqrt(2.0) * abs(a)))
            hist["theta_norm"].append(float(np.sqrt(2.0 * a * a + 2.0 * h * h)))
            hist["grad_norm"].append(float(np.linalg.norm(grad)))

        if it == max_iter:
            break

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1 ** (it + 1))
        v_hat = v / (1.0 - beta2 ** (it + 1))
        step = lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        a -= float(step[0])
        q -= float(step[1])

    return hist


def run_profiled_optimizer(max_iter: int = 100_000, lr: float = 1e-2) -> dict[str, list[float]]:
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
        "relative_error": [],
        "separation": [],
        "amplitude": [],
        "weight_norm": [],
        "theta_norm": [],
        "grad_norm": [],
    }

    for it in range(max_iter + 1):
        loss, rel_error, grad_q, h, amplitude = reduced_loss_and_grad(q)

        if it in record_iters or it == max_iter:
            hist["iteration"].append(float(it))
            hist["relative_error"].append(float(rel_error))
            hist["separation"].append(float(2.0 * h))
            hist["amplitude"].append(float(amplitude))
            hist["weight_norm"].append(float(np.sqrt(2.0) * abs(amplitude)))
            hist["theta_norm"].append(float(np.sqrt(2.0 * amplitude * amplitude + 2.0 * h * h)))
            hist["grad_norm"].append(float(abs(grad_q)))

        if it == max_iter:
            break

        m = beta1 * m + (1.0 - beta1) * grad_q
        v = beta2 * v + (1.0 - beta2) * grad_q * grad_q
        m_hat = m / (1.0 - beta1 ** (it + 1))
        v_hat = v / (1.0 - beta2 ** (it + 1))
        q -= float(lr * m_hat / (np.sqrt(v_hat) + eps_adam))

    return hist


def sci(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{digits}f}\times 10^{{{exponent}}}"


def write_table(rows: list[dict[str, float]]) -> None:
    lines = [
        r"\begin{tabular}{|c|c|c|c|c|}",
        r"\hline",
        r"$\delta$ & $\|w\|_2$ & $\kappa(G(b))$ & rel. err. float64 & rel. err. float32 \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            rf"${sci(row['delta'], 1)}$ & "
            rf"${sci(row['weight_norm'], 1)}$ & "
            rf"${sci(row['gram_condition'], 1)}$ & "
            rf"${sci(row['rel_error_float64'], 1)}$ & "
            rf"${sci(row['rel_error_float32'], 1)}$ \\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    OUT_TABLE.write_text("\n".join(lines) + "\n")


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
    GAUSSIAN_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
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

    eps_grid = np.logspace(-1.0, -9.0, 65)
    delta_grid = 2.0 * eps_grid
    err64 = np.array([relative_l2_error(u_eps_float64(x, e), target, trap) for e in eps_grid])
    err32 = np.array([relative_l2_error(u_eps_float32(x, e), target, trap) for e in eps_grid])
    cond = gram_condition(eps_grid)
    weight_norm = np.sqrt(2.0) / (2.0 * eps_grid)
    term_amp = np.array([(1.0 / (2.0 * e)) * float(np.max(np.abs(gaussian(x + e)))) for e in eps_grid])
    state_amp = np.array([float(np.max(np.abs(u_eps_float64(x, e)))) for e in eps_grid])
    cancellation_ratio = term_amp / state_amp

    table_eps = [1e-2, 1e-4, 1e-6]
    table_rows = []
    for e in table_eps:
        table_rows.append(
            {
                "epsilon": e,
                "delta": 2.0 * e,
                "weight_norm": np.sqrt(2.0) / (2.0 * e),
                "gram_condition": gram_condition(e),
                "rel_error_float64": relative_l2_error(u_eps_float64(x, e), target, trap),
                "rel_error_float32": relative_l2_error(u_eps_float32(x, e), target, trap),
            }
        )
    write_table(table_rows)

    opt_hist = run_profiled_optimizer(max_iter=1_000_000, lr=5e-1)
    iters = np.asarray(opt_hist["iteration"])
    train_err64, train_err32 = observed_training_errors(opt_hist, x, target, trap)
    opt_err = train_err64
    opt_weight = np.asarray(opt_hist["weight_norm"])
    opt_delta = np.asarray(opt_hist["separation"])
    opt_amp = np.asarray(opt_hist["amplitude"])
    float32_deterioration_iter = float(iters[int(np.argmin(train_err32))])

    blue = "#0000FF"
    red = "#FF0000"
    gray = "0.25"
    marker_gray = "0.45"

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.28, top=0.84, wspace=0.48)

    ax = axes[0]
    ax.semilogy(iters, train_err64, color=blue, lw=1.7, label="float64")
    ax.semilogy(iters, train_err32, color=red, lw=1.7, label="float32")
    ax.axvline(float32_deterioration_iter, color=marker_gray, lw=0.8, ls=(0, (1.5, 1.5)))
    ax.set_xlabel("profiled Adam iteration")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(r"(a) trained states, finite precision")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    err_line, = ax.semilogy(iters, opt_err, color=blue, lw=1.7, label=r"relative $L^2$ error")
    sep_line, = ax.semilogy(iters, opt_delta, color=gray, lw=1.5, ls=(0, (3, 2)), label=r"separation $\delta$")
    ax.axvline(float32_deterioration_iter, color=marker_gray, lw=0.8, ls=(0, (1.5, 1.5)))
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
    ax.tick_params(axis="y")
    ax2 = ax.twinx()
    norm_line, = ax2.plot(iters, opt_weight, color=red, lw=1.7, label=r"$\|w\|_2$")
    ax2.set_ylabel(r"$\|w\|_2$", color=red)
    ax2.tick_params(axis="y", labelcolor=red)
    ax.set_title("(b) function settles, parameters drift")
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

    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")

    payload = {
        "finite_precision_sweep": [
            {
                "epsilon": float(e),
                "delta": float(d),
                "rel_error_float64": float(e64),
                "rel_error_float32": float(e32),
                "gram_condition": float(k),
                "weight_norm": float(wn),
                "cancellation_ratio": float(cr),
            }
            for e, d, e64, e32, k, wn, cr in zip(
                eps_grid, delta_grid, err64, err32, cond, weight_norm, cancellation_ratio
            )
        ],
        "optimizer_trace": opt_hist,
        "float32_deterioration_iteration": float32_deterioration_iter,
        "training_precision_trace": [
            {
                "iteration": float(it),
                "rel_error_float64": float(e64),
                "rel_error_float32": float(e32),
                "separation": float(sep),
                "weight_norm": float(wn),
            }
            for it, e64, e32, sep, wn in zip(iters, train_err64, train_err32, opt_delta, opt_weight)
        ],
        "table_rows": table_rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_TABLE}")
    print(f"Wrote {OUT_JSON}")
    print(
        "Final optimizer state: "
        f"rel_error={opt_err[-1]:.3e}, delta={opt_delta[-1]:.3e}, "
        f"|w|={opt_amp[-1]:.3e}, ||w||={opt_weight[-1]:.3e}"
    )


if __name__ == "__main__":
    main()
