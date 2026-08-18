"""Loss history for the two-neuron tanh parameter-escape trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from artifact_paths import TANH_DATA_DIRECTORY, experiment_output_directory


OUTPUT_DIRECTORY = experiment_output_directory("tanh")


def output_paths(dtype_name: str) -> tuple[Path, Path, Path]:
    suffix = "" if dtype_name == "float64" else f"_{dtype_name}"
    return (
        OUTPUT_DIRECTORY / f"tanh_parameter_escape_loss{suffix}.pdf",
        OUTPUT_DIRECTORY / f"tanh_parameter_escape_loss{suffix}.png",
        TANH_DATA_DIRECTORY / f"tanh_parameter_escape_loss{suffix}.json",
    )


def sech2_np(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def u_true_torch(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(x) ** 2


def f_true_torch(x: torch.Tensor) -> torch.Tensor:
    u = u_true_torch(x)
    return -3.0 * u + 6.0 * u * u


def nn_u(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    return w1 * torch.tanh(x - b1) + w2 * torch.tanh(x - b2)


def trapezoid_weights_torch(x: torch.Tensor) -> torch.Tensor:
    dx = float((x[1] - x[0]).detach())
    weights = torch.ones_like(x) * dx
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


def training_energy(
    radius: float,
    n_points: int,
    dtype: torch.dtype,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    x = torch.linspace(-radius, radius, n_points, dtype=dtype, requires_grad=True)
    u = nn_u(x, w1, b1, w2, b2)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    f = f_true_torch(x.detach())
    weights = trapezoid_weights_torch(x)
    integrand = 0.5 * (ux * ux + u * u) - f * u
    return (2.0 * radius) * torch.sum(integrand * weights)


def diagnostic_energy_np(radius: float, n_points: int, w1: float, b1: float, w2: float, b2: float) -> float:
    x = np.linspace(-radius, radius, n_points)
    dx = x[1] - x[0]
    weights = np.full_like(x, dx)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    s1 = sech2_np(x - b1)
    s2 = sech2_np(x - b2)
    u = w1 * np.tanh(x - b1) + w2 * np.tanh(x - b2)
    ux = w1 * s1 + w2 * s2
    u_star = sech2_np(x)
    f = -3.0 * u_star + 6.0 * u_star * u_star
    return float((2.0 * radius) * np.sum((0.5 * (ux * ux + u * u) - f * u) * weights))


def exact_energy_np(radius: float, n_points: int) -> float:
    x = np.linspace(-radius, radius, n_points)
    dx = x[1] - x[0]
    weights = np.full_like(x, dx)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    u = sech2_np(x)
    ux = -2.0 * u * np.tanh(x)
    f = -3.0 * u + 6.0 * u * u
    return float((2.0 * radius) * np.sum((0.5 * (ux * ux + u * u) - f * u) * weights))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    args = parser.parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TANH_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    out_pdf, out_png, out_json = output_paths(args.dtype)

    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    radius = 10.0
    n_adam_quad = 1001
    n_eval_quad = 2001
    epochs = 8000
    snap_every = 10
    eps = 1.0

    w1 = torch.nn.Parameter(torch.tensor(1.0 / (2.0 * eps) + 3.0, dtype=dtype))
    w2 = torch.nn.Parameter(torch.tensor(-1.0 / (2.0 * eps) + 2.0, dtype=dtype))
    b1 = torch.nn.Parameter(torch.tensor(eps, dtype=dtype))
    b2 = torch.nn.Parameter(torch.tensor(-eps, dtype=dtype))
    params = [w1, b1, w2, b2]

    j_star = exact_energy_np(radius, n_eval_quad)
    history: list[dict[str, float | str]] = []

    def record(iteration: int, phase: str) -> None:
        value = diagnostic_energy_np(radius, n_eval_quad, w1.item(), b1.item(), w2.item(), b2.item())
        history.append(
            {
                "iteration": float(iteration),
                "phase": phase,
                "energy": value,
                "energy_gap": max(value - j_star, 1e-16),
                "w1": float(w1.item()),
                "b1": float(b1.item()),
                "w2": float(w2.item()),
                "b2": float(b2.item()),
            }
        )

    record(0, "initial")
    opt = torch.optim.Adam(params, lr=1e-3)
    for it in range(1, epochs + 1):
        opt.zero_grad()
        loss = training_energy(radius, n_adam_quad, dtype, w1, b1, w2, b2)
        loss.backward()
        opt.step()
        if it % snap_every == 0:
            record(it, "adam")

    opt2 = torch.optim.LBFGS(params, lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
    lbfgs_count = 0

    def closure() -> torch.Tensor:
        nonlocal lbfgs_count
        opt2.zero_grad()
        loss = training_energy(radius, n_eval_quad, dtype, w1, b1, w2, b2)
        loss.backward()
        lbfgs_count += 1
        record(epochs + lbfgs_count, "lbfgs")
        return loss

    opt2.step(closure)
    record(epochs + lbfgs_count, "final")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.labelsize": 9.0,
        }
    )

    iters = np.asarray([row["iteration"] for row in history], dtype=float)
    gaps = np.asarray([row["energy_gap"] for row in history], dtype=float)
    phases = np.asarray([row["phase"] for row in history])
    adam_mask = np.isin(phases, ["initial", "adam"])
    lbfgs_mask = np.isin(phases, ["lbfgs", "final"])

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.semilogy(iters[adam_mask], gaps[adam_mask], color="#0000FF", lw=1.7, label="Adam")
    ax.semilogy(iters[lbfgs_mask], gaps[lbfgs_mask], color="#FF0000", lw=1.45, label="LBFGS")
    ax.axvline(epochs, color="0.45", lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$J(u_\theta)-J(u^\ast)$")
    ax.set_title("loss gap")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")

    out_json.write_text(
        json.dumps(
            {
                "target": "sech^2(x), matching wb_trajectories.pdf",
                "dtype": args.dtype,
                "exact_energy": j_star,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")
    print(f"Final loss gap: {gaps[-1]:.3e}")


if __name__ == "__main__":
    main()
