"""Reproduce the four-state tanh slope-collision figure and JSON snapshots.

This is the script form of the former notebook-only calculation.  The two
LBFGS columns are independent continuations from the same 5,000-step Adam
endpoint.  No random quantities or minibatches are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
FIGURE_OUTPUT = WORKSPACE / "figures" / "descriptive"
STEM = "tanh_x_tanhprime_energy_minimization"
RADIUS = 10.0
ADAM_STEPS = 5_000
ADAM_LR = 1e-3
LBFGS_BUDGETS = (200, 1_000)
ENERGY_NODES_ADAM = 1_001
ENERGY_NODES_LBFGS = 2_001
DIAGNOSTIC_NODES = 4_001


def trapezoid_weights(x: torch.Tensor) -> torch.Tensor:
    weights = torch.full_like(x, float(x[1] - x[0]))
    weights[[0, -1]] *= 0.5
    return weights


def target(x: torch.Tensor) -> torch.Tensor:
    return x / torch.cosh(x) ** 2


def target_derivative(x: torch.Tensor) -> torch.Tensor:
    sech2 = 1.0 / torch.cosh(x) ** 2
    return sech2 - 2.0 * x * sech2 * torch.tanh(x)


def forcing(x: torch.Tensor) -> torch.Tensor:
    sech2 = 1.0 / torch.cosh(x) ** 2
    tanh_x = torch.tanh(x)
    return 4.0 * sech2 * tanh_x - 3.0 * x * sech2 + 6.0 * x * sech2 * sech2


def represented(x: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    w1, a1, b1, w2, a2, b2 = params
    return w1 * torch.tanh(a1 * x + b1) + w2 * torch.tanh(a2 * x + b2)


def represented_derivative(
    x: torch.Tensor, params: list[torch.nn.Parameter]
) -> torch.Tensor:
    w1, a1, b1, w2, a2, b2 = params
    z1, z2 = a1 * x + b1, a2 * x + b2
    return w1 * a1 * (1.0 - torch.tanh(z1) ** 2) + w2 * a2 * (
        1.0 - torch.tanh(z2) ** 2
    )


def energy(params: list[torch.nn.Parameter], node_count: int) -> torch.Tensor:
    x = torch.linspace(-RADIUS, RADIUS, node_count, dtype=torch.float64, requires_grad=True)
    u = represented(x, params)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    integrand = 0.5 * (ux * ux + u * u) - forcing(x.detach()) * u
    # Preserve the positive notebook scaling; it does not change stationary points.
    return (2.0 * RADIUS) * torch.sum(trapezoid_weights(x.detach()) * integrand)


def make_params() -> list[torch.nn.Parameter]:
    return [
        torch.nn.Parameter(torch.tensor(3.5, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)),
    ]


def clone_state(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in params]


def restore_state(params: list[torch.nn.Parameter], state: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(params, state):
            parameter.copy_(value)


def diagnostics(
    label: str,
    optimizer: str,
    budget: int,
    params: list[torch.nn.Parameter],
) -> dict[str, object]:
    x = torch.linspace(-RADIUS, RADIUS, DIAGNOSTIC_NODES, dtype=torch.float64)
    weights = trapezoid_weights(x)
    with torch.no_grad():
        truth = target(x)
        truth_x = target_derivative(x)
        approximation = represented(x, params)
        approximation_x = represented_derivative(x, params)
        relative_l2 = torch.sqrt(torch.sum(weights * (approximation - truth) ** 2)) / torch.sqrt(
            torch.sum(weights * truth**2)
        )
        relative_h1 = torch.sqrt(
            torch.sum(weights * ((approximation - truth) ** 2 + (approximation_x - truth_x) ** 2))
        ) / torch.sqrt(torch.sum(weights * (truth**2 + truth_x**2)))
        w1, a1, b1, w2, a2, b2 = params
        # tanh(-z)=-tanh(z): normalize the second slope to be positive before
        # measuring the collision in (slope,bias) feature space.
        normalized_a2 = torch.abs(a2)
        normalized_b2 = torch.where(a2 < 0.0, -b2, b2)
        sign_normalized_feature_separation = torch.sqrt(
            (a1 - normalized_a2) ** 2 + (b1 - normalized_b2) ** 2
        )
        return {
            "label": label,
            "optimizer": optimizer,
            "budget": budget,
            "parameters": {
                "w1": float(w1),
                "a1": float(a1),
                "b1": float(b1),
                "w2": float(w2),
                "a2": float(a2),
                "b2": float(b2),
            },
            "coefficient_norm": float(torch.sqrt(w1 * w1 + w2 * w2)),
            "sign_normalized_feature_separation": float(sign_normalized_feature_separation),
            "relative_l2_error": float(relative_l2),
            "relative_h1_error": float(relative_h1),
        }


def run() -> list[dict[str, object]]:
    torch.set_num_threads(1)
    params = make_params()
    optimizer = torch.optim.Adam(params, lr=ADAM_LR)
    snapshots: list[dict[str, object]] = []
    for iteration in range(1, ADAM_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = energy(params, ENERGY_NODES_ADAM)
        loss.backward()
        optimizer.step()
        if iteration == ADAM_STEPS // 3:
            snapshots.append(diagnostics("Adam 1,666", "Adam", iteration, params))
    snapshots.append(diagnostics("Adam 5,000", "Adam", ADAM_STEPS, params))
    adam_endpoint = clone_state(params)

    for budget in LBFGS_BUDGETS:
        restore_state(params, adam_endpoint)
        optimizer_lbfgs = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=budget,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            loss = energy(params, ENERGY_NODES_LBFGS)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)
        snapshots.append(diagnostics(f"LBFGS {budget:,}", "LBFGS", budget, params))
    return snapshots


def make_plot(snapshots: list[dict[str, object]], output_pdf: Path, output_png: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.0,
            "axes.titlesize": 9.4,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
        }
    )
    x = torch.linspace(-RADIUS, RADIUS, DIAGNOSTIC_NODES, dtype=torch.float64)
    x_numpy = x.numpy()
    truth = target(x).numpy()
    fig, axes = plt.subplots(4, 2, figsize=(4.685, 6.65), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.065, top=0.965, wspace=0.20, hspace=0.22)
    for snapshot_index, snapshot in enumerate(snapshots):
        group_row = 2 * (snapshot_index // 2)
        column = snapshot_index % 2
        parameter_values = snapshot["parameters"]
        w1 = float(parameter_values["w1"])
        a1 = float(parameter_values["a1"])
        b1 = float(parameter_values["b1"])
        w2 = float(parameter_values["w2"])
        a2 = float(parameter_values["a2"])
        b2 = float(parameter_values["b2"])
        neuron1 = w1 * np.tanh(a1 * x_numpy + b1)
        neuron2 = w2 * np.tanh(a2 * x_numpy + b2)
        total = neuron1 + neuron2

        top = axes[group_row, column]
        top.plot(x_numpy, neuron1, color="#1f5aa6", lw=1.25, ls="-", label="neuron 1")
        top.plot(x_numpy, neuron2, color="#b33c2e", lw=1.25, ls=(0, (3, 1.5)), label="neuron 2")
        top.axhline(0.0, color="0.35", lw=0.6, ls=":")
        top.set_ylim(-30.0, 30.0)
        top.set_title(str(snapshot["label"]))
        if column == 0:
            top.set_ylabel("weighted neurons")
        if snapshot_index == 0:
            top.legend(frameon=False, loc="upper right")

        bottom = axes[group_row + 1, column]
        bottom.plot(x_numpy, total, color="0.1", lw=1.25, ls="--", label="sum")
        bottom.plot(
            x_numpy,
            truth,
            color="#188038",
            lw=1.15,
            ls="-",
            marker="^",
            markevery=500,
            markersize=2.4,
            label="target",
        )
        bottom.axhline(0.0, color="0.35", lw=0.6, ls=":")
        bottom.set_xlim(-RADIUS, RADIUS)
        bottom.set_ylim(-1.0, 1.0)
        bottom.set_xlabel(r"$x$")
        if column == 0:
            bottom.set_ylabel("state")
        if snapshot_index == 0:
            bottom.legend(frameon=False, loc="upper right")
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync-figures",
        "--sync-paper",
        dest="sync_figures",
        action="store_true",
        help="copy the rendered PDF and PNG to figures/descriptive",
    )
    parser.add_argument(
        "--plot-from-cache",
        action="store_true",
        help="redraw the PDF/PNG from the existing JSON without rerunning optimization",
    )
    args = parser.parse_args()
    output_json = HERE / f"{STEM}.json"
    output_pdf = HERE / f"{STEM}.pdf"
    output_png = HERE / f"{STEM}.png"
    if args.plot_from_cache:
        payload = json.loads(output_json.read_text())
        snapshots = payload.get("snapshots")
        if not isinstance(snapshots, list) or len(snapshots) != 4:
            raise ValueError("cached tanh-slope JSON must contain four snapshots")
        make_plot(snapshots, output_pdf, output_png)
        if args.sync_figures:
            FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
            for source in (output_pdf, output_png):
                shutil.copy2(source, FIGURE_OUTPUT / source.name)
        print(f"Wrote {output_pdf} and {output_png} from {output_json}")
        return
    snapshots = run()
    payload = {
        "experiment": "tanh slope collision",
        "target": "x sech^2(x)",
        "model": "w1*tanh(a1*x+b1)+w2*tanh(a2*x+b2)",
        "deterministic": True,
        "random_seed": None,
        "bulk_energy_only": True,
        "domain": [-RADIUS, RADIUS],
        "optimizer": {
            "adam_steps": ADAM_STEPS,
            "adam_learning_rate": ADAM_LR,
            "adam_nodes": ENERGY_NODES_ADAM,
            "lbfgs_budgets": list(LBFGS_BUDGETS),
            "lbfgs_nodes": ENERGY_NODES_LBFGS,
            "lbfgs_line_search": "strong_wolfe",
            "note": "Each LBFGS run starts independently from the Adam endpoint.",
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": mpl.__version__,
        },
        "snapshots": snapshots,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    make_plot(snapshots, output_pdf, output_png)
    if args.sync_figures:
        FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
        for source in (output_pdf, output_png):
            shutil.copy2(source, FIGURE_OUTPUT / source.name)
    print(json.dumps(snapshots, indent=2))


if __name__ == "__main__":
    main()
