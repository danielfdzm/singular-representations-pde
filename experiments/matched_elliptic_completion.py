"""Matched weak-PDE comparison of translated and completed Gaussian coordinates.

The experiment solves the manufactured Neumann problem

    -u'' + u = f                    in (-8, 8),
     n u' = n u_*'                 at {-8, 8},

whose weak solution is ``u_* = K''' + 0.6 K(.-1.4)``.  Both nine-coordinate
dictionaries minimize the same discretized H1 energy, start from zero, and
receive the same Adam and strong-Wolfe LBFGS budgets.  The translated
dictionary uses a prescribed third-derivative finite-difference cluster; the
completed dictionary is oracle supplied and contains K''' explicitly.  This
is a coordinate-conditioning test, not an automatic collision-discovery test.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import shutil
import time

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from artifact_paths import (
    PREVIEW_FIGURE_DIRECTORY,
    WEAK_PDE_DATA_DIRECTORY,
    experiment_output_directory,
)

STEM = "matched_elliptic_completion"
OUTPUT_DIRECTORY = experiment_output_directory("weak_pde")
REFERENCE_JSON = WEAK_PDE_DATA_DIRECTORY / f"{STEM}.json"
REFERENCE_CSV = WEAK_PDE_DATA_DIRECTORY / f"{STEM}_summary.csv"

DOMAIN_RADIUS = 8.0
TRAIN_NODES = 4_001
VALIDATION_NODES = 20_001
ADAM_STEPS = 5_000
ADAM_LR = 0.1
LBFGS_STEPS = 100
CLUSTER_SCALE = 0.1


def trapezoid_grid(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-DOMAIN_RADIUS, DOMAIN_RADIUS, node_count, dtype=np.float64)
    weights = np.full(node_count, float(x[1] - x[0]), dtype=np.float64)
    weights[[0, -1]] *= 0.5
    return x, weights


def probabilists_hermite(order: int, x: np.ndarray) -> np.ndarray:
    previous = np.ones_like(x)
    if order == 0:
        return previous
    current = x.copy()
    if order == 1:
        return current
    for degree in range(1, order):
        previous, current = current, x * current - degree * previous
    return current


def gaussian_derivative(order: int, x: np.ndarray) -> np.ndarray:
    """Return d^order/dx^order exp(-x^2/2)."""
    return ((-1.0) ** order) * probabilists_hermite(order, x) * np.exp(-0.5 * x * x)


def target_fields(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = gaussian_derivative(3, x) + 0.6 * gaussian_derivative(0, x - 1.4)
    target_derivative = gaussian_derivative(4, x) + 0.6 * gaussian_derivative(1, x - 1.4)
    forcing = (
        -gaussian_derivative(5, x)
        + gaussian_derivative(3, x)
        + 0.6 * (-gaussian_derivative(2, x - 1.4) + gaussian_derivative(0, x - 1.4))
    )
    return target, target_derivative, forcing


def translated_dictionary(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    centers = np.concatenate(
        (CLUSTER_SCALE * np.asarray([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64), [1.4])
    )
    values = np.asarray([gaussian_derivative(0, x - center) for center in centers])
    derivatives = np.asarray([gaussian_derivative(1, x - center) for center in centers])
    labels = [f"K(x-{center:.1f})" for center in centers]
    return values, derivatives, labels, centers


def completed_dictionary(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], None]:
    extra_centers = np.asarray([-7.0, -5.0, -3.0, 0.0, 3.5, 5.5, 7.0])
    values = np.concatenate(
        (
            np.asarray([gaussian_derivative(3, x), gaussian_derivative(0, x - 1.4)]),
            np.asarray([gaussian_derivative(0, x - center) for center in extra_centers]),
        )
    )
    derivatives = np.concatenate(
        (
            np.asarray([gaussian_derivative(4, x), gaussian_derivative(1, x - 1.4)]),
            np.asarray([gaussian_derivative(1, x - center) for center in extra_centers]),
        )
    )
    labels = ["K'''(x)", "K(x-1.4)"] + [f"K(x-{center:.1f})" for center in extra_centers]
    return values, derivatives, labels, None


def weak_system(
    values: np.ndarray,
    derivatives: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    target_derivative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gram = (values * weights) @ values.T + (derivatives * weights) @ derivatives.T
    rhs = (values * weights) @ target + (derivatives * weights) @ target_derivative
    column_norms = np.sqrt(np.diag(gram))
    normalized_gram = gram / np.outer(column_norms, column_norms)
    normalized_rhs = rhs / column_norms
    return normalized_gram, normalized_rhs, column_norms


def relative_h1_error(
    coefficients: np.ndarray,
    values: np.ndarray,
    derivatives: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    target_derivative: np.ndarray,
) -> float:
    residual = coefficients @ values - target
    derivative_residual = coefficients @ derivatives - target_derivative
    numerator = np.sum(weights * (residual * residual + derivative_residual * derivative_residual))
    denominator = np.sum(weights * (target * target + target_derivative * target_derivative))
    return float(np.sqrt(numerator / denominator))


def pde_rhs_consistency(
    values: np.ndarray,
    derivatives: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    target_derivative: np.ndarray,
    forcing: np.ndarray,
) -> float:
    weak_rhs = (values * weights) @ target + (derivatives * weights) @ target_derivative
    # The outward flux is g(-R)=-u_*'(-R), g(R)=u_*'(R).
    strong_rhs = (values * weights) @ forcing
    strong_rhs += target_derivative[-1] * values[:, -1] - target_derivative[0] * values[:, 0]
    return float(np.linalg.norm(weak_rhs - strong_rhs) / np.linalg.norm(weak_rhs))


def run_optimizer(
    normalized_gram: np.ndarray,
    normalized_rhs: np.ndarray,
    column_norms: np.ndarray,
    validation_payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
    values, derivatives, weights, target, target_derivative = validation_payload
    gram_tensor = torch.tensor(normalized_gram, dtype=torch.float64)
    rhs_tensor = torch.tensor(normalized_rhs, dtype=torch.float64)
    coordinates = torch.nn.Parameter(torch.zeros(len(column_norms), dtype=torch.float64))

    trace: list[dict[str, float]] = []

    def diagnostics(iteration: int, phase: str) -> dict[str, float]:
        physical = coordinates.detach().cpu().numpy() / column_norms
        return {
            "iteration": float(iteration),
            "phase": phase,
            "relative_h1_error": relative_h1_error(
                physical, values, derivatives, weights, target, target_derivative
            ),
            "coefficient_norm": float(np.linalg.norm(physical)),
        }

    trace.append(diagnostics(0, "initial"))
    adam = torch.optim.Adam([coordinates], lr=ADAM_LR)
    adam_start = time.perf_counter()
    for iteration in range(1, ADAM_STEPS + 1):
        adam.zero_grad(set_to_none=True)
        loss = 0.5 * coordinates @ (gram_tensor @ coordinates) - rhs_tensor @ coordinates
        loss.backward()
        adam.step()
        if iteration % 50 == 0 or iteration == ADAM_STEPS:
            trace.append(diagnostics(iteration, "Adam"))
    adam_seconds = time.perf_counter() - adam_start

    adam_endpoint = diagnostics(ADAM_STEPS, "Adam endpoint")
    lbfgs = torch.optim.LBFGS(
        [coordinates],
        lr=1.0,
        max_iter=LBFGS_STEPS,
        tolerance_grad=1e-14,
        tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss = 0.5 * coordinates @ (gram_tensor @ coordinates) - rhs_tensor @ coordinates
        loss.backward()
        return loss

    lbfgs_start = time.perf_counter()
    lbfgs.step(closure)
    lbfgs_seconds = time.perf_counter() - lbfgs_start
    trace.append(diagnostics(ADAM_STEPS + LBFGS_STEPS, "LBFGS endpoint"))
    physical_coefficients = coordinates.detach().cpu().numpy() / column_norms
    timing = {
        "adam_seconds": adam_seconds,
        "lbfgs_seconds": lbfgs_seconds,
        "total_optimizer_seconds": adam_seconds + lbfgs_seconds,
        "adam_endpoint_relative_h1_error": adam_endpoint["relative_h1_error"],
    }
    return physical_coefficients, trace, timing


def make_plot(results: list[dict[str, object]], output_pdf: Path, output_png: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        }
    )
    styles = {
        "translated": {"color": "#1f5aa6", "linestyle": "-", "marker": "o"},
        "completed": {"color": "#b33c2e", "linestyle": "--", "marker": "s"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(4.685, 2.55))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.18, top=0.90, wspace=0.50)
    for result in results:
        name = str(result["name"])
        trace = result["trace"]
        iterations = np.asarray([row["iteration"] for row in trace], dtype=float)
        error = np.asarray([row["relative_h1_error"] for row in trace], dtype=float)
        norm = np.asarray([row["coefficient_norm"] for row in trace], dtype=float)
        style = styles[name]
        axes[0].semilogy(
            iterations,
            error,
            label=name.capitalize(),
            markevery=max(1, len(iterations) // 8),
            markersize=3.2,
            linewidth=1.5,
            **style,
        )
        # The optimization starts from the exact zero vector.  A logarithmic
        # axis would have to replace that valid zero by an artificial tiny
        # positive number, which destroys the useful scale of this panel.
        axes[1].plot(
            iterations,
            norm,
            label=name.capitalize(),
            markevery=max(1, len(iterations) // 8),
            markersize=3.2,
            linewidth=1.5,
            **style,
        )
    for axis in axes:
        axis.axvline(ADAM_STEPS, color="0.35", linewidth=0.8, linestyle=":")
        axis.set_xlabel("optimizer budget")
        axis.grid(True, which="both", color="0.9", linewidth=0.45)
    axes[0].set_ylabel(r"relative $H^1$ error")
    axes[1].set_ylabel(r"$\|w\|_2$", labelpad=1.5)
    norm_max = max(
        float(np.max(np.asarray([row["coefficient_norm"] for row in result["trace"]], dtype=float)))
        for result in results
    )
    axes[1].set_ylim(0.0, 1.06 * norm_max)
    axes[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    axes[0].set_title("(a) weak-solution error")
    axes[1].set_title("(b) coordinate size")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, loc="center right")
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def sync_preview(output_png: Path) -> None:
    """Update the browser-friendly PNG in ``figures/previews``."""

    PREVIEW_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_png, PREVIEW_FIGURE_DIRECTORY / output_png.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync-figures",
        action="store_true",
        help="update the PNG preview in figures/previews",
    )
    parser.add_argument(
        "--plot-from-cache",
        action="store_true",
        help="redraw the PDF/PNG from the existing JSON without changing histories or timings",
    )
    args = parser.parse_args()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_json = REFERENCE_JSON
    output_csv = REFERENCE_CSV
    output_pdf = OUTPUT_DIRECTORY / f"{STEM}.pdf"
    output_png = OUTPUT_DIRECTORY / f"{STEM}.png"
    if args.plot_from_cache:
        payload = json.loads(output_json.read_text())
        cached_results = payload.get("results")
        if not isinstance(cached_results, list) or len(cached_results) != 2:
            raise ValueError("cached matched-PDE JSON must contain two result records")
        make_plot(cached_results, output_pdf, output_png)
        if args.sync_figures:
            sync_preview(output_png)
        print(f"Wrote {output_pdf} and {output_png} from {output_json}")
        return

    torch.set_num_threads(1)
    train_x, train_weights = trapezoid_grid(TRAIN_NODES)
    validation_x, validation_weights = trapezoid_grid(VALIDATION_NODES)
    train_target, train_target_derivative, train_forcing = target_fields(train_x)
    validation_target, validation_target_derivative, validation_forcing = target_fields(validation_x)

    dictionary_builders = {
        "translated": translated_dictionary,
        "completed": completed_dictionary,
    }
    results: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for name, builder in dictionary_builders.items():
        train_values, train_derivatives, labels, centers = builder(train_x)
        validation_values, validation_derivatives, _, _ = builder(validation_x)
        normalized_gram, normalized_rhs, column_norms = weak_system(
            train_values,
            train_derivatives,
            train_weights,
            train_target,
            train_target_derivative,
        )
        coefficients, trace, timing = run_optimizer(
            normalized_gram,
            normalized_rhs,
            column_norms,
            (
                validation_values,
                validation_derivatives,
                validation_weights,
                validation_target,
                validation_target_derivative,
            ),
        )
        least_squares_normalized, _, _, _ = np.linalg.lstsq(
            normalized_gram, normalized_rhs, rcond=1e-15
        )
        least_squares_coefficients = least_squares_normalized / column_norms
        least_squares_error = relative_h1_error(
            least_squares_coefficients,
            validation_values,
            validation_derivatives,
            validation_weights,
            validation_target,
            validation_target_derivative,
        )
        final_error = relative_h1_error(
            coefficients,
            validation_values,
            validation_derivatives,
            validation_weights,
            validation_target,
            validation_target_derivative,
        )
        minimum_separation = (
            float(np.min(np.diff(np.sort(centers)))) if centers is not None else None
        )
        result = {
            "name": name,
            "basis_labels": labels,
            "basis_count": len(labels),
            "centers": None if centers is None else centers.tolist(),
            "minimum_center_separation": minimum_separation,
            "normalized_h1_gram_condition_number": float(np.linalg.cond(normalized_gram)),
            "pde_rhs_relative_consistency_error_train_grid": pde_rhs_consistency(
                train_values,
                train_derivatives,
                train_weights,
                train_target,
                train_target_derivative,
                train_forcing,
            ),
            "pde_rhs_relative_consistency_error_validation_grid": pde_rhs_consistency(
                validation_values,
                validation_derivatives,
                validation_weights,
                validation_target,
                validation_target_derivative,
                validation_forcing,
            ),
            "adam_endpoint_relative_h1_error": timing["adam_endpoint_relative_h1_error"],
            "final_relative_h1_error": final_error,
            "least_squares_reference_relative_h1_error": least_squares_error,
            "final_coefficient_norm": float(np.linalg.norm(coefficients)),
            "least_squares_reference_coefficient_norm": float(
                np.linalg.norm(least_squares_coefficients)
            ),
            "final_coefficients": coefficients.tolist(),
            "least_squares_reference_coefficients": least_squares_coefficients.tolist(),
            "timing_seconds": {
                "adam": timing["adam_seconds"],
                "lbfgs": timing["lbfgs_seconds"],
                "total_optimizer": timing["total_optimizer_seconds"],
            },
            "trace": trace,
        }
        results.append(result)
        csv_rows.append(
            {
                "model": name,
                "coordinates": len(labels),
                "minimum_center_separation": "NA" if minimum_separation is None else minimum_separation,
                "normalized_h1_gram_condition_number": result[
                    "normalized_h1_gram_condition_number"
                ],
                "adam_endpoint_relative_h1_error": result["adam_endpoint_relative_h1_error"],
                "final_relative_h1_error": final_error,
                "least_squares_reference_relative_h1_error": least_squares_error,
                "final_coefficient_norm": result["final_coefficient_norm"],
                "optimizer_wall_seconds": timing["total_optimizer_seconds"],
            }
        )

    payload = {
        "experiment": "matched weak-PDE coordinate comparison",
        "interpretation": (
            "Oracle-assisted coordinate-conditioning test; the collision structure and "
            "derivative order are prescribed, and no discovery claim is made."
        ),
        "domain": [-DOMAIN_RADIUS, DOMAIN_RADIUS],
        "weak_problem": {
            "operator": "-d^2/dx^2 + I",
            "forcing": "-u_*'' + u_*",
            "natural_boundary_data": "n u' = n u_*' at {-8,8}",
            "target": "u_*(x)=K'''(x)+0.6 K(x-1.4), K(x)=exp(-x^2/2)",
        },
        "quadrature": {
            "rule": "composite trapezoidal",
            "training_nodes": TRAIN_NODES,
            "validation_nodes": VALIDATION_NODES,
        },
        "matched_optimizer": {
            "initialization": "all normalized coordinates equal to zero",
            "random_seed": None,
            "determinism": "No random quantities, minibatches, or resampling are used.",
            "adam_steps": ADAM_STEPS,
            "adam_learning_rate": ADAM_LR,
            "adam_betas": [0.9, 0.999],
            "adam_epsilon": 1e-8,
            "lbfgs_max_iterations": LBFGS_STEPS,
            "lbfgs_learning_rate": 1.0,
            "lbfgs_line_search": "strong_wolfe",
            "arithmetic": "float64",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": mpl.__version__,
            "platform": platform.platform(),
        },
        "results": results,
    }

    WEAK_PDE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    make_plot(results, output_pdf, output_png)

    if args.sync_figures:
        sync_preview(output_png)

    print(json.dumps(csv_rows, indent=2))


if __name__ == "__main__":
    main()
