"""Matched LBFGS polish for the cached fixed translated-feature controls.

This script reads the final Adam states stored by
``boundary_completion_experiment.py`` and applies the same strong-Wolfe
LBFGS configuration used to polish the completed dictionary.  It does not
repeat the 100,000 Adam iterations.

The optimization grid matches the original experiment.  Reported validation
errors are recomputed on a finer grid that is not used by LBFGS.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from artifact_paths import COMPLETION_DATA_DIRECTORY, experiment_output_directory

DEFAULT_INPUT = (
    COMPLETION_DATA_DIRECTORY
    / "boundary_completion_third_derivative_atoms_lbfgs_plateau.json"
)
DEFAULT_OUTPUT = (
    experiment_output_directory("matched-controls")
    / "boundary_completion_matched_control_lbfgs.json"
)

LEAF_OFFSETS = torch.tensor(
    [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0],
    dtype=torch.float64,
)
BETA = 0.6
TARGET_CENTER = 1.4
GUARD_RADIUS = 0.85
GUARD_WEIGHT = 1.0


def trap_rule(x: torch.Tensor) -> torch.Tensor:
    dx = x[1] - x[0]
    weights = torch.full_like(x, dx)
    weights[0] = 0.5 * dx
    weights[-1] = 0.5 * dx
    return weights


def gaussian(x: torch.Tensor, center: torch.Tensor | float) -> torch.Tensor:
    return torch.exp(-0.5 * (x - center) ** 2)


def gaussian_third_derivative(x: torch.Tensor) -> torch.Tensor:
    return (3.0 * x - x**3) * torch.exp(-0.5 * x**2)


def gaussian_derivative_atom(
    x: torch.Tensor, center: torch.Tensor | float, order: int
) -> torch.Tensor:
    z = x - center
    kernel = torch.exp(-0.5 * z**2)
    if order == 0:
        return kernel
    if order == 1:
        return -z * kernel
    if order == 2:
        return (z**2 - 1.0) * kernel
    if order == 3:
        return (3.0 * z - z**3) * kernel
    raise ValueError(f"unsupported derivative order {order}")


def target_fn(x: torch.Tensor) -> torch.Tensor:
    return gaussian_third_derivative(x) + BETA * gaussian(x, TARGET_CENTER)


def translated_basis(
    x: torch.Tensor,
    log_h: torch.Tensor,
    primary_center: torch.Tensor,
    fixed_extra_centers: torch.Tensor,
) -> torch.Tensor:
    h = torch.exp(log_h)
    cluster_centers = LEAF_OFFSETS.to(x) * h
    ordinary_centers = torch.cat((primary_center.reshape(1), fixed_extra_centers))
    centers = torch.cat((cluster_centers, ordinary_centers))
    return torch.exp(-0.5 * (x[:, None] - centers[None, :]) ** 2)


def guard_penalty(primary_center: torch.Tensor) -> torch.Tensor:
    gap = torch.relu(primary_center.new_tensor(GUARD_RADIUS) - torch.abs(primary_center))
    return primary_center.new_tensor(GUARD_WEIGHT) * torch.mean(gap * gap)


def relative_l2(
    x: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    quadrature = trap_rule(x)
    numerator = torch.sum(quadrature * (prediction - target) ** 2)
    denominator = torch.sum(quadrature * target**2)
    return float(torch.sqrt(torch.clamp(numerator / denominator, min=0.0)))


def evaluate(
    x: torch.Tensor,
    weights: torch.Tensor,
    log_h: torch.Tensor,
    primary_center: torch.Tensor,
    fixed_extra_centers: torch.Tensor,
) -> float:
    with torch.no_grad():
        phi = translated_basis(x, log_h, primary_center, fixed_extra_centers)
        return relative_l2(x, phi @ weights, target_fn(x))


def validate_cached_state(control: dict[str, Any]) -> None:
    final = control["final"]
    width = int(control["width"])
    weights = np.asarray(final["weights"], dtype=np.float64)
    ordinary = np.asarray(final["ordinary_centers"], dtype=np.float64)
    basis = np.asarray(final["basis"], dtype=np.float64)
    h = float(final["h"])

    if weights.shape != (width,):
        raise ValueError(f"P={width}: expected {width} cached weights, got {weights.shape}")
    if ordinary.shape != (width - 8,):
        raise ValueError(
            f"P={width}: expected {width - 8} ordinary centers, got {ordinary.shape}"
        )
    if basis.shape != (width, 2) or np.any(basis[:, 0] != 0.0):
        raise ValueError(f"P={width}: cache is not a purely translated Gaussian basis")

    expected_centers = np.concatenate((LEAF_OFFSETS.numpy() * h, ordinary))
    if not np.allclose(basis[:, 1], expected_centers, rtol=0.0, atol=1e-13):
        raise ValueError(f"P={width}: cached basis cannot be reconstructed from h and centers")


def polish_control(
    control: dict[str, Any],
    train_grid_size: int,
    validation_grid_size: int,
    max_iter: int,
    learning_rate: float,
) -> dict[str, Any]:
    validate_cached_state(control)
    final = control["final"]
    width = int(control["width"])

    x_train = torch.linspace(-8.0, 8.0, train_grid_size, dtype=torch.float64)
    x_validation = torch.linspace(-8.0, 8.0, validation_grid_size, dtype=torch.float64)
    target_train = target_fn(x_train)
    quadrature_train = trap_rule(x_train)
    target_norm_sq = torch.sum(quadrature_train * target_train**2)

    ordinary = np.asarray(final["ordinary_centers"], dtype=np.float64)
    weights = torch.nn.Parameter(torch.tensor(final["weights"], dtype=torch.float64))
    log_h = torch.nn.Parameter(torch.tensor(math.log(float(final["h"])), dtype=torch.float64))
    primary_center = torch.nn.Parameter(torch.tensor([ordinary[0]], dtype=torch.float64))
    fixed_extra_centers = torch.tensor(ordinary[1:], dtype=torch.float64)
    parameters = [weights, log_h, primary_center]

    before = {
        "train_relative_l2": evaluate(
            x_train, weights, log_h, primary_center, fixed_extra_centers
        ),
        "validation_relative_l2": evaluate(
            x_validation, weights, log_h, primary_center, fixed_extra_centers
        ),
        "coefficient_l2_norm": float(torch.linalg.norm(weights.detach())),
        "h": float(torch.exp(log_h.detach())),
        "primary_center": float(primary_center.detach()[0]),
    }
    cached_error = float(final["relative_error"])
    if not math.isclose(
        before["train_relative_l2"], cached_error, rel_tol=1e-11, abs_tol=1e-13
    ):
        raise ValueError(
            f"P={width}: reconstructed training error {before['train_relative_l2']:.16e} "
            f"does not match cached error {cached_error:.16e}; check the training grid"
        )

    def losses() -> tuple[torch.Tensor, torch.Tensor]:
        phi = translated_basis(x_train, log_h, primary_center, fixed_extra_centers)
        prediction = phi @ weights
        relative_squared_l2 = (
            torch.sum(quadrature_train * (prediction - target_train) ** 2) / target_norm_sq
        )
        return relative_squared_l2, relative_squared_l2 + guard_penalty(primary_center)

    optimizer = torch.optim.LBFGS(
        parameters,
        lr=learning_rate,
        max_iter=max_iter,
        max_eval=max(2 * max_iter, max_iter + 1),
        tolerance_grad=1e-14,
        tolerance_change=1e-18,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _, objective = losses()
        objective.backward()
        return objective

    optimizer.step(closure)
    with torch.no_grad():
        _, final_objective = losses()

    state = optimizer.state[weights]
    after = {
        "train_relative_l2": evaluate(
            x_train, weights, log_h, primary_center, fixed_extra_centers
        ),
        "validation_relative_l2": evaluate(
            x_validation, weights, log_h, primary_center, fixed_extra_centers
        ),
        "coefficient_l2_norm": float(torch.linalg.norm(weights.detach())),
        "h": float(torch.exp(log_h.detach())),
        "primary_center": float(primary_center.detach()[0]),
        "guard_penalty": float(guard_penalty(primary_center.detach())),
        "objective": float(final_objective),
        "lbfgs_iterations": int(state.get("n_iter", 0)),
        "closure_evaluations": int(state.get("func_evals", 0)),
    }

    return {
        "width": width,
        "fixed_extra_centers": fixed_extra_centers.tolist(),
        "before_lbfgs": before,
        "after_lbfgs": after,
    }


def evaluate_completed_state(
    result: dict[str, Any], train_grid_size: int, validation_grid_size: int
) -> dict[str, Any]:
    width = int(result["width"])
    summary = result.get("summaries", {}).get("stage_3")
    if not isinstance(summary, dict):
        raise ValueError(f"P={width}: cache has no completed stage_3 summary")

    basis = summary.get("basis")
    weights = torch.tensor(summary.get("weights"), dtype=torch.float64)
    if not isinstance(basis, list) or len(basis) != len(weights):
        raise ValueError(f"P={width}: malformed completed basis or weight vector")

    def state_error(grid_size: int) -> float:
        x = torch.linspace(-8.0, 8.0, grid_size, dtype=torch.float64)
        columns = [
            gaussian_derivative_atom(x, float(center), int(order))
            for order, center in basis
        ]
        prediction = torch.stack(columns, dim=1) @ weights
        return relative_l2(x, prediction, target_fn(x))

    train_error = state_error(train_grid_size)
    cached_error = float(summary["relative_error"])
    if not math.isclose(train_error, cached_error, rel_tol=2e-7, abs_tol=1e-14):
        raise ValueError(
            f"P={width}: reconstructed completed error {train_error:.16e} does not "
            f"match cached error {cached_error:.16e}"
        )
    return {
        "width": width,
        "train_relative_l2": train_error,
        "validation_relative_l2": state_error(validation_grid_size),
        "coefficient_l2_norm": float(torch.linalg.norm(weights)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-grid-size", type=int, default=2001)
    parser.add_argument("--validation-grid-size", type=int, default=20001)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    args = parser.parse_args()

    if args.train_grid_size < 2 or args.validation_grid_size <= args.train_grid_size:
        raise ValueError("validation grid must be finer than the training grid")
    if args.max_iter < 0 or args.learning_rate <= 0.0:
        raise ValueError("LBFGS settings must be nonnegative iterations and positive learning rate")

    source = json.loads(args.input.read_text())
    controls = source.get("fixed_translated_control")
    if not isinstance(controls, list):
        raise ValueError("input cache has no fixed_translated_control list")
    widths = [int(control["width"]) for control in controls]
    if widths != [9, 10, 12]:
        raise ValueError(f"expected cached widths [9, 10, 12], got {widths}")

    results = [
        polish_control(
            control,
            args.train_grid_size,
            args.validation_grid_size,
            args.max_iter,
            args.learning_rate,
        )
        for control in controls
    ]
    completed_results = source.get("results")
    if not isinstance(completed_results, list):
        raise ValueError("input cache has no completed results list")
    completed_validation = [
        evaluate_completed_state(
            result, args.train_grid_size, args.validation_grid_size
        )
        for result in completed_results
    ]
    payload = {
        "source": args.input.name,
        "method": (
            "float64 strong-Wolfe LBFGS initialized from each cached 100000-step "
            "Adam state; trainable variables are all coefficients, log(h), and the "
            "primary ordinary center; the original center guard is retained"
        ),
        "domain": [-8.0, 8.0],
        "train_grid_size": args.train_grid_size,
        "validation_grid_size": args.validation_grid_size,
        "quadrature": "composite trapezoidal rule",
        "lbfgs": {
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "max_eval": max(2 * args.max_iter, args.max_iter + 1),
            "tolerance_grad": 1e-14,
            "tolerance_change": 1e-18,
            "line_search": "strong_wolfe",
        },
        "guard": {"radius": GUARD_RADIUS, "weight": GUARD_WEIGHT},
        "matched_fixed_controls": results,
        "completed_stage_3_validation": completed_validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {args.output}")
    for result in results:
        before = result["before_lbfgs"]
        after = result["after_lbfgs"]
        print(
            f"P={result['width']}: "
            f"train {before['train_relative_l2']:.6e} -> {after['train_relative_l2']:.6e}; "
            f"fine {before['validation_relative_l2']:.6e} -> "
            f"{after['validation_relative_l2']:.6e}; "
            f"||w|| {before['coefficient_l2_norm']:.6e} -> "
            f"{after['coefficient_l2_norm']:.6e}; "
            f"LBFGS iterations={after['lbfgs_iterations']}"
        )
    for result in completed_validation:
        print(
            f"completed P={result['width']}: "
            f"train={result['train_relative_l2']:.6e}; "
            f"fine={result['validation_relative_l2']:.6e}; "
            f"||w||={result['coefficient_l2_norm']:.6e}"
        )


if __name__ == "__main__":
    main()
