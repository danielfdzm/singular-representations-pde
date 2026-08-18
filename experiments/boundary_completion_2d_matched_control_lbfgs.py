"""Apply the completion run's LBFGS budget to the cached 2D translated control.

The source experiment gives the translated and completed models the same Adam
initialization and update budget, but only the completed dictionary receives a
final strong-Wolfe LBFGS polish.  This script closes that comparison gap without
repeating Adam: it reconstructs the terminal translated state from the JSON
cache, applies the identical LBFGS configuration, and validates all three
terminal states on the independent grid used by the paper.

It also reports a scale-invariant feature-conditioning diagnostic.  If Phi is
the validation-grid design matrix and W contains the tensor-product trapezoid
weights, set G = Phi.T W Phi and D = diag(G).  The reported condition number is

    kappa_2(D^{-1/2} G D^{-1/2}),

computed by SVD without clipping or regularization.  Thus every design column
has unit discrete L2 norm before conditioning is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

import boundary_completion_2d_experiment as experiment


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_INPUT = HERE / "boundary_completion_2d_atoms_adam.json"
DEFAULT_OUTPUT = HERE / "boundary_completion_2d_matched_control_lbfgs.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE.resolve()))
    except ValueError:
        return str(resolved)


def dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def execution_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": dependency_version("numpy"),
        "torch": dependency_version("torch"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def symmetric_square_grid(specification: dict[str, Any]) -> tuple[int, float]:
    nodes = int(specification["nodes_per_axis"])
    domain = [float(value) for value in specification["domain"]]
    if len(domain) != 4:
        raise ValueError(f"expected a four-entry tensor-grid domain, got {domain}")
    radius = domain[1]
    expected = [-radius, radius, -radius, radius]
    if radius <= 0.0 or not np.allclose(domain, expected, rtol=0.0, atol=0.0):
        raise ValueError(f"expected a symmetric square domain, got {domain}")
    if nodes < 3:
        raise ValueError("a tensor grid needs at least three nodes per axis")
    return nodes, radius


def make_grid(specification: dict[str, Any]) -> experiment.TorchGrid:
    nodes, radius = symmetric_square_grid(specification)
    _, _, x, y, quadrature = experiment.tensor_grid(nodes, radius)
    return experiment.make_torch_grid(x, y, quadrature)


def relative_error_on_grid(
    stage: int,
    log_h: torch.Tensor | None,
    center: torch.Tensor,
    coefficients: torch.Tensor,
    grid: experiment.TorchGrid,
) -> float:
    with torch.no_grad():
        loss = experiment.loss_on_grid(
            stage,
            log_h,
            experiment.assemble_ordinary_centers(center),
            coefficients,
            grid,
        )
    return experiment.relative_error(loss)


def gram_diagnostics(
    stage: int,
    log_h: torch.Tensor | None,
    center: torch.Tensor,
    grid: experiment.TorchGrid,
) -> dict[str, Any]:
    with torch.no_grad():
        design, _, _ = experiment.torch_basis_for_stage(
            grid.x,
            grid.y,
            stage,
            log_h,
            experiment.assemble_ordinary_centers(center),
        )
    phi = design.detach().cpu().numpy()
    weights = grid.quadrature.detach().cpu().numpy()
    gram = phi.T @ (weights[:, None] * phi)
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise ValueError("the weighted design Gramian has an invalid diagonal")
    normalized = gram / np.sqrt(np.outer(diagonal, diagonal))
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    smallest = float(singular_values[-1])
    largest = float(singular_values[0])
    condition = math.inf if smallest == 0.0 else largest / smallest
    reciprocal = 0.0 if math.isinf(condition) else 1.0 / condition
    return {
        "matrix": "D^(-1/2) Phi^T W Phi D^(-1/2)",
        "norm": "spectral 2-norm via singular values",
        "column_normalization": "unit discrete validation-L2 norm",
        "regularization_or_clipping": "none",
        "number_of_features": int(phi.shape[1]),
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_number": condition,
        "reciprocal_condition_number": reciprocal,
        "numerical_note": (
            "Condition estimates with reciprocal condition near machine epsilon "
            "are necessarily sensitive to floating-point and BLAS details."
        ),
    }


def validate_cached_translated_state(final: dict[str, Any]) -> None:
    weights = np.asarray(final["weights"], dtype=np.float64)
    basis = np.asarray(final["basis"], dtype=np.float64)
    ordinary = np.asarray(final["ordinary_centers"], dtype=np.float64)
    h = float(final["h"])
    if weights.shape != (experiment.WIDTH,):
        raise ValueError(f"expected {experiment.WIDTH} translated coefficients")
    if basis.shape != (experiment.WIDTH, 3) or np.any(basis[:, 0] != 0.0):
        raise ValueError("the cached control is not a purely translated 2D basis")
    if ordinary.shape != (1, 2):
        raise ValueError("the cached control must contain one ordinary center")
    expected_cluster = experiment.LEAF_OFFSETS[:, None] * h * experiment.DIRECTION[None, :]
    expected_centers = np.vstack((expected_cluster, ordinary))
    if not np.allclose(basis[:, 1:], expected_centers, rtol=0.0, atol=1e-13):
        raise ValueError("the cached translated basis cannot be reconstructed")


def state_metrics(
    *,
    label: str,
    stage: int,
    log_h: torch.Tensor | None,
    center: torch.Tensor,
    coefficients: torch.Tensor,
    training_grid: experiment.TorchGrid,
    validation_grid: experiment.TorchGrid,
    optimization: str,
) -> dict[str, Any]:
    h = float(torch.exp(log_h.detach())) if log_h is not None else None
    return {
        "label": label,
        "optimization": optimization,
        "training_relative_l2": relative_error_on_grid(
            stage, log_h, center, coefficients, training_grid
        ),
        "validation_relative_l2": relative_error_on_grid(
            stage, log_h, center, coefficients, validation_grid
        ),
        "coefficients": coefficients.detach().cpu().tolist(),
        "coefficient_l2_norm": float(torch.linalg.vector_norm(coefficients.detach())),
        "h": h,
        "active_pair_separation": (
            experiment.collapsing_pair_separation(stage, h) if h is not None else None
        ),
        "ordinary_center": center.detach().cpu().tolist(),
        "guard_penalty": float(experiment.ordinary_center_guard_penalty(center.detach())),
        "validation_design_gram": gram_diagnostics(
            stage, log_h, center, validation_grid
        ),
    }


def completed_state(
    source: dict[str, Any],
    training_grid: experiment.TorchGrid,
    validation_grid: experiment.TorchGrid,
) -> dict[str, Any]:
    results = source.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("expected exactly one cached 2D completion run")
    summary = results[0].get("summaries", {}).get("stage_3")
    if not isinstance(summary, dict):
        raise ValueError("the cache has no completed stage_3 state")
    coefficients = torch.tensor(summary["weights"], dtype=torch.float64)
    ordinary = np.asarray(summary["ordinary_centers"], dtype=np.float64)
    if ordinary.shape != (1, 2):
        raise ValueError("the completed cache must contain one ordinary center")
    center = torch.tensor(ordinary[0], dtype=torch.float64)
    metrics = state_metrics(
        label="completed dictionary after Adam transfers and LBFGS",
        stage=3,
        log_h=None,
        center=center,
        coefficients=coefficients,
        training_grid=training_grid,
        validation_grid=validation_grid,
        optimization="same Adam initialization and budget; 200-step strong-Wolfe LBFGS",
    )
    cached_error = float(summary["relative_error"])
    if not math.isclose(
        metrics["validation_relative_l2"], cached_error, rel_tol=2e-12, abs_tol=1e-14
    ):
        raise ValueError(
            "reconstructed completed validation error does not match the source cache"
        )
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.input.read_text())
    controls = source.get("fixed_translated_control")
    if not isinstance(controls, list) or len(controls) != 1:
        raise ValueError("expected exactly one cached fixed translated control")
    final = controls[0].get("final")
    if not isinstance(final, dict):
        raise ValueError("the source cache has no terminal translated state")
    validate_cached_translated_state(final)

    training_grid = make_grid(source["training_quadrature"])
    validation_grid = make_grid(source["validation_quadrature"])
    coefficients = torch.nn.Parameter(torch.tensor(final["weights"], dtype=torch.float64))
    log_h = torch.nn.Parameter(
        torch.tensor(math.log(float(final["h"])), dtype=torch.float64)
    )
    center = torch.nn.Parameter(
        torch.tensor(final["ordinary_centers"][0], dtype=torch.float64)
    )
    parameters = [coefficients, log_h, center]

    before = state_metrics(
        label="translated dictionary after Adam",
        stage=0,
        log_h=log_h,
        center=center,
        coefficients=coefficients,
        training_grid=training_grid,
        validation_grid=validation_grid,
        optimization="cached Adam endpoint; no LBFGS",
    )
    cached_error = float(final["relative_error"])
    if not math.isclose(
        before["validation_relative_l2"], cached_error, rel_tol=2e-12, abs_tol=1e-14
    ):
        raise ValueError("reconstructed translated validation error does not match the cache")

    def objective() -> torch.Tensor:
        loss = experiment.loss_on_grid(
            0,
            log_h,
            experiment.assemble_ordinary_centers(center),
            coefficients,
            training_grid,
        )
        return loss + experiment.ordinary_center_guard_penalty(center)

    optimizer = torch.optim.LBFGS(
        parameters,
        lr=args.learning_rate,
        max_iter=args.max_iter,
        max_eval=args.max_eval,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal closure_evaluations
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    start = time.perf_counter()
    optimizer.step(closure)
    lbfgs_wall_seconds = time.perf_counter() - start
    state = optimizer.state[coefficients]

    after = state_metrics(
        label="translated dictionary after Adam and matched LBFGS",
        stage=0,
        log_h=log_h,
        center=center,
        coefficients=coefficients,
        training_grid=training_grid,
        validation_grid=validation_grid,
        optimization="cached Adam endpoint followed by matched strong-Wolfe LBFGS",
    )
    completed = completed_state(source, training_grid, validation_grid)

    return {
        "schema_version": 1,
        "source": {
            "path": portable_path(args.input),
            "sha256": sha256(args.input),
            "seed": source.get("seed"),
            "adam_max_iteration": source.get("max_iteration"),
        },
        "method": (
            "Reconstruct the cached terminal translated state and optimize all "
            "coefficients, log(h), and the ordinary center with the identical "
            "strong-Wolfe LBFGS settings used by the completed dictionary."
        ),
        "training_quadrature": source["training_quadrature"],
        "validation_quadrature": source["validation_quadrature"],
        "lbfgs": {
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "max_eval": args.max_eval,
            "tolerance_grad": args.tolerance_grad,
            "tolerance_change": args.tolerance_change,
            "line_search": "strong_wolfe",
            "iterations_performed": int(state.get("n_iter", 0)),
            "closure_evaluations": int(state.get("func_evals", closure_evaluations)),
            "closure_evaluations_observed": closure_evaluations,
            "wall_seconds": lbfgs_wall_seconds,
        },
        "condition_number_definition": {
            "formula": "kappa_2(D^(-1/2) Phi^T W Phi D^(-1/2))",
            "Phi": "terminal design matrix on the independent validation grid",
            "W": "tensor-product composite-trapezoid weights",
            "D": "diag(Phi^T W Phi)",
            "computation": "SVD, without clipping or regularization",
        },
        "terminal_metrics": [before, after, completed],
        "execution_environment": execution_environment(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-eval", type=int, default=400)
    parser.add_argument("--tolerance-grad", type=float, default=1e-14)
    parser.add_argument("--tolerance-change", type=float, default=1e-18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0 or args.max_iter < 0 or args.max_eval < 1:
        raise ValueError("invalid LBFGS iteration or learning-rate settings")
    if args.tolerance_grad < 0.0 or args.tolerance_change < 0.0:
        raise ValueError("LBFGS tolerances must be nonnegative")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    torch.set_num_threads(1)
    payload = run(args)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    before, after, completed = payload["terminal_metrics"]
    print(f"Wrote {args.output}")
    for row in (before, after, completed):
        print(
            f"{row['label']}: error={row['validation_relative_l2']:.6e}, "
            f"||w||={row['coefficient_l2_norm']:.6e}, "
            f"kappa={row['validation_design_gram']['condition_number']:.6e}"
        )
    print(
        f"Matched LBFGS: {payload['lbfgs']['iterations_performed']} iterations, "
        f"{payload['lbfgs']['closure_evaluations']} function evaluations, "
        f"{payload['lbfgs']['wall_seconds']:.3f} s"
    )


if __name__ == "__main__":
    main()
