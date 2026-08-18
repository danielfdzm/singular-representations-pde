"""Two-dimensional directional derivative-completion experiment.

This script is the two-dimensional counterpart of
boundary_completion_experiment.py. It uses

    K(z;c) = exp(-|z-c|^2/2),
    u*(z) = D_v^3 K(z;0) + beta K(z;c0),

where v is a fixed unit direction. Eight translated Gaussian atoms form a
collapsing stencil along v. Full-batch float64 Adam trains the linear weights,
the logarithmic stencil scale, and one displaced ordinary center. A stage
change occurs only when the active pair separation crosses a prescribed
fraction of its value at the start of that stage. The collapsing pair is then
replaced by its same-order and next-order directional Taylor atoms. After the
third insertion, strong-Wolfe LBFGS polishes the completed dictionary.

A matched fixed-translated Adam run is included as a control. Training and
validation errors use different tensor-product trapezoidal grids.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch

from artifact_paths import (
    COMPLETION_DATA_DIRECTORY,
    PREVIEW_FIGURE_DIRECTORY,
    experiment_output_directory,
)

OUT_STEM = "boundary_completion_2d_atoms"
OUT_REP_STEM = "boundary_completion_2d_representation"
OUTPUT_DIRECTORY = experiment_output_directory("completion")
OUT_PDF = OUTPUT_DIRECTORY / f"{OUT_STEM}_adam.pdf"
OUT_PNG = OUTPUT_DIRECTORY / f"{OUT_STEM}_adam.png"
OUT_REP_PDF = OUTPUT_DIRECTORY / f"{OUT_REP_STEM}_adam.pdf"
OUT_REP_PNG = OUTPUT_DIRECTORY / f"{OUT_REP_STEM}_adam.png"

WIDTH = 9
BETA = 0.6
TARGET_CENTER = np.array([1.35, -0.85], dtype=np.float64)
PRIMARY_INITIAL_CENTER = np.array([2.25, -1.45], dtype=np.float64)
DIRECTION = np.array([1.0, 0.65], dtype=np.float64)
DIRECTION /= np.linalg.norm(DIRECTION)

MAX_ITER = 40_000
ADAM_LEARNING_RATE = 1.95e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
GRAD_CLIP = 100.0
RECORD_EVERY = 250
ERROR_FLOOR = 1e-16
INITIAL_WEIGHT_SCALE = 1e-3
ORDINARY_CENTER_GUARD = 0.85
ORDINARY_CENTER_GUARD_WEIGHT = 1.0
COLLAPSE_RATIOS = (0.35, 0.75, 0.86)
DEFAULT_SEED = 80_209

LEAF_OFFSETS = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
FIRST_OFFSETS = np.array([-6.0, -2.0, 2.0, 6.0])
SECOND_OFFSETS = np.array([-4.0, 4.0])
STAGE_LABELS = (
    "translated atoms",
    r"after adding $D_vK$",
    r"after adding $D_v^2K$",
    r"after adding $D_v^3K$",
)


@dataclass(frozen=True)
class Atom2D:
    order: int
    center: tuple[float, float]


@dataclass(frozen=True)
class TorchGrid:
    x: torch.Tensor
    y: torch.Tensor
    quadrature: torch.Tensor
    target: torch.Tensor
    target_norm_sq: torch.Tensor


def directional_gaussian_atom(
    x: np.ndarray,
    y: np.ndarray,
    center: tuple[float, float] | np.ndarray,
    order: int,
) -> np.ndarray:
    """Evaluate D_v^order K((x,y);center) for orders zero through three."""
    cx, cy = float(center[0]), float(center[1])
    zx = x - cx
    zy = y - cy
    s = DIRECTION[0] * zx + DIRECTION[1] * zy
    kernel = np.exp(-0.5 * (zx * zx + zy * zy))
    if order == 0:
        return kernel
    if order == 1:
        return -s * kernel
    if order == 2:
        return (s * s - 1.0) * kernel
    if order == 3:
        return (3.0 * s - s**3) * kernel
    raise ValueError(f"unsupported derivative order {order}")


def torch_directional_gaussian_atom(
    x: torch.Tensor,
    y: torch.Tensor,
    center: torch.Tensor,
    order: int,
) -> torch.Tensor:
    direction = x.new_tensor(DIRECTION)
    zx = x - center[0]
    zy = y - center[1]
    s = direction[0] * zx + direction[1] * zy
    kernel = torch.exp(-0.5 * (zx * zx + zy * zy))
    if order == 0:
        return kernel
    if order == 1:
        return -s * kernel
    if order == 2:
        return (s * s - 1.0) * kernel
    if order == 3:
        return (3.0 * s - s**3) * kernel
    raise ValueError(f"unsupported derivative order {order}")


def target_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return directional_gaussian_atom(x, y, (0.0, 0.0), 3) + BETA * directional_gaussian_atom(
        x, y, TARGET_CENTER, 0
    )


def trap_rule_1d(nodes: np.ndarray) -> np.ndarray:
    if len(nodes) < 2:
        raise ValueError("the trapezoidal rule needs at least two nodes")
    spacing = nodes[1] - nodes[0]
    weights = np.full_like(nodes, spacing)
    weights[0] = weights[-1] = 0.5 * spacing
    return weights


def tensor_grid(
    n_grid: int,
    domain_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes_x = np.linspace(-domain_radius, domain_radius, n_grid, dtype=np.float64)
    nodes_y = np.linspace(-domain_radius, domain_radius, n_grid, dtype=np.float64)
    weights_x = trap_rule_1d(nodes_x)
    weights_y = trap_rule_1d(nodes_y)
    xx, yy = np.meshgrid(nodes_x, nodes_y, indexing="xy")
    quadrature = np.outer(weights_y, weights_x)
    return nodes_x, nodes_y, xx.ravel(), yy.ravel(), quadrature.ravel()


def make_torch_grid(x: np.ndarray, y: np.ndarray, quadrature: np.ndarray) -> TorchGrid:
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    quadrature_t = torch.tensor(quadrature, dtype=torch.float64)
    target_t = torch.tensor(target_fn(x, y), dtype=torch.float64)
    target_norm_sq = torch.sum(quadrature_t * target_t * target_t)
    return TorchGrid(x_t, y_t, quadrature_t, target_t, target_norm_sq)


def atoms_for_stage(stage: int, h: float | None, ordinary: np.ndarray) -> list[Atom2D]:
    atoms: list[Atom2D] = []

    def center_at(offset: float) -> tuple[float, float]:
        if h is None:
            raise ValueError(f"stage {stage} requires a scale h")
        center = float(offset) * h * DIRECTION
        return float(center[0]), float(center[1])

    if stage == 0:
        atoms.extend(Atom2D(0, center_at(offset)) for offset in LEAF_OFFSETS)
    elif stage == 1:
        for offset in FIRST_OFFSETS:
            center = center_at(offset)
            atoms.extend((Atom2D(0, center), Atom2D(1, center)))
    elif stage == 2:
        atoms.extend(Atom2D(0, center_at(offset)) for offset in FIRST_OFFSETS)
        for offset in SECOND_OFFSETS:
            center = center_at(offset)
            atoms.extend((Atom2D(1, center), Atom2D(2, center)))
    elif stage == 3:
        atoms.extend(Atom2D(order, (0.0, 0.0)) for order in range(4))
    else:
        raise ValueError(f"unknown stage {stage}")

    atoms.extend(Atom2D(0, (float(center[0]), float(center[1]))) for center in ordinary)
    return atoms


def basis_from_atoms(x: np.ndarray, y: np.ndarray, atoms: list[Atom2D]) -> np.ndarray:
    return np.column_stack(
        [directional_gaussian_atom(x, y, atom.center, atom.order) for atom in atoms]
    )


def torch_basis_for_stage(
    x: torch.Tensor,
    y: torch.Tensor,
    stage: int,
    log_h: torch.Tensor | None,
    ordinary: torch.Tensor,
) -> tuple[torch.Tensor, list[int], list[torch.Tensor]]:
    direction = x.new_tensor(DIRECTION)
    orders: list[int] = []
    centers: list[torch.Tensor] = []
    h = torch.exp(log_h) if log_h is not None else None

    def center_at(offset: float) -> torch.Tensor:
        if h is None:
            raise ValueError(f"stage {stage} requires log_h")
        return x.new_tensor(float(offset)) * h * direction

    if stage == 0:
        for offset in LEAF_OFFSETS:
            orders.append(0)
            centers.append(center_at(float(offset)))
    elif stage == 1:
        for offset in FIRST_OFFSETS:
            center = center_at(float(offset))
            orders.extend((0, 1))
            centers.extend((center, center))
    elif stage == 2:
        for offset in FIRST_OFFSETS:
            orders.append(0)
            centers.append(center_at(float(offset)))
        for offset in SECOND_OFFSETS:
            center = center_at(float(offset))
            orders.extend((1, 2))
            centers.extend((center, center))
    elif stage == 3:
        zero = torch.zeros(2, dtype=x.dtype, device=x.device)
        for order in range(4):
            orders.append(order)
            centers.append(zero)
    else:
        raise ValueError(f"unknown stage {stage}")

    for center in ordinary:
        orders.append(0)
        centers.append(center)

    basis = torch.stack(
        [
            torch_directional_gaussian_atom(x, y, center, order)
            for order, center in zip(orders, centers)
        ],
        dim=1,
    )
    return basis, orders, centers


def collapsing_pair_separation(stage: int, h: float | None) -> float | None:
    if stage == 0:
        if h is None:
            raise ValueError("stage 0 requires h")
        return 2.0 * h
    if stage == 1:
        if h is None:
            raise ValueError("stage 1 requires h")
        return 4.0 * h
    if stage == 2:
        if h is None:
            raise ValueError("stage 2 requires h")
        return 8.0 * h
    if stage == 3:
        return None
    raise ValueError(f"unknown stage {stage}")


def initialize_weights(n_atoms: int, seed: int, scale: float) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return scale * torch.randn(n_atoms, generator=generator, dtype=torch.float64)


def ordinary_center_guard_penalty(primary_center: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(primary_center)
    gap = torch.relu(primary_center.new_tensor(ORDINARY_CENTER_GUARD) - radius)
    return primary_center.new_tensor(ORDINARY_CENTER_GUARD_WEIGHT) * gap * gap


def assemble_ordinary_centers(primary_center: torch.Tensor) -> torch.Tensor:
    return primary_center.reshape(1, 2)


def projected_center(center: tuple[float, float]) -> float:
    return float(np.dot(np.asarray(center, dtype=np.float64), DIRECTION))


def taylor_initialize_next_stage(
    previous_atoms: list[Atom2D],
    previous_weights: np.ndarray,
    next_stage: int,
    h: float,
    ordinary: np.ndarray,
) -> torch.Tensor:
    """Transfer weights by the directional Taylor expansion."""
    next_atoms = atoms_for_stage(next_stage, None if next_stage == 3 else h, ordinary)
    output = np.zeros(len(next_atoms), dtype=np.float64)
    n_ordinary = len(ordinary)
    core_atoms = previous_atoms[:-n_ordinary]
    core_weights = previous_weights[:-n_ordinary]
    output[-n_ordinary:] = previous_weights[-n_ordinary:]

    def add_to_next(order: int, center: np.ndarray, value: float) -> None:
        candidates = [
            (float(np.linalg.norm(np.asarray(atom.center) - center)), index)
            for index, atom in enumerate(next_atoms[:-n_ordinary])
            if atom.order == order
        ]
        if not candidates:
            raise KeyError((order, center.tolist()))
        distance, index = min(candidates, key=lambda item: item[0])
        if distance > 1e-8 * max(1.0, float(np.linalg.norm(center))):
            raise KeyError((order, center.tolist()))
        output[index] += value

    if next_stage == 1:
        items = sorted(
            zip(core_atoms, core_weights),
            key=lambda item: projected_center(item[0].center),
        )
        for (left_atom, left_weight), (right_atom, right_weight) in zip(
            items[0::2], items[1::2]
        ):
            left = np.asarray(left_atom.center, dtype=np.float64)
            right = np.asarray(right_atom.center, dtype=np.float64)
            midpoint = 0.5 * (left + right)
            half_gap = 0.5 * float(np.dot(right - left, DIRECTION))
            add_to_next(left_atom.order, midpoint, float(left_weight + right_weight))
            add_to_next(
                left_atom.order + 1,
                midpoint,
                float(half_gap * (left_weight - right_weight)),
            )
    elif next_stage == 2:
        for atom, weight in zip(core_atoms, core_weights):
            if atom.order == 0:
                add_to_next(0, np.asarray(atom.center, dtype=np.float64), float(weight))
        items = sorted(
            (
                (atom, weight)
                for atom, weight in zip(core_atoms, core_weights)
                if atom.order == 1
            ),
            key=lambda item: projected_center(item[0].center),
        )
        for (left_atom, left_weight), (right_atom, right_weight) in zip(
            items[0::2], items[1::2]
        ):
            left = np.asarray(left_atom.center, dtype=np.float64)
            right = np.asarray(right_atom.center, dtype=np.float64)
            midpoint = 0.5 * (left + right)
            half_gap = 0.5 * float(np.dot(right - left, DIRECTION))
            add_to_next(1, midpoint, float(left_weight + right_weight))
            add_to_next(2, midpoint, float(half_gap * (left_weight - right_weight)))
    elif next_stage == 3:
        zero_indices = {
            atom.order: index for index, atom in enumerate(next_atoms[:-n_ordinary])
        }
        for atom, weight in zip(core_atoms, core_weights):
            displacement = projected_center(atom.center)
            for target_order in range(atom.order, 4):
                exponent = target_order - atom.order
                output[zero_indices[target_order]] += (
                    float(weight) * ((-displacement) ** exponent) / math.factorial(exponent)
                )
    else:
        raise ValueError(f"unknown next stage {next_stage}")

    return torch.tensor(output, dtype=torch.float64)


def loss_on_grid(
    stage: int,
    log_h: torch.Tensor | None,
    ordinary: torch.Tensor,
    coefficients: torch.Tensor,
    grid: TorchGrid,
) -> torch.Tensor:
    basis, _, _ = torch_basis_for_stage(grid.x, grid.y, stage, log_h, ordinary)
    prediction = basis @ coefficients
    return torch.sum(grid.quadrature * (prediction - grid.target) ** 2) / grid.target_norm_sq


def relative_error(loss: torch.Tensor) -> float:
    return max(float(torch.sqrt(torch.clamp(loss.detach(), min=0.0))), ERROR_FLOOR)


def should_record(local_iteration: int, stage_steps: int, global_iteration: int, every: int) -> bool:
    return local_iteration == 0 or local_iteration == stage_steps or global_iteration % every == 0


def new_trace() -> dict[str, list[object]]:
    return {
        "iteration": [],
        "stage": [],
        "training_relative_error": [],
        "relative_error": [],
        "weight_norm": [],
        "collapsing_pair_separation": [],
        "h": [],
        "ordinary_center": [],
    }


def record_state(
    *,
    trace: dict[str, list[object]],
    iteration: int,
    stage: int,
    log_h: torch.Tensor | None,
    ordinary: torch.Tensor,
    coefficients: torch.Tensor,
    training_loss: torch.Tensor,
    validation_grid: TorchGrid,
) -> None:
    with torch.no_grad():
        validation_loss = loss_on_grid(stage, log_h, ordinary, coefficients, validation_grid)
    h = float(torch.exp(log_h).detach()) if log_h is not None else None
    trace["iteration"].append(iteration)
    trace["stage"].append(stage)
    trace["training_relative_error"].append(relative_error(training_loss))
    trace["relative_error"].append(relative_error(validation_loss))
    trace["weight_norm"].append(float(torch.linalg.vector_norm(coefficients.detach())))
    trace["collapsing_pair_separation"].append(collapsing_pair_separation(stage, h))
    trace["h"].append(h)
    trace["ordinary_center"].append(ordinary.detach().cpu().numpy()[0].tolist())


def snapshot(
    *,
    iteration: int,
    stage: int,
    log_h: torch.Tensor | None,
    ordinary: torch.Tensor,
    coefficients: torch.Tensor,
    training_loss: torch.Tensor,
    validation_grid: TorchGrid,
    stage_start_iteration: int,
    stage_start_separation: float | None,
    collapse_threshold: float | None,
    collapse_triggered: bool,
) -> dict[str, object]:
    with torch.no_grad():
        validation_loss = loss_on_grid(stage, log_h, ordinary, coefficients, validation_grid)
    h = float(torch.exp(log_h).detach()) if log_h is not None else None
    ordinary_np = ordinary.detach().cpu().numpy().copy()
    atoms = atoms_for_stage(stage, h, ordinary_np)
    separation = collapsing_pair_separation(stage, h)
    return {
        "iteration": iteration,
        "h": h,
        "training_relative_error": relative_error(training_loss),
        "relative_error": relative_error(validation_loss),
        "weight_norm": float(torch.linalg.vector_norm(coefficients.detach())),
        "collapsing_pair_separation": separation,
        "stage_start_iteration": stage_start_iteration,
        "stage_start_separation": stage_start_separation,
        "collapse_threshold": collapse_threshold,
        "collapse_ratio": (
            separation / stage_start_separation
            if separation is not None
            and stage_start_separation is not None
            and stage_start_separation > 0.0
            else None
        ),
        "collapse_triggered": collapse_triggered,
        "ordinary_centers": ordinary_np.tolist(),
        "basis": [[atom.order, atom.center[0], atom.center[1]] for atom in atoms],
        "weights": coefficients.detach().cpu().numpy().tolist(),
    }


def run_fixed_translated(
    *,
    max_iter: int,
    learning_rate: float,
    record_every: int,
    h_init: float,
    initial_weight_scale: float,
    seed: int,
    training_grid: TorchGrid,
    validation_grid: TorchGrid,
) -> dict[str, object]:
    primary_center = torch.nn.Parameter(torch.tensor(PRIMARY_INITIAL_CENTER, dtype=torch.float64))
    log_h = torch.nn.Parameter(torch.tensor(math.log(h_init), dtype=torch.float64))
    coefficients = torch.nn.Parameter(initialize_weights(WIDTH, seed, initial_weight_scale))
    parameters = [coefficients, log_h, primary_center]
    optimizer = torch.optim.Adam(
        parameters, lr=learning_rate, betas=ADAM_BETAS, eps=ADAM_EPS
    )
    trace = new_trace()
    final: dict[str, object] = {}

    for iteration in range(max_iter + 1):
        ordinary = assemble_ordinary_centers(primary_center)
        training_loss = loss_on_grid(0, log_h, ordinary, coefficients, training_grid)
        objective = training_loss + ordinary_center_guard_penalty(primary_center)
        if iteration == 0 or iteration == max_iter or iteration % record_every == 0:
            record_state(
                trace=trace,
                iteration=iteration,
                stage=0,
                log_h=log_h,
                ordinary=ordinary,
                coefficients=coefficients,
                training_loss=training_loss,
                validation_grid=validation_grid,
            )
        if iteration == max_iter:
            final = snapshot(
                iteration=iteration,
                stage=0,
                log_h=log_h,
                ordinary=ordinary,
                coefficients=coefficients,
                training_loss=training_loss,
                validation_grid=validation_grid,
                stage_start_iteration=0,
                stage_start_separation=2.0 * h_init,
                collapse_threshold=None,
                collapse_triggered=False,
            )
            break
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
        optimizer.step()

    return {
        "width": WIDTH,
        **trace,
        "minimum_center_separation": trace["collapsing_pair_separation"],
        "ordinary_initial_center": PRIMARY_INITIAL_CENTER.tolist(),
        "final": final,
    }


def run_completed(
    *,
    collapse_ratios: tuple[float, float, float],
    max_iter: int,
    learning_rate: float,
    record_every: int,
    h_init: float,
    initial_weight_scale: float,
    seed: int,
    training_grid: TorchGrid,
    validation_grid: TorchGrid,
    final_stage_polish_steps: int,
    final_stage_lbfgs_iter: int,
    final_stage_lbfgs_lr: float,
) -> dict[str, object]:
    current_primary_center = torch.tensor(PRIMARY_INITIAL_CENTER, dtype=torch.float64)
    current_h = h_init
    next_coefficients: torch.Tensor | None = None
    global_iteration = 0
    trace = new_trace()
    summaries: dict[str, dict[str, object]] = {}
    event_iterations: list[int] = []
    collapse_thresholds: list[float] = []
    stage_start_iterations: list[int] = []
    stage_start_separations: list[float | None] = []

    for stage in range(4):
        stage_start = global_iteration
        stage_steps = max_iter - stage_start
        if stage_steps <= 0:
            break
        log_h = (
            torch.nn.Parameter(torch.tensor(math.log(current_h), dtype=torch.float64))
            if stage < 3
            else None
        )
        primary_center = torch.nn.Parameter(current_primary_center.clone().detach())
        n_stage_atoms = (8 if stage < 3 else 4) + 1
        initial = (
            initialize_weights(n_stage_atoms, seed, initial_weight_scale)
            if next_coefficients is None
            else next_coefficients
        )
        coefficients = torch.nn.Parameter(initial.clone().detach())
        parameters: list[torch.nn.Parameter] = [coefficients, primary_center]
        if log_h is not None:
            parameters.append(log_h)

        initial_separation = collapsing_pair_separation(
            stage, current_h if stage < 3 else None
        )
        collapse_threshold = (
            collapse_ratios[stage] * initial_separation
            if stage < 3 and initial_separation is not None
            else None
        )
        stage_start_iterations.append(stage_start)
        stage_start_separations.append(initial_separation)
        if collapse_threshold is not None:
            collapse_thresholds.append(collapse_threshold)

        if stage == 3:
            def final_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                ordinary_local = assemble_ordinary_centers(primary_center)
                training_local = loss_on_grid(
                    stage, None, ordinary_local, coefficients, training_grid
                )
                objective_local = training_local + ordinary_center_guard_penalty(primary_center)
                return training_local, objective_local, ordinary_local

            training_loss, _, ordinary = final_losses()
            record_state(
                trace=trace,
                iteration=stage_start,
                stage=stage,
                log_h=None,
                ordinary=ordinary,
                coefficients=coefficients,
                training_loss=training_loss,
                validation_grid=validation_grid,
            )
            if final_stage_lbfgs_iter > 0:
                lbfgs = torch.optim.LBFGS(
                    parameters,
                    lr=final_stage_lbfgs_lr,
                    max_iter=final_stage_lbfgs_iter,
                    max_eval=max(2 * final_stage_lbfgs_iter, final_stage_lbfgs_iter + 1),
                    tolerance_grad=1e-14,
                    tolerance_change=1e-18,
                    line_search_fn="strong_wolfe",
                )

                def closure() -> torch.Tensor:
                    lbfgs.zero_grad(set_to_none=True)
                    _, objective_local, _ = final_losses()
                    objective_local.backward()
                    return objective_local

                lbfgs.step(closure)

            training_loss, _, ordinary = final_losses()
            polish_end = min(max_iter, stage_start + max(1, final_stage_polish_steps))
            record_state(
                trace=trace,
                iteration=polish_end,
                stage=stage,
                log_h=None,
                ordinary=ordinary,
                coefficients=coefficients,
                training_loss=training_loss,
                validation_grid=validation_grid,
            )
            next_record = ((polish_end // record_every) + 1) * record_every
            for record_iteration in range(next_record, max_iter + 1, record_every):
                record_state(
                    trace=trace,
                    iteration=record_iteration,
                    stage=stage,
                    log_h=None,
                    ordinary=ordinary,
                    coefficients=coefficients,
                    training_loss=training_loss,
                    validation_grid=validation_grid,
                )
            if trace["iteration"][-1] != max_iter:
                record_state(
                    trace=trace,
                    iteration=max_iter,
                    stage=stage,
                    log_h=None,
                    ordinary=ordinary,
                    coefficients=coefficients,
                    training_loss=training_loss,
                    validation_grid=validation_grid,
                )
            summaries["stage_3"] = snapshot(
                iteration=max_iter,
                stage=stage,
                log_h=None,
                ordinary=ordinary,
                coefficients=coefficients,
                training_loss=training_loss,
                validation_grid=validation_grid,
                stage_start_iteration=stage_start,
                stage_start_separation=None,
                collapse_threshold=None,
                collapse_triggered=False,
            )
            summaries["stage_3"].update(
                {
                    "final_stage_optimizer": "LBFGS with strong-Wolfe line search",
                    "final_stage_polish_start_iteration": stage_start,
                    "final_stage_polish_end_iteration": polish_end,
                    "final_stage_lbfgs_iter": final_stage_lbfgs_iter,
                }
            )
            global_iteration = max_iter
            break

        optimizer = torch.optim.Adam(
            parameters, lr=learning_rate, betas=ADAM_BETAS, eps=ADAM_EPS
        )
        for local_iteration in range(stage_steps + 1):
            global_iteration = stage_start + local_iteration
            ordinary = assemble_ordinary_centers(primary_center)
            training_loss = loss_on_grid(
                stage, log_h, ordinary, coefficients, training_grid
            )
            objective = training_loss + ordinary_center_guard_penalty(primary_center)
            if should_record(local_iteration, stage_steps, global_iteration, record_every):
                record_state(
                    trace=trace,
                    iteration=global_iteration,
                    stage=stage,
                    log_h=log_h,
                    ordinary=ordinary,
                    coefficients=coefficients,
                    training_loss=training_loss,
                    validation_grid=validation_grid,
                )

            h = float(torch.exp(log_h).detach())
            separation = collapsing_pair_separation(stage, h)
            collapse_triggered = (
                local_iteration > 0
                and collapse_threshold is not None
                and separation is not None
                and separation <= collapse_threshold
            )
            reached_budget = global_iteration >= max_iter
            if collapse_triggered or reached_budget:
                if (
                    not trace["iteration"]
                    or trace["iteration"][-1] != global_iteration
                    or trace["stage"][-1] != stage
                ):
                    record_state(
                        trace=trace,
                        iteration=global_iteration,
                        stage=stage,
                        log_h=log_h,
                        ordinary=ordinary,
                        coefficients=coefficients,
                        training_loss=training_loss,
                        validation_grid=validation_grid,
                    )
                summaries[f"stage_{stage}"] = snapshot(
                    iteration=global_iteration,
                    stage=stage,
                    log_h=log_h,
                    ordinary=ordinary,
                    coefficients=coefficients,
                    training_loss=training_loss,
                    validation_grid=validation_grid,
                    stage_start_iteration=stage_start,
                    stage_start_separation=initial_separation,
                    collapse_threshold=collapse_threshold,
                    collapse_triggered=collapse_triggered,
                )
                current_primary_center = primary_center.detach().clone()
                if collapse_triggered:
                    event_iterations.append(global_iteration)
                    current_h = h
                    ordinary_np = ordinary.detach().cpu().numpy().copy()
                    previous_atoms = atoms_for_stage(stage, h, ordinary_np)
                    next_coefficients = taylor_initialize_next_stage(
                        previous_atoms,
                        coefficients.detach().cpu().numpy(),
                        stage + 1,
                        h,
                        ordinary_np,
                    )
                else:
                    global_iteration = max_iter
                break

            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()

    return {
        "width": WIDTH,
        **trace,
        "stage_label": [STAGE_LABELS[int(stage)] for stage in trace["stage"]],
        "event_iterations": event_iterations,
        "collapse_ratio_thresholds": list(collapse_ratios),
        "collapse_thresholds": collapse_thresholds,
        "stage_start_iterations": stage_start_iterations,
        "stage_start_separations": stage_start_separations,
        "ordinary_initial_center": PRIMARY_INITIAL_CENTER.tolist(),
        "summaries": summaries,
    }


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.4,
            "ytick.major.size": 3.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.6,
            "legend.fontsize": 8.0,
        }
    )


def plot_diagnostics(result: dict[str, object], fixed_result: dict[str, object]) -> None:
    set_plot_style()
    blue, gray, black, green = "#0000FF", "#666666", "#111111", "#188038"
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25))
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.28, top=0.86, wspace=0.68)
    axes[0].set_title("fixed translated dictionary")
    axes[1].set_title("completed directional dictionary")
    axes[2].set_title("completed weights and separation")

    fixed_iterations = np.asarray(fixed_result["iteration"], dtype=float)
    fixed_error = np.asarray(fixed_result["relative_error"], dtype=float)
    fixed_norm = np.asarray(fixed_result["weight_norm"], dtype=float)
    fixed_norm_axis = axes[0].twinx()
    error_line = axes[0].semilogy(
        fixed_iterations, fixed_error, color=blue, lw=1.2, label="validation error"
    )[0]
    norm_line = fixed_norm_axis.semilogy(
        fixed_iterations, fixed_norm, color=black, lw=1.05, label=r"$\|w\|_2$"
    )[0]
    axes[0].set_ylabel("relative $L^2$ error", color=blue)
    axes[0].tick_params(axis="y", colors=blue)
    fixed_norm_axis.set_ylabel("weight norm")
    axes[0].set_xlabel("iteration")
    axes[0].legend(
        [error_line, norm_line],
        [error_line.get_label(), norm_line.get_label()],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=2,
        handlelength=1.35,
        columnspacing=0.7,
    )

    iterations = np.asarray(result["iteration"], dtype=float)
    error = np.asarray(result["relative_error"], dtype=float)
    norm = np.asarray(result["weight_norm"], dtype=float)
    separation = np.asarray(
        [np.nan if value is None else float(value) for value in result["collapsing_pair_separation"]]
    )
    events = [float(value) for value in result["event_iterations"]]
    event_labels = (r"add $D_vK$", r"add $D_v^2K$", r"add $D_v^3K$")
    axes[1].semilogy(iterations, error, color=blue, lw=1.25)
    for event, label in zip(events, event_labels):
        axes[1].axvline(
            event, color=black if label == event_labels[-1] else gray, lw=0.9, ls="--", label=label
        )
    axes[1].set_ylabel("validation relative $L^2$ error", color=blue)
    axes[1].tick_params(axis="y", colors=blue)
    axes[1].set_xlabel("iteration")
    if events:
        axes[1].legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.29),
            ncol=len(events),
            handlelength=1.2,
            columnspacing=0.55,
        )

    separation_axis = axes[2].twinx()
    axes[2].semilogy(iterations, norm, color=black, lw=1.15)
    separation_axis.semilogy(iterations, separation, color=green, lw=1.05, ls="-.")
    for event in events:
        axes[2].axvline(event, color=gray, lw=0.9, ls="--")
    for start, event, threshold in zip(
        result["stage_start_iterations"], events, result["collapse_thresholds"]
    ):
        separation_axis.hlines(
            float(threshold), float(start), float(event), color=green, lw=0.9, ls=":"
        )
    axes[2].set_ylabel("weight norm")
    separation_axis.set_ylabel("active pair separation", color=green)
    separation_axis.tick_params(axis="y", colors=green)
    axes[2].set_xlabel("iteration")
    dummy_norm = axes[2].plot([], [], color=black, lw=1.15, label=r"$\|w\|_2$")[0]
    dummy_sep = separation_axis.plot(
        [], [], color=green, lw=1.05, ls="-.", label=r"$\delta_v$"
    )[0]
    dummy_threshold = separation_axis.plot(
        [], [], color=green, lw=0.9, ls=":", label="trigger level"
    )[0]
    axes[2].legend(
        [dummy_norm, dummy_sep, dummy_threshold],
        [dummy_norm.get_label(), dummy_sep.get_label(), dummy_threshold.get_label()],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=3,
        handlelength=1.2,
        columnspacing=0.55,
    )
    for axis in axes:
        axis.grid(False)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_representation_evolution(
    result: dict[str, object], domain_radius: float, n_plot: int
) -> None:
    set_plot_style()
    order_colors = {0: "#0000FF", 1: "#D55E00", 2: "#7B3294", 3: "#188038"}
    order_markers = {0: "o", 1: "s", 2: "^", 3: "D"}
    ordinary_color, gray, target_color = "#E69F00", "#6A6A6A", "#111111"
    nodes_x, nodes_y, x_plot, y_plot, _ = tensor_grid(n_plot, domain_radius)
    xx, yy = np.meshgrid(nodes_x, nodes_y, indexing="xy")
    target_plot = target_fn(x_plot, y_plot).reshape(n_plot, n_plot)
    value_limit = 1.05 * max(float(np.max(np.abs(target_plot))), 1e-12)
    fig = plt.figure(figsize=(4.685, 6.65))
    grid = fig.add_gridspec(
        6, 2,
        height_ratios=(0.96, 1.45, 1.15, 0.96, 1.45, 1.15),
        left=0.14, right=0.93, bottom=0.070, top=0.955,
        wspace=0.28, hspace=0.66
    )
    axes = np.asarray(
        [[fig.add_subplot(grid[row, column]) for column in range(2)] for row in range(6)]
    )
    summaries = result["summaries"]
    iterations = np.asarray(result["iteration"], dtype=float)
    validation_error = np.asarray(result["relative_error"], dtype=float)
    events = [float(value) for value in result["event_iterations"]]
    stage_notes = (
        r"$8$ translated atoms",
        r"$4$ collapsing $D_vK$ atoms",
        r"$2$ collapsing $D_v^2K$ atoms",
        r"explicit $D_v^3K$ atom",
    )
    all_centers = np.asarray(
        [
            (float(center_x), float(center_y))
            for stage in range(4)
            for _, center_x, center_y in summaries[f"stage_{stage}"]["basis"]
        ]
        + [(0.0, 0.0)],
        dtype=np.float64,
    )
    x_span = max(float(np.ptp(all_centers[:, 0])), 1.0)
    y_span = max(float(np.ptp(all_centers[:, 1])), 1.0)
    center_xlim = (
        float(np.min(all_centers[:, 0]) - 0.08 * x_span),
        float(np.max(all_centers[:, 0]) + 0.08 * x_span),
    )
    center_ylim = (
        float(np.min(all_centers[:, 1]) - 0.08 * y_span),
        float(np.max(all_centers[:, 1]) + 0.08 * y_span),
    )

    for stage in range(4):
        block_row = 3 * (stage // 2)
        column = stage % 2
        summary = summaries[f"stage_{stage}"]
        atoms = [
            Atom2D(int(order), (float(center_x), float(center_y)))
            for order, center_x, center_y in summary["basis"]
        ]
        coefficients = np.asarray(summary["weights"], dtype=np.float64)
        represented = (basis_from_atoms(x_plot, y_plot, atoms) @ coefficients).reshape(
            n_plot, n_plot
        )
        center_axis = axes[block_row, column]
        max_log_weight = max(float(np.max(np.log1p(np.abs(coefficients)))), 1e-12)
        center_axis.axhline(0.0, color="#CFCFCF", lw=0.7, zorder=0)
        center_axis.axvline(0.0, color="#CFCFCF", lw=0.7, zorder=0)
        center_axis.quiver(
            [0.0], [0.0], [0.55 * DIRECTION[0]], [0.55 * DIRECTION[1]],
            angles="xy", scale_units="xy", scale=1.0, color=gray, width=0.012, alpha=0.8
        )
        for index, (atom, coefficient) in enumerate(zip(atoms, coefficients)):
            ordinary = index == len(atoms) - 1
            color = ordinary_color if ordinary else order_colors[atom.order]
            alpha = 0.92 if ordinary or atom.order == stage else 0.42
            size = 13.0 + 55.0 * np.log1p(abs(coefficient)) / max_log_weight
            center_axis.scatter(
                [atom.center[0]], [atom.center[1]], s=size, color=color,
                marker="*" if ordinary else order_markers[atom.order],
                edgecolor="white", linewidth=0.45, alpha=alpha, zorder=3
            )
        center_axis.set_xlim(*center_xlim)
        center_axis.set_ylim(*center_ylim)
        center_axis.set_aspect("equal", adjustable="box")
        center_axis.set_xticks([])
        center_axis.set_yticks([])
        center_axis.set_title(
            STAGE_LABELS[stage] + "\n" + stage_notes[stage], pad=3.0, fontsize=8.4
        )
        if column == 0:
            center_axis.text(
                -0.13, 0.5, "centers", rotation=90, va="center", ha="center",
                transform=center_axis.transAxes, fontsize=8.2
            )

        field_axis = axes[block_row + 1, column]
        image = field_axis.imshow(
            represented, extent=(-domain_radius, domain_radius, -domain_radius, domain_radius),
            origin="lower", cmap="coolwarm", vmin=-value_limit, vmax=value_limit,
            interpolation="bilinear", rasterized=True
        )
        field_axis.contour(
            xx, yy, target_plot, levels=7, colors=target_color, linewidths=0.45, alpha=0.78
        )
        field_axis.scatter(
            [TARGET_CENTER[0]], [TARGET_CENTER[1]], s=12, color=ordinary_color,
            marker="*", edgecolor="white", linewidth=0.35
        )
        field_axis.set_aspect("equal", adjustable="box")
        field_axis.set_xticks([])
        field_axis.set_yticks([])
        if stage == 3:
            position = field_axis.get_position()
            color_axis = fig.add_axes([position.x1 + 0.010, position.y0, 0.018, position.height])
            colorbar = fig.colorbar(image, cax=color_axis)
            colorbar.ax.tick_params(labelsize=8.0, width=0.55, length=2.4)
        if column == 0:
            field_axis.text(
                -0.13, 0.5, "represented field", rotation=90, va="center", ha="center",
                transform=field_axis.transAxes, fontsize=8.2
            )

        error_axis = axes[block_row + 2, column]
        current_iteration = float(summary["iteration"])
        mask = iterations <= current_iteration
        error_axis.semilogy(iterations[mask], validation_error[mask], color="#0000FF", lw=1.35)
        for event in events:
            if event <= current_iteration:
                error_axis.axvline(event, color=gray, lw=0.75, ls="--", alpha=0.62)
        error_axis.scatter(
            [current_iteration], [float(summary["relative_error"])], s=14, color="#0000FF",
            edgecolor="white", linewidth=0.35, zorder=3
        )
        error_axis.set_ylim(ERROR_FLOOR, 1e1)
        error_axis.set_xlim(0.0, float(np.max(iterations)))
        error_axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        error_axis.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _position: f"{value / 1.0e4:g}")
        )
        if block_row == 0:
            error_axis.tick_params(axis="x", labelbottom=False)
        error_axis.text(
            0.96, 0.08,
            rf"$\mathrm{{err}}_{{\rm val}}={summary['relative_error']:.1e}$"
            + "\n" + rf"$\|w\|_2={summary['weight_norm']:.1e}$",
            transform=error_axis.transAxes, ha="right", va="bottom", fontsize=8.0,
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD",
                  "boxstyle": "round,pad=0.2", "alpha": 0.88}
        )
        if column == 0:
            error_axis.set_ylabel("validation error")
        error_axis.grid(False)

    fig.supxlabel(r"iteration ($\times 10^4$)", x=0.535, y=0.016, fontsize=8.2)
    fig.savefig(OUT_REP_PDF)
    fig.savefig(OUT_REP_PNG, dpi=240)
    plt.close(fig)


def configure_outputs(output_directory: Path, output_suffix: str) -> None:
    global OUT_PDF, OUT_PNG, OUT_REP_PDF, OUT_REP_PNG
    output_directory.mkdir(parents=True, exist_ok=True)
    suffix = output_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    OUT_PDF = output_directory / f"{OUT_STEM}{suffix}.pdf"
    OUT_PNG = output_directory / f"{OUT_STEM}{suffix}.png"
    OUT_REP_PDF = output_directory / f"{OUT_REP_STEM}{suffix}.pdf"
    OUT_REP_PNG = output_directory / f"{OUT_REP_STEM}{suffix}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grid", type=int, default=31)
    parser.add_argument("--validation-grid", type=int, default=121)
    parser.add_argument("--domain-radius", type=float, default=6.0)
    parser.add_argument("--plot-grid", type=int, default=151)
    parser.add_argument("--plot-domain-radius", type=float, default=3.0)
    parser.add_argument("--h-init", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=ADAM_LEARNING_RATE)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument(
        "--collapse-ratios", type=float, nargs=3, default=list(COLLAPSE_RATIOS)
    )
    parser.add_argument("--record-every", type=int, default=RECORD_EVERY)
    parser.add_argument("--initial-weight-scale", type=float, default=INITIAL_WEIGHT_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--final-stage-polish-steps", type=int, default=1000)
    parser.add_argument("--final-stage-lbfgs-iter", type=int, default=200)
    parser.add_argument("--final-stage-lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument(
        "--cache-output",
        type=Path,
        help=(
            "JSON record written by a full run; defaults to data/completion "
            "with the selected output suffix"
        ),
    )
    parser.add_argument("--output-suffix", default="adam")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--sync-figures",
        action="store_true",
        help="update the representation PNG preview in figures/previews",
    )
    parser.add_argument(
        "--plot-from-cache",
        type=Path,
        metavar="JSON",
        help="redraw the representation figure from this JSON without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_outputs(args.output_directory, args.output_suffix)
    cache_output = args.cache_output
    if cache_output is None:
        suffix = args.output_suffix.strip()
        if suffix and not suffix.startswith("_"):
            suffix = f"_{suffix}"
        cache_output = COMPLETION_DATA_DIRECTORY / f"{OUT_STEM}{suffix}.json"
    if args.plot_from_cache is not None:
        if not args.plot_from_cache.is_file():
            raise FileNotFoundError(args.plot_from_cache)
        cached = json.loads(args.plot_from_cache.read_text())
        cached_results = cached.get("results")
        if not isinstance(cached_results, list) or len(cached_results) != 1:
            raise ValueError("cached JSON must contain exactly one completed result")
        plot_representation_evolution(
            cached_results[0], args.plot_domain_radius, args.plot_grid
        )
        if args.sync_figures:
            PREVIEW_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OUT_REP_PNG, PREVIEW_FIGURE_DIRECTORY / OUT_REP_PNG.name)
        print(f"Wrote {OUT_REP_PDF} from {args.plot_from_cache}")
        print(f"Wrote {OUT_REP_PNG} from {args.plot_from_cache}")
        return
    collapse_ratios = tuple(float(value) for value in args.collapse_ratios)
    if len(collapse_ratios) != 3 or any(
        value <= 0.0 or value >= 1.0 for value in collapse_ratios
    ):
        raise ValueError("expected three collapse ratios in (0,1)")
    if args.n_grid < 3 or args.validation_grid < 3:
        raise ValueError("training and validation grids need at least three nodes per axis")
    if args.n_grid == args.validation_grid:
        raise ValueError("use different training and validation grids")
    if args.domain_radius <= 0.0 or args.plot_domain_radius <= 0.0:
        raise ValueError("domain radii must be positive")
    if args.h_init <= 0.0:
        raise ValueError("h_init must be positive")
    if args.max_iter < 1 or args.record_every < 1:
        raise ValueError("iteration counts must be positive")
    if args.final_stage_lbfgs_iter < 0 or args.final_stage_lbfgs_lr <= 0.0:
        raise ValueError("invalid LBFGS configuration")
    torch.set_num_threads(1)

    _, _, train_x, train_y, train_quadrature = tensor_grid(args.n_grid, args.domain_radius)
    _, _, validation_x, validation_y, validation_quadrature = tensor_grid(
        args.validation_grid, args.domain_radius
    )
    training_grid = make_torch_grid(train_x, train_y, train_quadrature)
    validation_grid = make_torch_grid(validation_x, validation_y, validation_quadrature)

    result = run_completed(
        collapse_ratios=collapse_ratios,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        record_every=args.record_every,
        h_init=args.h_init,
        initial_weight_scale=args.initial_weight_scale,
        seed=args.seed,
        training_grid=training_grid,
        validation_grid=validation_grid,
        final_stage_polish_steps=args.final_stage_polish_steps,
        final_stage_lbfgs_iter=args.final_stage_lbfgs_iter,
        final_stage_lbfgs_lr=args.final_stage_lbfgs_lr,
    )
    fixed_result = run_fixed_translated(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        record_every=args.record_every,
        h_init=args.h_init,
        initial_weight_scale=args.initial_weight_scale,
        seed=args.seed,
        training_grid=training_grid,
        validation_grid=validation_grid,
    )
    completed_all_stages = all(
        f"stage_{stage}" in result["summaries"] for stage in range(4)
    )
    if not args.no_plots:
        plot_diagnostics(result, fixed_result)
        if completed_all_stages:
            plot_representation_evolution(result, args.plot_domain_radius, args.plot_grid)
        else:
            print("Skipped the representation figure because not all collapse triggers were reached.")

    payload = {
        "kernel": "K(z;c)=exp(-|z-c|^2/2)",
        "target": (
            f"D_v^3 K(z;0) + {BETA:g} "
            f"K(z;({TARGET_CENTER[0]:g},{TARGET_CENTER[1]:g}))"
        ),
        "direction": DIRECTION.tolist(),
        "width": WIDTH,
        "ordinary_initial_center": PRIMARY_INITIAL_CENTER.tolist(),
        "ordinary_target_center": TARGET_CENTER.tolist(),
        "ordinary_center_guard": ORDINARY_CENTER_GUARD,
        "ordinary_center_guard_weight": ORDINARY_CENTER_GUARD_WEIGHT,
        "training_quadrature": {
            "rule": "tensor-product trapezoidal",
            "domain": [-args.domain_radius, args.domain_radius] * 2,
            "nodes_per_axis": args.n_grid,
        },
        "validation_quadrature": {
            "rule": "independent tensor-product trapezoidal grid",
            "domain": [-args.domain_radius, args.domain_radius] * 2,
            "nodes_per_axis": args.validation_grid,
        },
        "optimizer": (
            f"full-batch CPU float64 Adam, lr={args.learning_rate:g}, "
            f"betas={ADAM_BETAS}, eps={ADAM_EPS:g}, "
            f"gradient-norm clipping at {GRAD_CLIP:g}"
        ),
        "seed": args.seed,
        "collapse_ratio_thresholds": list(collapse_ratios),
        "event_iterations": result["event_iterations"],
        "max_iteration": args.max_iter,
        "h_init": args.h_init,
        "record_every": args.record_every,
        "final_stage_optimizer": "LBFGS with strong-Wolfe line search",
        "final_stage_polish_steps": args.final_stage_polish_steps,
        "final_stage_lbfgs_iter": args.final_stage_lbfgs_iter,
        "final_stage_lbfgs_lr": args.final_stage_lbfgs_lr,
        "description": (
            "P=9 directional 2D boundary completion. Adam trains the weights, "
            "logarithmic stencil scale, and displaced ordinary center. Each "
            "derivative insertion is triggered by the observed active-pair "
            "separation and Taylor-initialized from the previous stage. "
            "Reported relative errors are unpenalized validation-grid errors. "
            "The fixed-translated control uses the same initial state, Adam "
            "settings, and iteration budget."
        ),
        "completed_all_stages": completed_all_stages,
        "results": [result],
        "fixed_translated_control": [fixed_result],
    }
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    cache_output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.sync_figures and not args.no_plots and completed_all_stages:
        PREVIEW_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_REP_PNG, PREVIEW_FIGURE_DIRECTORY / OUT_REP_PNG.name)
    if not args.no_plots:
        print(f"Wrote {OUT_PDF}")
        print(f"Wrote {OUT_PNG}")
        if completed_all_stages:
            print(f"Wrote {OUT_REP_PDF}")
            print(f"Wrote {OUT_REP_PNG}")
    print(f"Wrote {cache_output}")
    print(
        f"P={WIDTH}: triggers={result['event_iterations']}; "
        f"completed_all_stages={completed_all_stages}; "
        f"fixed final validation error={fixed_result['final']['relative_error']:.3e}"
    )


if __name__ == "__main__":
    main()
