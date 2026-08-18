"""Validate or reproduce the numerical artifacts used in the paper.

The repository intentionally ships the experiment sources, curated caches,
and manuscript-facing figures without duplicating the manuscript source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
FIGURE_DIRECTORY = WORKSPACE / "figures" / "manuscript"
DESCRIPTIVE_FIGURE_DIRECTORY = WORKSPACE / "figures" / "descriptive"
REPRODUCTION_OUTPUT_DIRECTORY = WORKSPACE / "reproduced"

EXPECTED_VERSIONS = {
    "numpy": "2.4.2",
    "matplotlib": "3.10.8",
    "torch": "2.10.0",
}

# Every PDF reachable through \includegraphics in the canonical source must
# have an entry here.  Entries list the executable source and raw/cache inputs
# needed to audit or reproduce the displayed result.
FIGURE_PROVENANCE: dict[str, dict[str, tuple[Path, ...]]] = {
    "matched_elliptic_completion.pdf": {
        "scripts": (HERE / "matched_elliptic_completion.py",),
        "data": (
            HERE / "matched_elliptic_completion.json",
            HERE / "matched_elliptic_completion_summary.csv",
        ),
    },
    "wb_trajectories.pdf": {
        "scripts": (
            HERE / "tanh_parameter_escape_loss.py",
            HERE / "tanh_wb_trajectories.py",
        ),
        "data": (
            HERE / "tanh_parameter_escape_loss.json",
            HERE / "tanh_parameter_escape_loss_float32.json",
        ),
    },
    "tanh_x_tanhprime_energy_minimization.pdf": {
        "scripts": (HERE / "tanh_slope_collision_snapshots.py",),
        "data": (HERE / "tanh_x_tanhprime_energy_minimization.json",),
    },
    "gaussian_rbf_instability_unpenalized_mixed_precision.pdf": {
        "scripts": (
            HERE / "gaussian_rbf_instability_figure5_mixed_precision.py",
            HERE / "tanh_instability_figure5_mixed_precision.py",
            HERE / "tanh_instability_figure5.py",
            HERE / "tanh_second_derivative_optimizer.c",
        ),
        "data": (HERE / "gaussian_rbf_instability_unpenalized_mixed_precision.json",),
    },
    "gaussian_rbf_instability_unpenalized_mixed_precision_diagnostics.pdf": {
        "scripts": (HERE / "gaussian_rbf_trace_diagnostics.py",),
        "data": (HERE / "gaussian_rbf_instability_unpenalized_mixed_precision.json",),
    },
    "gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.pdf": {
        "scripts": (
            HERE / "gaussian_instability_adaptive_weight_penalty.py",
            HERE / "gaussian_instability_weight_penalty.py",
            HERE / "gaussian_instability_experiment.py",
        ),
        "data": (HERE / "gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json",),
    },
    "boundary_completion_third_derivative_atoms_lbfgs_plateau.pdf": {
        "scripts": (
            HERE / "boundary_completion_experiment.py",
            HERE / "boundary_completion_matched_control_lbfgs.py",
        ),
        "data": (
            HERE / "boundary_completion_third_derivative_atoms_lbfgs_plateau.json",
            HERE / "boundary_completion_matched_control_lbfgs.json",
        ),
    },
    "boundary_completion_third_derivative_representation_lbfgs_plateau.pdf": {
        "scripts": (HERE / "boundary_completion_experiment.py",),
        "data": (HERE / "boundary_completion_third_derivative_atoms_lbfgs_plateau.json",),
    },
    "boundary_completion_2d_representation_adam.pdf": {
        "scripts": (
            HERE / "boundary_completion_2d_experiment.py",
            HERE / "boundary_completion_2d_matched_control_lbfgs.py",
        ),
        "data": (
            HERE / "boundary_completion_2d_atoms_adam.json",
            HERE / "boundary_completion_2d_matched_control_lbfgs.json",
        ),
    },
    "boundary_completion_2d_representation.pdf": {
        "scripts": (
            HERE / "boundary_completion_2d_experiment.py",
            HERE / "boundary_completion_2d_matched_control_lbfgs.py",
        ),
        "data": (
            HERE / "boundary_completion_2d_atoms_adam.json",
            HERE / "boundary_completion_2d_matched_control_lbfgs.json",
        ),
    },
    "ingham_panel_T_2_5_10_an1_styled.pdf": {
        "scripts": (HERE / "ingham_illustration.py",),
        "data": (HERE / "ingham_panel_T_2_5_10_an1_styled.json",),
    },
}

NUMBERED_FIGURES = {
    "Fig1.pdf": "wb_trajectories.pdf",
    "Fig2.pdf": "tanh_x_tanhprime_energy_minimization.pdf",
    "Fig3.pdf": "gaussian_rbf_instability_unpenalized_mixed_precision_diagnostics.pdf",
    "Fig4.pdf": "gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.pdf",
    "Fig5.pdf": "boundary_completion_third_derivative_atoms_lbfgs_plateau.pdf",
    "Fig6.pdf": "boundary_completion_third_derivative_representation_lbfgs_plateau.pdf",
    "Fig7.pdf": "boundary_completion_2d_representation_adam.pdf",
    "Fig8.pdf": "ingham_panel_T_2_5_10_an1_styled.pdf",
}

# The manuscript PDFs are preserved separately from regenerated audit output.
# Figures 1--7 retain the selected wide presentation; preserving Figure 8 also
# prevents harmless PDF metadata changes from altering its publication bytes.
PRESERVED_FIGURE_DIRECTORY = WORKSPACE / "figures" / "preserved"
PRESERVED_MANUSCRIPT_FIGURES = frozenset(NUMBERED_FIGURES.values())

# Expected MediaBox widths for the seven preserved wide figures and the
# journal-width Ingham illustration. Ranges allow harmless PDF writer rounding.
EXPECTED_FIGURE_WIDTHS = {
    "Fig1.pdf": (519.0, 521.0),
    "Fig2.pdf": (1278.0, 1280.0),
    "Fig3.pdf": (509.0, 511.0),
    "Fig4.pdf": (945.0, 947.0),
    "Fig5.pdf": (928.0, 930.0),
    "Fig6.pdf": (790.0, 792.0),
    "Fig7.pdf": (810.0, 812.0),
    "Fig8.pdf": (330.0, 340.0),
}
for numbered_name, descriptive_name in NUMBERED_FIGURES.items():
    FIGURE_PROVENANCE[numbered_name] = FIGURE_PROVENANCE[descriptive_name]


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require_file(path: Path, minimum_size: int = 1) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < minimum_size:
        raise ValueError(f"artifact is unexpectedly small: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checksum_manifest(path: Path) -> int:
    require_file(path)
    checked = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"invalid checksum line {path}:{line_number}")
        expected, relative_name = fields
        target = WORKSPACE / relative_name.strip()
        require_file(target)
        actual = sha256(target)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {relative_name}: expected {expected}, found {actual}"
            )
        checked += 1
    if checked == 0:
        raise ValueError(f"empty checksum manifest: {path}")
    return checked


def included_figures() -> set[str]:
    """Return the eight numbered figures included in the submitted paper."""
    return set(NUMBERED_FIGURES)


def check_figure_provenance() -> set[str]:
    figures = included_figures()
    unmapped = figures.difference(FIGURE_PROVENANCE)
    if unmapped:
        raise ValueError(f"manuscript figures without provenance entries: {sorted(unmapped)}")
    for figure in sorted(figures):
        manuscript_figure = FIGURE_DIRECTORY / figure
        require_file(manuscript_figure, minimum_size=1_000)
        pdf_bytes = manuscript_figure.read_bytes()
        if re.search(rb"/Subtype\s*/Type3\b", pdf_bytes):
            raise ValueError(f"Type 3 font found in manuscript figure: {figure}")
        media_box = re.search(
            rb"/MediaBox\s*\[\s*0(?:\.0*)?\s+0(?:\.0*)?\s+([0-9.]+)",
            pdf_bytes,
        )
        if media_box is None:
            raise ValueError(f"could not read PDF MediaBox width: {figure}")
        width_points = float(media_box.group(1))
        lower_width, upper_width = EXPECTED_FIGURE_WIDTHS[figure]
        if not lower_width <= width_points <= upper_width:
            raise ValueError(
                f"{figure} width is {width_points:.2f} pt; expected "
                f"{lower_width:.0f}--{upper_width:.0f} pt"
            )
        provenance = FIGURE_PROVENANCE[figure]
        for script in provenance["scripts"]:
            require_file(script)
        for data in provenance["data"]:
            require_file(data)
        if figure in NUMBERED_FIGURES:
            descriptive = DESCRIPTIVE_FIGURE_DIRECTORY / NUMBERED_FIGURES[figure]
            require_file(descriptive, minimum_size=1_000)
            if sha256(manuscript_figure) != sha256(descriptive):
                raise ValueError(
                    f"{figure} is stale relative to its descriptive source copy "
                    f"{descriptive.name}"
                )
            if descriptive.name in PRESERVED_MANUSCRIPT_FIGURES:
                preserved = PRESERVED_FIGURE_DIRECTORY / descriptive.name
                require_file(preserved, minimum_size=1_000)
                if sha256(descriptive) != sha256(preserved):
                    raise ValueError(
                        f"{descriptive.name} no longer matches its preserved "
                        "pre-revision presentation"
                    )
    return figures


def check_tanh_offset() -> tuple[int, int]:
    counts: list[int] = []
    for precision, path in (
        ("float64", HERE / "tanh_parameter_escape_loss.json"),
        ("float32", HERE / "tanh_parameter_escape_loss_float32.json"),
    ):
        payload = read_json(path)
        history = payload.get("history")
        if not isinstance(history, list) or len(history) < 2:
            raise ValueError(f"invalid tanh {precision} history")
        final = history[-1]
        for key in ("iteration", "energy_gap", "w1", "b1", "w2", "b2"):
            if key not in final:
                raise ValueError(f"missing {key} in tanh {precision} final record")
        counts.append(len(history))
    return counts[0], counts[1]


def check_matched_pde() -> tuple[float, float]:
    payload = read_json(HERE / "matched_elliptic_completion.json")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("matched PDE cache must contain two coordinate systems")
    labels = [str(result.get("name")) for result in results]
    if labels != ["translated", "completed"]:
        raise ValueError(f"unexpected matched PDE labels: {labels}")
    errors = tuple(float(result["final_relative_h1_error"]) for result in results)
    conditions = tuple(
        float(result["normalized_h1_gram_condition_number"]) for result in results
    )
    if not all(error > 0.0 for error in errors):
        raise ValueError("invalid matched PDE errors")
    if not all(condition >= 1.0 for condition in conditions):
        raise ValueError("invalid matched PDE Gram conditions")
    return errors


def check_tanh_slope() -> int:
    payload = read_json(HERE / "tanh_x_tanhprime_energy_minimization.json")
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 4:
        raise ValueError("tanh slope cache must contain four displayed snapshots")
    if payload.get("random_seed", "missing") is not None:
        raise ValueError("the deterministic tanh slope calculation must have no random seed")
    for snapshot in snapshots:
        if float(snapshot["relative_l2_error"]) <= 0.0:
            raise ValueError("invalid tanh slope error")
    return len(snapshots)


def check_gaussian() -> tuple[float, float]:
    payload = read_json(HERE / "gaussian_rbf_instability_unpenalized_mixed_precision.json")
    orders = payload.get("orders")
    if not isinstance(orders, list) or len(orders) != 1 or int(orders[0]["order"]) != 1:
        raise ValueError("the paper Gaussian cache must contain exactly the order-one run")
    final_errors: list[float] = []
    for precision in ("float32", "float64"):
        record = orders[0][precision]
        trace = record["trace"]
        realized = record["realized_relative_error"]
        if len(trace["iteration"]) != len(realized):
            raise ValueError(f"inconsistent Gaussian {precision} trace")
        if float(trace["separation"][-1]) <= 0.0:
            raise ValueError(f"nonpositive Gaussian {precision} separation")
        final_errors.append(float(realized[-1]))
    return final_errors[0], final_errors[1]


def check_penalty() -> tuple[float, float]:
    payload = read_json(HERE / "gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json")
    runs = payload.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"float32", "float64"}:
        raise ValueError("adaptive penalty cache must contain both precision runs")
    errors = []
    for precision in ("float32", "float64"):
        trace = runs[precision].get("optimizer_trace")
        if not isinstance(trace, dict) or len(trace.get("iteration", [])) < 2:
            raise ValueError(f"invalid adaptive penalty {precision} history")
        errors.append(float(runs[precision]["final"]["rel_error_float64"]))
    return errors[0], errors[1]


def check_completion() -> tuple[list[int], float]:
    source = read_json(HERE / "boundary_completion_third_derivative_atoms_lbfgs_plateau.json")
    widths = [int(value) for value in source.get("widths", [])]
    if widths != [9, 10, 12]:
        raise ValueError(f"unexpected one-dimensional completion widths: {widths}")
    if len(source.get("event_iterations", [])) != 3:
        raise ValueError("invalid one-dimensional completion histories")
    matched = read_json(HERE / "boundary_completion_matched_control_lbfgs.json")
    controls = matched.get("matched_fixed_controls")
    completed = matched.get("completed_stage_3_validation")
    if not isinstance(controls, list) or [int(row["width"]) for row in controls] != widths:
        raise ValueError("invalid matched one-dimensional controls")
    if not isinstance(completed, list) or [int(row["width"]) for row in completed] != widths:
        raise ValueError("invalid completed one-dimensional validation cache")

    source_2d_path = HERE / "boundary_completion_2d_atoms_adam.json"
    source_2d = read_json(source_2d_path)
    if int(source_2d.get("seed", -1)) != 80_209:
        raise ValueError("unexpected two-dimensional completion seed")
    matched_2d = read_json(HERE / "boundary_completion_2d_matched_control_lbfgs.json")
    embedded_source = matched_2d.get("source")
    if not isinstance(embedded_source, dict):
        raise ValueError("matched two-dimensional cache has no source record")
    if embedded_source.get("sha256") != sha256(source_2d_path):
        raise ValueError("matched two-dimensional cache is stale relative to its source")
    terminal = matched_2d.get("terminal_metrics")
    if not isinstance(terminal, list) or len(terminal) != 3:
        raise ValueError("invalid matched two-dimensional terminal metrics")
    for row, expected_width in zip(terminal, (9, 9, 5), strict=True):
        coefficients = row.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != expected_width:
            raise ValueError("invalid matched two-dimensional coefficients")
        if float(row["validation_relative_l2"]) <= 0.0:
            raise ValueError("invalid matched two-dimensional validation error")
    return widths, float(terminal[-1]["validation_relative_l2"])


def check_ingham() -> int:
    payload = read_json(HERE / "ingham_panel_T_2_5_10_an1_styled.json")
    if payload.get("windows") != [2.0, 5.0, 10.0]:
        raise ValueError("unexpected T values in the Ingham illustration")
    node_count = int(payload.get("delta_points", 0))
    if node_count < 50:
        raise ValueError("unexpected Ingham delta grid")
    for key in ("exact_integral_formula", "lower_bound_formula"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"missing {key} in the Ingham cache")
    return node_count


def installed_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for distribution in EXPECTED_VERSIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def check_versions(allow_version_drift: bool = False) -> dict[str, str]:
    versions = installed_versions()
    for distribution, expected in EXPECTED_VERSIONS.items():
        # CPU-only PyTorch wheels append a PEP 440 local-version tag such as
        # ``+cpu`` while retaining the same upstream release.
        installed_release = versions[distribution].partition("+")[0]
        if installed_release != expected:
            message = (
                f"{distribution} version mismatch: expected {expected}, "
                f"found {versions[distribution]}"
            )
            if not allow_version_drift:
                raise ValueError(message)
            print(f"WARNING: {message}", file=sys.stderr)
    return versions


def check_artifacts(
    allow_version_drift: bool = False,
    verify_checksums: bool = True,
) -> None:
    require_file(WORKSPACE / "requirements-lock.txt")
    environment = read_json(WORKSPACE / "execution_environment.json")
    if environment.get("captured_for_artifact_version") != "1.0.0":
        raise ValueError("unexpected execution-environment artifact version")
    figures = check_figure_provenance()
    versions = check_versions(allow_version_drift=allow_version_drift)
    tanh_counts = check_tanh_offset()
    matched_errors = check_matched_pde()
    slope_count = check_tanh_slope()
    gaussian_errors = check_gaussian()
    penalty_errors = check_penalty()
    widths, completed_2d_error = check_completion()
    ingham_nodes = check_ingham()
    data_hashes = None
    figure_hashes = None
    if verify_checksums:
        data_hashes = check_checksum_manifest(WORKSPACE / "DATA_SHA256SUMS")
        figure_hashes = check_checksum_manifest(WORKSPACE / "FIGURE_SHA256SUMS")

    print("Artifact check passed for the curated experiment repository.")
    print("  versions: " + ", ".join(f"{key}={value}" for key, value in versions.items()))
    print(f"  mapped manuscript figures: {len(figures)}")
    print(f"  tanh offset records: float64={tanh_counts[0]}, float32={tanh_counts[1]}")
    print(
        "  matched weak-PDE final relative H1 errors: "
        f"translated={matched_errors[0]:.6e}, completed={matched_errors[1]:.6e}"
    )
    print(f"  tanh slope snapshots: {slope_count}")
    print(
        "  Gaussian final relative L2 errors: "
        f"float32={gaussian_errors[0]:.6e}, float64={gaussian_errors[1]:.6e}"
    )
    print(
        "  penalized Gaussian final float64-evaluated errors: "
        f"float32 path={penalty_errors[0]:.6e}, float64 path={penalty_errors[1]:.6e}"
    )
    print(f"  completion widths: {widths}; completed 2D error={completed_2d_error:.6e}")
    print(f"  Ingham delta nodes per panel: {ingham_nodes}")
    if verify_checksums:
        print(f"  verified checksums: data={data_hashes}, manuscript figures={figure_hashes}")
    else:
        print("  checksum comparison skipped after regenerating numerical caches")


def run(command: list[str], environment: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, env=environment, check=True)


def sync_from_code(names: tuple[str, ...]) -> None:
    DESCRIPTIVE_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(HERE / name, DESCRIPTIVE_FIGURE_DIRECTORY / name)


def sync_numbered_figures() -> None:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for numbered_name, descriptive_name in NUMBERED_FIGURES.items():
        shutil.copy2(
            DESCRIPTIVE_FIGURE_DIRECTORY / descriptive_name,
            FIGURE_DIRECTORY / numbered_name,
        )


def sync_preserved_figure_layouts() -> None:
    DESCRIPTIVE_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for descriptive_name in sorted(PRESERVED_MANUSCRIPT_FIGURES):
        source = PRESERVED_FIGURE_DIRECTORY / descriptive_name
        require_file(source, minimum_size=1_000)
        shutil.copy2(source, DESCRIPTIVE_FIGURE_DIRECTORY / descriptive_name)


def configured_environment(cache_directory: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = cache_directory
    environment["MPLBACKEND"] = "Agg"
    environment["XDG_CACHE_HOME"] = cache_directory
    environment["SOURCE_DATE_EPOCH"] = "1785888000"
    return environment


def run_fast_plots() -> None:
    with tempfile.TemporaryDirectory(prefix="coercivity-mpl-") as cache_directory:
        environment = configured_environment(cache_directory)
        run(
            [
                sys.executable,
                str(HERE / "matched_elliptic_completion.py"),
                "--plot-from-cache",
            ],
            environment,
        )
        run(
            [
                sys.executable,
                str(HERE / "tanh_slope_collision_snapshots.py"),
                "--plot-from-cache",
            ],
            environment,
        )
        run([sys.executable, str(HERE / "tanh_wb_trajectories.py")], environment)
        run(
            [
                sys.executable,
                str(HERE / "gaussian_rbf_trace_diagnostics.py"),
                "--input",
                str(HERE / "gaussian_rbf_instability_unpenalized_mixed_precision.json"),
                "--output-directory",
                str(HERE),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                str(HERE / "gaussian_instability_adaptive_weight_penalty.py"),
                "--plot-from-json",
                str(HERE / "gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.json"),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                str(HERE / "boundary_completion_experiment.py"),
                "--output-suffix",
                "lbfgs_plateau",
                "--plot-from-cache",
                str(HERE / "boundary_completion_third_derivative_atoms_lbfgs_plateau.json"),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                str(HERE / "boundary_completion_2d_experiment.py"),
                "--plot-from-cache",
                "--output-directory",
                str(HERE),
                "--output-suffix",
                "adam",
            ],
            environment,
        )
        run([sys.executable, str(HERE / "ingham_illustration.py")], environment)
        sync_preserved_figure_layouts()
        sync_numbered_figures()
    print("Regenerated the deterministic and cached-state figures.")


def run_matched_controls() -> None:
    REPRODUCTION_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="coercivity-mpl-") as cache_directory:
        environment = configured_environment(cache_directory)
        run(
            [
                sys.executable,
                str(HERE / "boundary_completion_matched_control_lbfgs.py"),
                "--input",
                str(HERE / "boundary_completion_third_derivative_atoms_lbfgs_plateau.json"),
                "--output",
                str(
                    REPRODUCTION_OUTPUT_DIRECTORY
                    / "boundary_completion_matched_control_lbfgs.json"
                ),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                str(HERE / "boundary_completion_2d_matched_control_lbfgs.py"),
                "--input",
                str(HERE / "boundary_completion_2d_atoms_adam.json"),
                "--output",
                str(
                    REPRODUCTION_OUTPUT_DIRECTORY
                    / "boundary_completion_2d_matched_control_lbfgs.json"
                ),
            ],
            environment,
        )
    print(f"Regenerated matched-polish controls in {REPRODUCTION_OUTPUT_DIRECTORY}.")


def run_full() -> None:
    """Run every optimizer trajectory used by the paper's numerical section."""
    with tempfile.TemporaryDirectory(prefix="coercivity-mpl-") as cache_directory:
        environment = configured_environment(cache_directory)
        for precision in ("float64", "float32"):
            run(
                [sys.executable, str(HERE / "tanh_parameter_escape_loss.py"), "--dtype", precision],
                environment,
            )
        run([sys.executable, str(HERE / "tanh_wb_trajectories.py")], environment)
        sync_from_code(("wb_trajectories.pdf", "wb_trajectories.png"))
        run([sys.executable, str(HERE / "tanh_slope_collision_snapshots.py")], environment)
        sync_from_code(
            (
                "tanh_x_tanhprime_energy_minimization.pdf",
                "tanh_x_tanhprime_energy_minimization.png",
            )
        )
        run(
            [
                sys.executable,
                str(HERE / "gaussian_rbf_instability_figure5_mixed_precision.py"),
                "--orders",
                "1",
                "--float32-iter",
                "100000000",
                "--float64-iter",
                "100000000",
                "--lr",
                "1e-6",
                "--q-min",
                "-16",
                "--q-max",
                "0.5",
                "--n-profile",
                "6000",
                "--n-quad",
                "20001",
                "--workers",
                "1",
            ],
            environment,
        )
        sync_from_code(
            (
                "gaussian_rbf_instability_unpenalized_mixed_precision.pdf",
                "gaussian_rbf_instability_unpenalized_mixed_precision.png",
            )
        )
        run(
            [
                sys.executable,
                str(HERE / "gaussian_rbf_trace_diagnostics.py"),
                "--input",
                str(HERE / "gaussian_rbf_instability_unpenalized_mixed_precision.json"),
                "--output-directory",
                str(HERE),
            ],
            environment,
        )
        sync_from_code(
            (
                "gaussian_rbf_instability_unpenalized_mixed_precision_diagnostics.pdf",
                "gaussian_rbf_instability_unpenalized_mixed_precision_diagnostics.png",
            )
        )
        run(
            [
                sys.executable,
                str(HERE / "gaussian_instability_adaptive_weight_penalty.py"),
                "--start-exp",
                "5",
                "--end-exp",
                "12",
                "--max-iter",
                "1000000",
                "--lr",
                "0.5",
                "--plateau-window",
                "50000",
                "--plateau-rtol",
                "1e-3",
            ],
            environment,
        )
        sync_from_code(
            (
                "gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.pdf",
                "gaussian_instability_weight_penalty_adaptive_1e-5_to_1e-12.png",
            )
        )
        run(
            [
                sys.executable,
                str(HERE / "boundary_completion_experiment.py"),
                "--final-stage-optimizer",
                "lbfgs",
                "--output-suffix",
                "lbfgs_plateau",
            ],
            environment,
        )
        sync_from_code(
            (
                "boundary_completion_third_derivative_atoms_lbfgs_plateau.pdf",
                "boundary_completion_third_derivative_atoms_lbfgs_plateau.png",
                "boundary_completion_third_derivative_representation_lbfgs_plateau.pdf",
                "boundary_completion_third_derivative_representation_lbfgs_plateau.png",
            )
        )
        run(
            [sys.executable, str(HERE / "boundary_completion_2d_experiment.py")],
            environment,
        )
        sync_from_code(
            (
                "boundary_completion_2d_atoms_adam.pdf",
                "boundary_completion_2d_atoms_adam.png",
                "boundary_completion_2d_representation_adam.pdf",
                "boundary_completion_2d_representation_adam.png",
            )
        )
        run([sys.executable, str(HERE / "matched_elliptic_completion.py")], environment)
        sync_from_code(("matched_elliptic_completion.pdf", "matched_elliptic_completion.png"))
        run([sys.executable, str(HERE / "ingham_illustration.py")], environment)
        sync_from_code(
            ("ingham_panel_T_2_5_10_an1_styled.pdf", "ingham_panel_T_2_5_10_an1_styled.png")
        )
    run_matched_controls()
    sync_preserved_figure_layouts()
    sync_numbered_figures()
    print("Completed the full reproduction workflow.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate sources and cached artifacts")
    mode.add_argument(
        "--fast",
        action="store_true",
        help="regenerate deterministic/cached-state figures, synchronize them, then check",
    )
    mode.add_argument(
        "--matched-controls",
        action="store_true",
        help="rerun cached-state 1D and 2D matched LBFGS controls, then check",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="rerun all experiments, including both 100-million-step Gaussian trajectories",
    )
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="warn instead of failing when installed package versions differ from the archive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fast:
        run_fast_plots()
    elif args.matched_controls:
        run_matched_controls()
    elif args.full:
        run_full()
    check_artifacts(
        allow_version_drift=args.allow_version_drift,
        verify_checksums=not args.full,
    )


if __name__ == "__main__":
    main()
