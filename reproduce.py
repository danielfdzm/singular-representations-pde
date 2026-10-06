#!/usr/bin/env python3
"""Verify the paper artifact, redraw Figures 1–9, or replay its numerical results."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FIGURE_ASSETS = ("figure_01.pdf", "figure_02.pdf", "figure_03.pdf", "figure_04.pdf",
                 "figure_05.pdf", "figure_11.pdf", "figure_09.pdf", "figure_10.pdf", "figure_08.pdf")
FIGURE_SOURCES = ("illustrations/plot_tanh_limit.py", "illustrations/plot_forced_singularity.py",
                  "illustrations/plot_completion_patterns.py", "experiments/tanh/plot_bias.py",
                  "experiments/tanh/historical_slope_source.py", "experiments/manufactured_1d/plot.py",
                  "experiments/raw_coordinates/plot_main.py", "experiments/raw_coordinates/plot_main.py",
                  "experiments/raw_coordinates/plot_main.py")


def verify_recorded_runs() -> int:
    stable = ("ordinary", "restart", "squared_gap", "completion")
    primary = ("ordinary", "completion")
    datasets = (("raw_main/matern", range(10), primary),
                ("raw_main/matern/quadrature96", range(10), primary),
                ("raw_main/poisson", range(10, 20), primary),
                ("raw_main/poisson/quadrature64", range(10, 20), primary),
                ("matern", range(10), stable), ("matern/quadrature96", range(10), stable),
                ("poisson", range(10, 20), stable), ("poisson/quadrature64", range(10, 20), stable))
    total = 0
    for study, seeds, expected_methods in datasets:
        directory = ROOT / "data" / study
        with gzip.open(directory / "results.json.gz", "rb") as stream:
            serialized = stream.read()
        runs = json.loads(serialized)["runs"]
        expected = {(seed, method) for seed in seeds for method in expected_methods}
        actual = [(record["seed"], record["method"]) for record in runs]
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f"Unexpected runs in {study}")
        provenance = json.loads((directory / "provenance.json").read_text())
        hashes = provenance["retained_run_sha256"]
        if set(hashes) != {f"{seed}:{method}" for seed, method in expected}:
            raise RuntimeError(f"Unexpected provenance run keys in {study}")
        for record in runs:
            key = f"{record['seed']}:{record['method']}"
            digest = hashlib.sha256(json.dumps(record, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if digest != hashes[key]:
                raise RuntimeError(f"Recorded run changed: {study}/{key}")
        if study.startswith("raw_main/"):
            if hashlib.sha256((directory / "results.json.gz").read_bytes()).hexdigest() != provenance["results_sha256"]:
                raise RuntimeError(f"Primary results differ from provenance: {study}")
            sources = provenance["source_sha256"]
            if not isinstance(sources, dict) or not sources:
                raise RuntimeError(f"Missing generating source hashes in {study}")
            for name, expected_digest in sources.items():
                source = ROOT / name
                if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_digest:
                    raise RuntimeError(f"Generating source differs: {study}/{name}")
        total += len(runs)
    return total


def verify() -> None:
    inventory = ROOT / "SHA256SUMS"
    if not inventory.is_file():
        raise RuntimeError("Missing SHA256SUMS inventory")
    recorded = set()
    for line in inventory.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        if name in recorded or Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"Invalid or duplicate inventory path: {name}")
        recorded.add(name)
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError(f"Missing file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"File differs from the recorded release: {name}")
    # Generated outputs, environments and interpreter caches are local work,
    # while every distributed source, record and publication asset is inventoried.
    distributed = {p.relative_to(ROOT).as_posix() for directory in
                   ("data", "docs", "experiments", "figures", "illustrations")
                   for p in (ROOT / directory).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts
                   and p.name != ".DS_Store" and p.suffix not in (".pyc", ".pyo", ".log")}
    distributed.update(p.name for p in ROOT.iterdir()
                       if p.is_file() and p.name not in ("SHA256SUMS", ".DS_Store")
                       and p.suffix not in (".pyc", ".pyo", ".log"))
    if distributed != recorded:
        raise RuntimeError(f"Release inventory differs: unlisted={sorted(distributed-recorded)}, "
                           f"outside release directories={sorted(recorded-distributed)}")
    figures = json.loads((ROOT / "figures/manifest.json").read_text())
    if [item["figure"] for item in figures] != list(range(1, 10)):
        raise RuntimeError("The figure map must contain exactly main Figures 1–9")
    for number, item in enumerate(figures, start=1):
        expected_metadata = {"file": f"figures/paper/figure_{number:02d}.pdf",
                             "preview": f"figures/previews/figure_{number:02d}.png",
                             "manuscript_asset": FIGURE_ASSETS[number-1],
                             "source": FIGURE_SOURCES[number-1]}
        if any(item.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(f"Incorrect manuscript mapping for Figure {number}")
        for key in ("file", "preview"):
            if not (ROOT / item[key]).is_file():
                raise RuntimeError(f"Missing Figure {item['figure']} {key}")
        source = ROOT / FIGURE_SOURCES[number-1]
        if not source.is_file():
            raise RuntimeError(f"Missing Figure {number} source: {source.relative_to(ROOT)}")
        if hashlib.sha256((ROOT / item["file"]).read_bytes()).hexdigest() != item.get("sha256"):
            raise RuntimeError(f"Publication PDF differs from Figure {number} metadata")
    if {p.name for p in (ROOT / "figures/paper").glob("*")} != {
        f"figure_{number:02d}.pdf" for number in range(1, 10)
    }:
        raise RuntimeError("Publication figure directory must contain exactly Figures 1–9")
    total = verify_recorded_runs()
    summary = json.loads((ROOT / "data/manufactured_1d/summary.json").read_text())
    with (ROOT / "data/manufactured_1d/sweep.csv").open(newline="") as stream:
        sweep = list(csv.DictReader(stream))
    expected_cases = {(case, method) for case in ("0", "0.01")
                      for method in ("prescribed", "symmetric_optimized")}
    if len(sweep) != summary["number_of_records"] or {
        (row["epsilon"], row["method"]) for row in sweep
    } != expected_cases:
        raise RuntimeError("Unexpected manufactured one-dimensional case coverage")
    caps = [{row["R"] for row in sweep if (row["epsilon"], row["method"]) == case}
            for case in sorted(expected_cases)]
    if any(values != caps[0] for values in caps[1:]) or len(sweep) != 4 * len(caps[0]):
        raise RuntimeError("Unmatched coefficient caps in the one-dimensional study")
    with (ROOT / "data/manufactured_1d/global_search.csv").open(newline="") as stream:
        search = list(csv.DictReader(stream))
    if len(search) != 3 * len(caps[0]) or {row["epsilon"] for row in search} != {"0", "0.001", "0.01"}:
        raise RuntimeError("Unexpected case coverage in the one-dimensional global search")
    precision_summary()
    print(f"Verified {len(recorded)} files, nine publication figures, all {total} recorded two-dimensional runs, "
          "the one-dimensional records and the appendix precision sweep.", flush=True)


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print("\nRunning:", script, *arguments, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def figures(reuse_analysis: bool = False) -> None:
    output = ROOT / "outputs/figures"
    output.mkdir(parents=True, exist_ok=True)
    for number, name in ((1, "tanh_limit"), (2, "forced_singularity"),
                         (3, "completion_patterns")):
        run(f"illustrations/plot_{name}.py", "--output", str(output / f"figure_{number:02d}"))
    run("experiments/tanh/plot_bias.py", "--pdf-only", "--output", str(output / "figure_04"))
    shutil.copy2(ROOT / "figures/paper/figure_05.pdf", output / "figure_05.pdf")
    print("Figure 5: restored the exact historical publication PDF.", flush=True)
    run("experiments/manufactured_1d/plot.py", "--input", str(ROOT / "data/manufactured_1d"),
        "--output", str(output / "figure_06"))
    arguments = ["--output", str(output)]
    if reuse_analysis:
        arguments += ["--analysis-directory", str(ROOT / "outputs/tables")]
    run("experiments/raw_coordinates/plot_main.py", *arguments)
    missing = [number for number in range(1, 10)
               if not (output / f"figure_{number:02d}.pdf").is_file()]
    if missing:
        raise RuntimeError(f"Missing generated figure PDFs: {missing}")
    print(f"\nAll nine paper figure PDFs are available in {output}", flush=True)


def precision_summary() -> list[dict]:
    """Replay the appendix evaluation-accuracy summaries from recorded samples."""
    supplied = json.loads((ROOT / "data/precision/summary.json").read_text())
    with (ROOT / "data/precision/sweep.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    expected = {(precision, order, k, method)
                for precision in ("float32", "float64") for order in (1, 3)
                for k in range(supplied["protocol"]["max_k"][f"{precision}_order{order}"]+1)
                for method in ("raw", "stable", "completed")}
    actual = [(r["precision"], int(r["order"]), int(r["k"]), r["method"]) for r in samples]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("Unexpected precision-sweep case coverage")
    rows = []
    for precision in ("float32", "float64"):
        eps = 2.0**(-23 if precision == "float32" else -52)
        for order in (1, 3):
            case = [r for r in samples if r["precision"] == precision and int(r["order"]) == order]
            raw = [r for r in case if r["method"] == "raw"]
            stable = [r for r in case if r["method"] == "stable"]
            completed = next(r for r in case if r["method"] == "completed")
            best = min(raw, key=lambda r: float(r["total_h1"]))
            small = [r for r in stable if float(r["model_h1"]) < eps]
            row = {"precision": precision, "order": order,
                   "raw_best_sampled_gap": float(best["gap"]),
                   "raw_minimum_sampled_h1": float(best["total_h1"]),
                   "stable_minimum_sampled_h1": min(float(r["total_h1"]) for r in stable),
                   "stable_small_gap_evaluation_max": max(float(r["evaluation_h1"]) for r in small),
                   "stable_small_gap_count": len(small), "completed_h1": float(completed["total_h1"]),
                   "nominal_eps": eps}
            recorded = next(r for r in supplied["summary"]
                            if r["precision"] == precision and r["order"] == order)
            for key, value in row.items():
                if isinstance(value, float):
                    if not math.isclose(value, recorded[key], rel_tol=1e-12, abs_tol=0.0):
                        raise RuntimeError(f"Precision summary differs: {precision}/order{order}/{key}")
                elif value != recorded[key]:
                    raise RuntimeError(f"Precision summary differs: {precision}/order{order}/{key}")
            row["stable_minimum_in_machine_eps"] = row["stable_minimum_sampled_h1"] / eps
            rows.append(row)
    return rows


def tables() -> None:
    output = ROOT / "outputs/tables"
    output.mkdir(parents=True, exist_ok=True)
    for name, higher in (("matern", "quadrature96"), ("poisson", "quadrature64")):
        directory = ROOT / "data/raw_main" / name
        run("experiments/raw_coordinates/analyse_main.py", "--study", name,
            "--results", str(directory), "--output", str(output / f"raw_{name}.json"))
        for data, label in ((directory, f"{name}_eval128"),
                            (directory / higher, f"{name}_{higher}")):
            run("experiments/raw_coordinates/analyse_main.py", "--study", name,
                "--results", str(data), "--quadrature-order", "128", "--final-only",
                "--output", str(output / f"raw_{label}.json"))
    run("experiments/raw_coordinates/write_tables.py", "--analysis-directory", str(output),
        "--output", str(output / "raw_tex"))
    # The appendix reports four stable-coordinate methods at the main orders.
    for directory, name in (("matern_swift_hohenberg_2d", "matern"),
                            ("poisson_neumann_2d", "poisson")):
        run(f"experiments/{directory}/analyse.py", "--final-only", "--output", str(output / f"{name}.json"))
    # Higher-order stable controls retain the evaluation orders of their records.
    for directory, name, higher, order in (("matern_swift_hohenberg_2d", "matern", "quadrature96", 96),
                                          ("poisson_neumann_2d", "poisson", "quadrature64", 128)):
        run(f"experiments/{directory}/analyse.py", "--results", str(ROOT / f"data/{name}/{higher}"),
            "--quadrature-order", str(order), "--final-only", "--output", str(output / f"{name}_{higher}.json"))
    run("experiments/training_quadrature.py", "--output", str(output / "training_quadrature.json"))
    precision = precision_summary()
    (output / "precision.json").write_text(json.dumps(precision, indent=2)+"\n")
    with (output / "precision.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(precision[0]))
        writer.writeheader()
        writer.writerows(precision)
    print(f"\nPaper tables and appendix coordinate, quadrature and precision summaries are available in {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the supplied scientific artifact")
    parser.add_argument("--figures", "--fast", action="store_true", help="Redraw only the nine figures printed in the paper")
    parser.add_argument("--tables", action="store_true", help="Replay paper tables and appendix coordinate, quadrature and precision summaries")
    args = parser.parse_args()
    if not any((args.check, args.figures, args.tables)):
        parser.print_help()
        return
    verify()
    if args.tables:
        tables()
    if args.figures:
        figures(reuse_analysis=args.tables)


if __name__ == "__main__":
    main()
