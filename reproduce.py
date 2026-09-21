#!/usr/bin/env python3
"""Verify the paper artifact, redraw its figures, or recompute its tables."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def verify() -> None:
    inventory = ROOT / "SHA256SUMS"
    if not inventory.is_file():
        raise RuntimeError("Missing SHA256SUMS inventory")
    count = 0
    for line in inventory.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError(f"Missing file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"File differs from the recorded release: {name}")
        count += 1
    figures = json.loads((ROOT / "figures/manifest.json").read_text())
    if [item["figure"] for item in figures] != list(range(1, 11)):
        raise RuntimeError("The figure map must contain exactly Figures 1–10")
    for item in figures:
        for key in ("file", "preview"):
            if not (ROOT / item[key]).is_file():
                raise RuntimeError(f"Missing Figure {item['figure']} {key}")
    for study, seeds in (("matern", range(10)), ("poisson", range(10, 20))):
        with gzip.open(ROOT / "data" / study / "results.json.gz", "rt") as stream:
            runs = json.load(stream)["runs"]
        expected = {(seed, method) for seed in seeds
                    for method in ("ordinary", "completion", "finite_restart")}
        actual = [(run["seed"], run["method"]) for run in runs]
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError(f"Unexpected matched runs in {study}")
    print(f"Verified {count} files, ten publication figures, and 60 matched runs.", flush=True)


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print("\nRunning:", script, *arguments, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def figures() -> None:
    output = ROOT / "outputs/figures"
    output.mkdir(parents=True, exist_ok=True)
    for number, name in ((1, "tanh_limit"), (2, "forced_singularity"),
                         (3, "completion_patterns")):
        run(f"illustrations/plot_{name}.py", "--output", str(output / f"figure_{number:02d}"))
    run("experiments/tanh/plot_bias.py", "--pdf-only", "--output", str(output / "figure_04"))
    shutil.copy2(ROOT / "figures/paper/figure_05.pdf", output / "figure_05.pdf")
    print("Figure 5: restored the exact historical publication PDF.", flush=True)
    run("experiments/precision/plot.py", "--output", str(output / "figure_06"))
    for directory, field_number, trajectory_number, stem in (
        ("matern_swift_hohenberg_2d", 7, 8, "matern_swift_hohenberg_2d"),
        ("poisson_neumann_2d", 9, 10, "poisson_neumann_2d"),
    ):
        run(f"experiments/{directory}/plot_fields.py", "--output", str(output))
        run(f"experiments/{directory}/plot_results.py", "--output", str(output))
        for number, source_stem in ((field_number, stem + "_fields"), (trajectory_number, stem)):
            for extension in ("pdf", "png", "svg"):
                source = output / f"{source_stem}.{extension}"
                if source.exists():
                    shutil.copy2(source, output / f"figure_{number:02d}.{extension}")
    missing = [number for number in range(1, 11)
               if not (output / f"figure_{number:02d}.pdf").is_file()]
    if missing:
        raise RuntimeError(f"Missing generated figure PDFs: {missing}")
    print(f"\nAll ten figure PDFs are available in {output}", flush=True)


def tables() -> None:
    output = ROOT / "outputs/tables"
    output.mkdir(parents=True, exist_ok=True)
    for directory, name in (("matern_swift_hohenberg_2d", "matern"),
                            ("poisson_neumann_2d", "poisson")):
        run(f"experiments/{directory}/analyse.py", "--output", str(output / f"{name}.json"))
    print(f"\nTable calculations are available in {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the supplied scientific artifact")
    parser.add_argument("--figures", "--fast", action="store_true", help="Redraw Figures 1–10 from supplied records")
    parser.add_argument("--tables", action="store_true", help="Recompute Tables 1–2 from stored final states")
    args = parser.parse_args()
    if not any((args.check, args.figures, args.tables)):
        parser.print_help()
        return
    verify()
    if args.figures:
        figures()
    if args.tables:
        tables()


if __name__ == "__main__":
    main()
