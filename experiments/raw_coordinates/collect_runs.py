#!/usr/bin/env python3
"""Collect two matched ordinary-coordinate methods into a replayable dataset."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from analyse_main import CODE, METHODS, fingerprint, read_results


def collect(study, ordinary, completion, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a dataset: {output}")
    sources = []
    runs = []
    source_hashes = {}
    validations = {}
    for method, directory in zip(METHODS, (ordinary, completion)):
        payload = read_results(directory)
        selected = [r for r in payload["runs"] if r["method"] in (method, "raw_"+method)]
        if len(selected) != 10:
            raise ValueError(f"Expected ten {method} records in {directory}")
        for r in selected:
            copied = dict(r)
            if copied["method"] != method:
                copied["source_method"] = copied["method"]
                copied["method"] = method
            runs.append(copied)
        path = directory / "results.json"
        if not path.exists():
            path = directory / "results.json.gz"
        sources.append({"method": method, "input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "configuration": payload.get("configuration"), "software": payload.get("software"),
            "original_run_sha256": {str(r["seed"]): fingerprint(r) for r in selected}})
        hashes = payload.get("source_sha256", payload.get("source_files_sha256", {}))
        for name, digest in hashes.items():
            current = CODE / name
            if not current.exists() or hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source differs from its generating version: {name}")
            if name in source_hashes and source_hashes[name] != digest:
                raise ValueError(f"Conflicting generating sources: {name}")
            source_hashes[name] = digest
        validation = directory / "validation.json"
        if validation.exists():
            validations[method] = json.loads(validation.read_text())
        elif "validation" in payload:
            validations[method] = payload["validation"]
    seeds = list(range(10)) if study == "matern" else list(range(10, 20))
    if {(r["seed"], r["method"]) for r in runs} != {(s, m) for s in seeds for m in METHODS}:
        raise ValueError("Incomplete matched-start coverage")
    by = {(r["seed"], r["method"]): r["initial_state"] for r in runs}
    for seed in seeds:
        left, right = (by[seed, method] for method in METHODS)
        for key in ("background_center", "singletons"):
            if key in left and left[key] != right[key]:
                raise ValueError(f"Unmatched initial geometry: seed {seed}, {key}")
        if len(left["pairs"]) != len(right["pairs"]):
            raise ValueError("Unmatched initial pair counts")
        for first, second in zip(left["pairs"], right["pairs"]):
            if any(first[key] != second[key] for key in ("center", "angle", "completed")):
                raise ValueError(f"Unmatched initial pair geometry: seed {seed}")
            if not math.isclose(first["half_gap"], second["half_gap"], rel_tol=2e-15, abs_tol=0):
                raise ValueError(f"Unmatched initial half-gap: seed {seed}")
    orders = {r["training_quadrature_order"] for r in runs}
    if len(orders) != 1:
        raise ValueError("Mismatched training quadrature")
    output.mkdir(parents=True)
    payload = {"schema_version": 1, "study": study, "configuration": {"methods": list(METHODS), "seeds": seeds,
        "training_quadrature_order": orders.pop(), "evaluation": "direct finite translates/direct completed derivatives",
        "coefficient_norm": "actual coefficient vector in the current representation"},
        "source_sha256": source_hashes, "runs": sorted(runs, key=lambda r: (r["seed"], r["method"]))}
    contents = (json.dumps(payload, separators=(",", ":"), allow_nan=False)+"\n").encode()
    (output / "results.json.gz").write_bytes(gzip.compress(contents, mtime=0))
    provenance = {"schema_version": 1, "description": "Two-method primary comparison, using ordinary weights until derivative transfer",
        "generating_inputs": sources, "source_sha256": source_hashes,
        "retained_run_sha256": {f"{r['seed']}:{r['method']}": fingerprint(r) for r in payload["runs"]},
        "results_sha256": hashlib.sha256((output / "results.json.gz").read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (output / "validation.json").write_text(json.dumps(validations, indent=2, allow_nan=False)+"\n")
    print(f"Collected all20 {study} runs into {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("matern", "poisson"), required=True)
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.study, args.ordinary, args.completion, args.output)


if __name__ == "__main__":
    main()
