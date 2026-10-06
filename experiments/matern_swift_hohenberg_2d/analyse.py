"""Replay final-state and shared-threshold work diagnostics for all ten starts."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from comparison_metrics import analyse, coefficient_norms, error_measure, print_summary, reference_summaries
from records import load_results
import experiment as E


def measures(order=96):
    return error_measure(E, order)


def weight_norm(state):
    return coefficient_norms(state)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE.parents[1]/"data/matern")
    parser.add_argument("--output", type=Path, help="Save final-state, trajectory and threshold diagnostics")
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1e-3, 1e-5, 1e-7])
    parser.add_argument("--final-only", action="store_true",
                        help="Evaluate returned states only, without history replay or threshold work")
    parser.add_argument("--reference-dir", type=Path,
                        help="Also write h1_errors.json, paper_table.json and shared_threshold_work.json here")
    args = parser.parse_args()
    if any(t <= 0 for t in args.thresholds):
        parser.error("thresholds must be positive")
    result = analyse(load_results(args.results), E, args.quadrature_order, args.thresholds,
                     histories=not args.final_only)
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    if args.reference_dir is not None and not args.final_only:
        for name, content in reference_summaries(result).items():
            (args.reference_dir/name).write_text(json.dumps(content, indent=2)+"\n")


if __name__ == "__main__":
    main()
