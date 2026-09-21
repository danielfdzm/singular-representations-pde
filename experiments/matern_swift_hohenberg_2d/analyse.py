"""Relative H1 errors against the manufactured target, per seed and method."""
from __future__ import annotations
import argparse, importlib.util, json, statistics, sys, math
from pathlib import Path
import numpy as np
from records import load_results


def load(module_directory: Path):
    sys.path.insert(0, str(module_directory))
    for stale in ("experiment", "matern"):
        sys.modules.pop(stale, None)
    spec = importlib.util.spec_from_file_location("experiment", module_directory/"experiment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment"] = module
    spec.loader.exec_module(module)
    return module


def errors(module, order=96):
    points, weights = module.quadrature(order)
    exact = module.manufactured_jet_solution(points[:, 0], points[:, 1])
    exact_gradient = module.manufactured_jet_gradient(points[:, 0], points[:, 1])
    norm = float(np.sqrt(np.dot(weights, exact**2+np.sum(exact_gradient**2, axis=1))))

    def measure(state):
        u, gradient = module.evaluate_state(state, points)
        difference = u-exact
        return float(np.sqrt(np.dot(weights, difference**2
                                    + np.sum((gradient-exact_gradient)**2, axis=1))))/norm
    return measure


def weight_norm(state):
    weights, c = [], state["coefficients"]
    for i, pair in enumerate(state["pairs"]):
        a, b = c[2*i:2*i+2]
        if pair["completed"]:
            weights.extend((a, b))
        else:
            h = pair["half_gap"]
            weights.extend((a/2+b/(2*h), a/2-b/(2*h)))
    weights.append(c[4])
    return float(np.linalg.norm(weights))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[2]/"data/matern")
    parser.add_argument("--output", type=Path, help="Optionally save recomputed per-start errors")
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()
    module = load(arguments.experiment)
    measure = errors(module)
    payload = load_results(arguments.results)
    table = {}
    for run in payload["runs"]:
        table.setdefault(run["seed"], {})[run["method"]] = {
            "error": measure(run["final_state"]), "status": run["status"],
            "solves": run["coefficient_solves"], "transfers": len(run["transfers"]),
            "energy": run["independent_quadrature"][-1]["energy"],
            "weight_norm": weight_norm(run["final_state"]),
            "min_half_gap": min((p["half_gap"] for p in run["final_state"]["pairs"] if not p["completed"]), default=0.0)}
    ratios, wins = [], 0
    print(f"\n{arguments.label}")
    print(f"{'seed':>4}  {'ordinary':>11}  {'completion':>11}  {'restart':>11}  "
          f"{'ord/comp':>9}  {'transfers':>9}")
    for seed in sorted(table):
        row = table[seed]
        o, c = row["ordinary"]["error"], row["completion"]["error"]
        r = row.get("finite_restart", {}).get("error", float("nan"))
        ratio = o/c
        ratios.append(ratio); wins += ratio > 1
        print(f"{seed:>4}  {o:11.4e}  {c:11.4e}  {r:11.4e}  {ratio:9.2f}  "
              f"{row['completion']['transfers']:>9}")
    print(f"\n  completion better on {wins}/{len(ratios)} seeds")
    print(f"  ratio ordinary/completion: min {min(ratios):.2f}  median "
          f"{statistics.median(ratios):.2f}  max {max(ratios):.2f}")
    best = min(table[s]["completion"]["error"] for s in table)
    print(f"  best completion error {best:.3e}; "
          f"best ordinary error {min(table[s]['ordinary']['error'] for s in table):.3e}")
    paper_seeds = [s for s in sorted(table) if table[s]["completion"]["transfers"] > 0]
    ranges = {m: {key: [min(table[s][m][key] for s in paper_seeds), max(table[s][m][key] for s in paper_seeds)]
                  for key in ("error", "weight_norm", "min_half_gap", "solves", "transfers")}
              for m in ("ordinary", "finite_restart", "completion")}
    print("\nTable1 (nine transferring starts):")
    print(json.dumps(ranges, indent=2))
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps({"per_start": table, "paper_table": ranges, "paper_seeds": paper_seeds}, indent=2)+"\n")


if __name__ == "__main__":
    main()
