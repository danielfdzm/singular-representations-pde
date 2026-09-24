"""Relative H1 errors and kernel weight norms, per seed and method."""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path
import numpy as np
from records import load_results
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as E


def measures(order=96):
    points, weights = E.quadrature(order)
    exact = E.manufactured_jet_solution(points[:, 0], points[:, 1])
    exact_gradient = E.manufactured_jet_gradient(points[:, 0], points[:, 1])
    norm = float(np.sqrt(np.dot(weights, exact**2+np.sum(exact_gradient**2, axis=1))))

    def error(state):
        u, g = E.evaluate_state(state, points)
        du, dg = u-exact, g-exact_gradient
        return float(np.sqrt(np.dot(weights, du*du+np.sum(dg*dg, axis=1))))/norm
    return error


def weight_norm(state):
    """Norm of the weights the underlying kernel atoms must carry."""
    out, c = [], state["coefficients"]
    for i, pair in enumerate(state["pairs"]):
        a, b = c[2*i], c[2*i+1]
        if pair["completed"]:
            out.append(math.hypot(a, b))
        else:
            h = pair["half_gap"]
            out.extend((abs(a/2+b/(2*h)), abs(a/2-b/(2*h))))
    out.extend(abs(x) for x in c[2*E.NPAIRS:])
    return float(np.linalg.norm(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[2]/"data/poisson")
    parser.add_argument("--output", type=Path, help="Optionally save recomputed per-start errors")
    arguments = parser.parse_args()
    error = measures()
    payload = load_results(arguments.results)
    by = {(r["seed"], r["method"]): r for r in payload["runs"]}
    seeds = sorted({r["seed"] for r in payload["runs"]})
    table, ratios = {}, []
    print(f"{'seed':>4} {'ordinary':>11} {'restart':>11} {'completion':>11} {'ratio':>8} "
          f"{'ord min h':>10} {'ord wnorm':>10} {'cmp wnorm':>10} {'tr':>3}")
    for s in seeds:
        row = {}
        for m in ("ordinary", "finite_restart", "completion"):
            run = by[(s, m)]
            gaps = [p["half_gap"] for p in run["final_state"]["pairs"] if not p["completed"]]
            row[m] = {"error": error(run["final_state"]), "weight_norm": weight_norm(run["final_state"]),
                      "min_half_gap": min(gaps) if gaps else 0.0, "solves": run["coefficient_solves"],
                      "transfers": len(run["transfers"]), "status": run["status"]}
        ratio = row["ordinary"]["error"]/row["completion"]["error"]
        ratios.append(ratio); table[s] = row
        print(f"{s:>4} {row['ordinary']['error']:11.3e} {row['finite_restart']['error']:11.3e} "
              f"{row['completion']['error']:11.3e} {ratio:8.1f} {row['ordinary']['min_half_gap']:10.2e} "
              f"{row['ordinary']['weight_norm']:10.2e} {row['completion']['weight_norm']:10.2e} "
              f"{row['completion']['transfers']:>3}")
    print(f"\n  all starts : wins {sum(r>1 for r in ratios)}/{len(ratios)}  "
          f"median {statistics.median(ratios):.2f}  range {min(ratios):.2f}-{max(ratios):.2f}")
    restart = [by[(s, "finite_restart")] for s in seeds]
    rr = [table[s]["finite_restart"]["error"]/table[s]["completion"]["error"] for s in seeds]
    ws = [table[s]["ordinary"]["solves"]/table[s]["completion"]["solves"] for s in seeds]
    print(f"  restart/completion: wins {sum(r>1 for r in rr)}/{len(rr)}  median {statistics.median(rr):.2f}  "
          f"range {min(rr):.2f}-{max(rr):.2f}")
    print(f"  work ordinary/completion: median {statistics.median(ws):.2f}  range {min(ws):.2f}-{max(ws):.2f}")
    for m in ("ordinary", "finite_restart", "completion"):
        e = [table[s][m]["error"] for s in seeds]; w = [table[s][m]["weight_norm"] for s in seeds]
        k = [table[s][m]["solves"] for s in seeds]; g = [table[s][m]["min_half_gap"] for s in seeds]
        print(f"  {m:>15}: err {min(e):.2e}-{max(e):.2e}  wnorm {min(w):.2e}-{max(w):.2e}  "
              f"solves {min(k)}-{max(k)}  min h {min(g):.1e}-{max(g):.1e}")
    ranges = {m: {key: [min(table[s][m][key] for s in seeds), max(table[s][m][key] for s in seeds)]
                  for key in ("error", "weight_norm", "min_half_gap", "solves", "transfers")}
              for m in ("ordinary", "finite_restart", "completion")}
    # Publication labels/placeholders; preserve the full per-start diagnostics.
    for values in ranges.values():
        values["renormalization_steps"] = values.pop("transfers")
    ranges["completion"]["weight_norm"] = None
    ranges["completion"]["min_half_gap"] = None
    print("\nTable2 (all ten reported starts):")
    print(json.dumps(ranges, indent=2))
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps({"per_start": table, "paper_table": ranges, "paper_seeds": seeds}, indent=2)+"\n")


if __name__ == "__main__":
    main()
