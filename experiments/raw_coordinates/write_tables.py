#!/usr/bin/env python3
"""Generate the two-method manuscript tables from independent replay results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyse_main import CODE, METHODS


def sci(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.1e}".split("e")
    return mantissa+rf"\times10^{{{int(exponent)}}}"


def span(values):
    return sci(values[0]) if values[0] == values[1] else sci(values[0])+r"\text{--}"+sci(values[1])


def quantity(values, fixed=None):
    if fixed is None:
        return r"\("+span(values)+r"\)"
    return r"\("+f"{values[0]:.{fixed}f}"+r"\text{--}"+f"{values[1]:.{fixed}f}"+r"\)"


def final_table(study, analysis):
    summary = analysis["summary"]
    label = "Mat\\'ern" if study == "matern" else "Poisson--Neumann"
    placement = "!ht" if study == "matern" else "H"
    lines = [rf"\begin{{table}}[{placement}]\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Final accuracy, coefficient sizes, and work for the {label}",
        r"problem over ten shared starts.}",
        rf"\label{{tab:{study}_raw_results}}",
        r"\begin{tabular}{@{}lcc@{}}\toprule",
        r" & Standard optimization & Completion\\\midrule"]
    rows = [(r"Relative \(H^1\) error", lambda m: quantity(summary[m]["error_range"])),
        (r"\quad median", lambda m: r"\("+sci(summary[m]["median_h1_error"])+r"\)")]
    if study == "matern":
        rows.append((r"Relative \(e_Q\)", lambda m: quantity(summary[m]["quadratic_error_range"])))
    rows.extend((("Coefficient norm", lambda m: quantity(summary[m]["coefficient_norm_range"], 5 if study == "poisson" and m == "completion" else None)),
        ("Smallest center half-gap", lambda m: quantity(summary[m]["half_gap_range"])),
        (("Fits" if study == "matern" else "Solves")+" (median)", lambda m: f"{summary[m]['solves_range'][0]}--{summary[m]['solves_range'][1]} ({summary[m]['median_solves']:g})"),
        ("Completed pairs", lambda m: str(summary[m]["completed_pairs_range"][0]) if summary[m]["completed_pairs_range"][0] == summary[m]["completed_pairs_range"][1] else f"{summary[m]['completed_pairs_range'][0]}--{summary[m]['completed_pairs_range'][1]}"),
        ("Normal optimizer returns", lambda m: f"{summary[m]['status_counts'].get('optimizer_converged', 0)}/10")))
    for title, getter in rows:
        lines.append(title+" & "+" & ".join(getter(m) for m in METHODS)+r"\\")
    lines.extend((r"\bottomrule\end{tabular}", r"\end{table}"))
    return "\n".join(lines)+"\n"


def work_table(analyses):
    lines = [r"\begin{table}[!ht]\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Work needed to reach each relative \(H^1\) error threshold.",
        r"Counts are out of ten starts; medians include only starts that reach the threshold.}",
        r"\label{tab:raw_common_accuracy_work}",
        r"\begin{tabular}{@{}llrrrr@{}}\toprule",
        r" & & \multicolumn{2}{c}{Standard optimization} & \multicolumn{2}{c}{Completion}\\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"Problem & \(\tau\) & Reached & Median & Reached & Median\\\midrule"]
    for i, study in enumerate(("poisson", "matern")):
        if i:
            lines.append(r"\midrule")
        for j, row in enumerate(analyses[study]["threshold_work"]):
            title = ("Mat\\'ern" if study == "matern" else "Poisson") if j == 0 else ""
            values = [title, r"\("+sci(row["threshold"])+r"\)"]
            for method in METHODS:
                item = row["methods"][method]
                values.extend((f"{item['reached']}/10", "---" if item["median_solves"] is None else f"{item['median_solves']:g}"))
            lines.append(" & ".join(values)+r"\\")
    lines.extend((r"\bottomrule\end{tabular}", r"\end{table}"))
    return "\n".join(lines)+"\n"


def sensitivity_table(analyses):
    lines = [r"\begin{table}[H]\centering\small",
        r"\caption{Completion retains lower median errors with finer training quadrature.}",
        r"\label{tab:raw_training_quadrature}",
        r"\begin{tabular}{@{}llcccc@{}}\toprule",
        r" & & \multicolumn{2}{c}{Standard optimization} & \multicolumn{2}{c}{Completion}\\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"Problem & Order & Median \(H^1\) & \(\le10^{-7}\) & Median \(H^1\) & \(\le10^{-7}\)\\\midrule"]
    for i, study in enumerate(("poisson", "matern")):
        if i:
            lines.append(r"\midrule")
        for j, (order, suffix) in enumerate(((64, "eval128"), (96, "quadrature96")) if study == "matern" else ((32, "eval128"), (64, "quadrature64"))):
            a = analyses[f"{study}_{suffix}"]
            values = [("Mat\\'ern" if study == "matern" else "Poisson") if j == 0 else "", str(order)]
            for method in METHODS:
                values.extend((r"\("+sci(a["summary"][method]["median_h1_error"])+r"\)",
                    f"{sum(row['h1_error'] <= 1e-7 for row in a['final'][method])}/10"))
            lines.append(" & ".join(values)+r"\\")
    lines.extend((r"\bottomrule\end{tabular}", r"\end{table}"))
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-directory", type=Path, default=CODE / "outputs/tables")
    parser.add_argument("--output", type=Path, default=CODE / "outputs/tables/raw_tex")
    args = parser.parse_args()
    names = ("matern", "poisson", "matern_eval128", "matern_quadrature96", "poisson_eval128", "poisson_quadrature64")
    analyses = {name: json.loads((args.analysis_directory / ("raw_"+name+".json")).read_text()) for name in names}
    args.output.mkdir(parents=True, exist_ok=True)
    for study in ("poisson", "matern"):
        (args.output / (study+".tex")).write_text(final_table(study, analyses[study]))
    (args.output / "work.tex").write_text(work_table(analyses))
    (args.output / "sensitivity.tex").write_text(sensitivity_table(analyses))
    print("Generated four tables from independent numerical replay.")


if __name__ == "__main__":
    main()
