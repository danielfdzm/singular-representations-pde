"""Run the exact historical slope-collision source in a separate output directory.

The preserved publication PDF is not overwritten. A fresh run also records
full-precision snapshots, which were not saved for the historical figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("historical_slope_source.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "tanh_slope")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    plt.show = lambda: None

    previous = Path.cwd()
    try:
        os.chdir(output)
        state = runpy.run_path(str(SOURCE), run_name="__main__")
    finally:
        os.chdir(previous)

    stages = {
        "Initial": "initial",
        "1/5 Adam": "Adam step 1666",
        "Final Adam": "Adam step 5000",
        "1/5 LBFGS": "independent LBFGS, max_iter=200",
        "Final LBFGS": "independent LBFGS, max_iter=1000",
    }
    snapshots = []
    for snapshot in state["snapshots"]:
        label = snapshot["label"]
        snapshots.append({
            "historical_label": label,
            "actual_stage": stages[label],
            **{key: float(value) for key, value in snapshot.items() if key != "label"},
        })
    record = {
        "note": "Fresh rerun; these are not the historical publication checkpoints.",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "torch": torch.__version__, "matplotlib": matplotlib.__version__},
        "snapshots": snapshots,
    }
    (output / "snapshots.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Full-precision rerun snapshots: {output / 'snapshots.json'}")


if __name__ == "__main__":
    main()
