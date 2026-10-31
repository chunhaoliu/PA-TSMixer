"""LaTeX rows of Table V, the combined-link-condition table (three-seed means) from link_impairment.json."""
import argparse, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M = ["CV", "CA", "CNN-GRU", "FlightBERT++", "DLinear", "Transformer", "PatchTST", "TimeMixer", "TSMixer", "PA-TSMixer"]
COLS = [(d, c) for d in ("trajair", "tartanaviation") for c in ("nominal", "moderate", "severe")]


def value(d, m, c, metric):
    return np.mean([r[metric] for r in R if r["experiment"] == "E3" and r["dataset"] == d
                    and r["method"] == m and r["params"]["condition"] == c])


def main():
    global R
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results/link/link_impairment.json")
    args = p.parse_args()
    R = json.loads(args.results.read_text(encoding="utf-8"))
    for metric in ("ADE", "FDE"):
        raw = {m: [value(d, m, c, metric) for d, c in COLS] for m in M}
        print("% " + metric)
        for m in M:
            cells = []
            for j in range(len(COLS)):
                order = sorted(raw[k][j] for k in M)
                s = f"{raw[m][j]:.1f}"
                if raw[m][j] == order[0]:
                    s = r"\textbf{" + s + "}"
                elif raw[m][j] == order[1]:
                    s = r"\underline{" + s + "}"
                cells.append(s)
            name = "PA-TSMixer (ours)" if m == "PA-TSMixer" else m
            print(name + " & " + " & ".join(cells) + r"\\")


if __name__ == "__main__":
    main()
