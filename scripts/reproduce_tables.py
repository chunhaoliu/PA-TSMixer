"""Recompute Tables II, III and V of the paper from the per-run results and compare every printed cell.

    python scripts/reproduce_tables.py          # exit status 0 when all cells agree

Means and sample standard deviations are taken over the three seeds (CV and CA are deterministic). As in the
paper, values are first rounded to two decimals (the precision of the supplement) and then printed with one
decimal; MSE is printed in 10^3 m^2. Rounding is half-up.
"""
import json, statistics, sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = json.loads((ROOT / "results/runs.json").read_text(encoding="utf-8"))
PAPER = json.loads((ROOT / "results/paper_tables.json").read_text(encoding="utf-8"))
METHODS = ["CV", "CA", "DLinear", "Transformer", "PatchTST", "TimeMixer", "TSMixer", "PA-TSMixer"]
COLS = [(d, h) for d in ("trajair", "tartanaviation") for h in (10, 20, 30, 60)]


def q(x, places):
    return Decimal(repr(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def summary(values):
    """(mean, sd) rounded to two decimals; sd is None for a single deterministic run."""
    mean = q(statistics.fmean(values), "0.01")
    return mean, (q(statistics.stdev(values), "0.01") if len(values) > 1 else None)


def cell(mean, sd, places="0.1"):
    m = mean.quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{m}" if sd is None else f"{m}±{sd.quantize(Decimal(places), rounding=ROUND_HALF_UP)}"


def improvement(ref, new):
    return str(((ref - new) / ref * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def values(study, d, h, m, key):
    return [r[key] for r in RUNS if r["study"] == study and r["dataset"] == d and r["horizon"] == h and r["method"] == m]


def table_main():
    out = {}
    for key, block in (("ADE", "ADE (m)"), ("FDE", "FDE (m)"), ("MSE", "Physical MSE ($10^3$ m$^2$)")):
        stats = {(c, m): summary(values("main", *c, m, key)) for c in COLS for m in METHODS}
        best = {c: min(stats[(c, m)][0] for m in METHODS) for c in COLS}
        rows = {}
        for m in METHODS:
            if key == "MSE":
                cells = [cell(stats[(c, m)][0] / 1000, None if stats[(c, m)][1] is None else
                              (stats[(c, m)][1] / 1000).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "0.01")
                         for c in COLS]
            else:
                cells = [cell(*stats[(c, m)]) for c in COLS]
            rows["PA-TSMixer (ours)" if m == "PA-TSMixer" else m] = cells + [str(sum(stats[(c, m)][0] == best[c] for c in COLS))]
        rows["Improvement (%)"] = [improvement(stats[(c, "TSMixer")][0], stats[(c, "PA-TSMixer")][0]) for c in COLS] + [""]
        out[block] = rows
    return out


def table_decomp():
    out = {}
    for key, block in (("horizontal_ADE", "Horizontal ADE (m)"), ("altitude_MAE", "Altitude MAE (m)")):
        out[block] = {("PA-TSMixer (ours)" if m == "PA-TSMixer" else m):
                      [str(q(statistics.fmean(values("main", *c, m, key)), "0.1")) for c in COLS] for m in METHODS}
    return out


def table_ablation():
    labels = ["PA-TSMixer (ref.)", "TSMixer, matched size", "PA, equal weights", "PA, uniform widths"]
    cols = [(d, h) for d in ("trajair", "tartanaviation") for h in (30, 60)]
    out = {}
    for key in ("ADE", "FDE"):
        stats = {(c, m): summary(values("ablation", *c, m, key)) for c in cols for m in labels}
        rows = {}
        for m in labels:
            params = next(r["parameters"] for r in RUNS if r["study"] == "ablation" and r["method"] == m)
            rows[m] = [f"{params:,}"] + [cell(*stats[(c, m)]) for c in cols]
        rows["Improvement (%)"] = [""] + [improvement(stats[(c, labels[1])][0], stats[(c, labels[0])][0]) for c in cols]
        out[f"{key} (m)"] = rows
    return out


def main():
    mismatches = total = 0
    for name, fn in (("main", table_main), ("decomp", table_decomp), ("ablation", table_ablation)):
        mine, printed = fn(), PAPER[name]
        for block, rows in printed.items():
            for label, cells in rows.items():
                for i, expected in enumerate(cells):
                    total += 1
                    got = mine[block][label][i]
                    if got != expected:
                        mismatches += 1
                        print(f"{name} | {block} | {label} | column {i + 1}: recomputed {got!r}, paper {expected!r}")
    print(f"{total - mismatches} of {total} printed cells reproduced")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
