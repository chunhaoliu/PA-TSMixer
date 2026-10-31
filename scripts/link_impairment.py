"""Evaluate frozen models under ADS-B loss, report staleness and combined conditions.

Fig. 6 and Table V, Section VI-D. Evaluates PA-TSMixer, TSMixer, CV and CA.
The full ten-method results are in results/link/link_impairment.json. No retraining.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from link_common import (ROOT, RESULTS, METHODS, RUNS, SEEDS, arguments, compare_records,
                         learned_records, load_model, neural, reference)
from pa_tsmixer.data import Windows
from kinematic_baselines import predict as kinematic

DATASETS = ("trajair", "tartanaviation")
LOSS_RATES = (0, .1, .2, .3, .4, .5)
BURST_LENGTHS = (1, 5, 10)
AGES = (0, 1, 2, 3, 4, 5)
CONDITIONS = (
    dict(condition="nominal", sigma=0, pi=0, L=5, k=0),
    dict(condition="moderate", sigma=10, pi=.2, L=5, k=1),
    dict(condition="severe", sigma=20, pi=.4, L=5, k=3),
)


def gilbert_elliott(hist, pi, L, rng):
    """Stationary GE emissions at 0..58; report 59 always received.

    Loss fraction uses the 59 eligible reports. Bursts are pooled runs of lost
    reports within each window, including runs truncated at either boundary.
    A zero-loss condition has mean burst length 0 by convention.
    """
    assert 0 <= pi <= .5 and L >= 1
    p_bg = 1. / L
    p_gb = pi / (L * (1. - pi))
    state = rng.random(len(hist)) < pi
    drop = np.empty((len(hist), 59), dtype=bool)
    for t in range(59):
        drop[:, t] = state
        if t < 58:
            u = rng.random(len(hist))
            state = np.where(state, u >= p_bg, u < p_gb)
    lost = int(drop.sum())
    starts = drop & ~np.concatenate((np.zeros((len(hist), 1), dtype=bool), drop[:, :-1]), axis=1)
    runs = int(starts.sum())
    check = dict(pi=pi, L=L, realized_loss=float(drop.mean()),
                 realized_mean_burst=float(lost / runs) if runs else 0.)
    assert abs(check["realized_loss"] - pi) <= .02, check
    if not lost:
        return hist, check
    out = hist.copy()
    t = np.arange(60)
    for i in np.flatnonzero(drop.any(axis=1)):
        keep = np.r_[~drop[i], True]
        for a in range(3):
            out[i, :, a] = np.interp(t, t[keep], hist[i, keep, a])
    assert np.array_equal(out[:, 59], hist[:, 59])
    return out, check


def scores(pred, truth):
    assert pred.shape == truth.shape
    d = np.linalg.norm(pred - truth, axis=-1)
    assert np.isfinite(d).all()
    return dict(ADE=float(d.mean()), FDE=float(d[:, -1].mean()))


def assert_reference(value, reference, context, audit):
    relative_error = abs(value - reference) / reference
    assert relative_error < 1e-5, (context, value, reference, relative_error)
    audit.append(relative_error)


def write_summary(rows, checks, elapsed, audit, counts, started, output):
    """Generate requested Markdown tables from the complete per-run records."""
    lines = ["# ADS-B link-impairment evaluation", "",
             f"Started (UTC): {started}. Evaluation runtime: {elapsed:.2f} s ({elapsed / 60:.2f} min).", "",
             "Frozen checkpoints; test split; FP32; TF32 disabled. "
             "Recomputed released-model rows. Targets remain clean. "
             "A fresh np.random.default_rng(2026) is used per dataset, experiment and level; "
             "each resulting input array is shared across methods and seeds.", "",
             "Learned-model cells are mean ± sample standard deviation across seeds 42/123/456 "
             "(ddof=1), in metres. CV/CA are deterministic single evaluations, marked (det.); "
             "their sample standard deviation is not defined. Bold marks the smallest mean per column.", "",
             f"Sanity assertions: {len(audit)} reference comparisons passed at relative tolerance 1e-5; "
             f"largest relative error {max(audit):.3e}. No assertions relaxed. "
             "H=30 clean/pi=0 ADE and every learned H=60 full-future ADE were checked. "
             "All realized eligible-report loss fractions passed absolute tolerance 0.02.", "",
             "Test windows: " + "; ".join(f"{d}: {counts[d]:,}" for d in DATASETS) + ".", "",
             "E1 uses H=30 checkpoints. E2 and E3 use H=60 checkpoints and score the "
             "unchanged future slice [k:k+30], i.e. leads k+1 through k+30 after the last report. "
             "Nominal E3 uses L=5 as an inert convention when pi=0.", ""]

    def table(dataset, columns):
        # columns: (heading, experiment, parameter subset, metric)
        stats = {}
        for method in METHODS:
            for j, (_, experiment, params, metric) in enumerate(columns):
                selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method
                            and r["experiment"] == experiment
                            and all(r["params"][k] == v for k, v in params.items())]
                assert len(selected) == (1 if method in ("CV", "CA") else 3)
                values = [r[metric] for r in selected]
                stats[method, j] = (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None)
        best = [min(stats[m, j][0] for m in METHODS) for j in range(len(columns))]
        lines.append("| Method | " + " | ".join(c[0] for c in columns) + " |")
        lines.append("|:--|" + "--:|" * len(columns))
        for method in METHODS:
            cells = []
            for j in range(len(columns)):
                mean, sd = stats[method, j]
                cell = f"{mean:.2f} ± {sd:.2f}" if sd is not None else f"{mean:.2f} (det.)"
                cells.append(f"**{cell}**" if mean == best[j] else cell)
            lines.append("| " + method + " | " + " | ".join(cells) + " |")
        lines.append("")

    for dataset in DATASETS:
        lines.extend([f"## {dataset}", ""])
        for L in BURST_LENGTHS:
            lines.extend([f"### E1 — mean ADE (m), L={L} s", ""])
            table(dataset, [(f"pi={pi:.0%}", "E1", dict(pi=pi, L=L), "ADE") for pi in LOSS_RATES])
        lines.extend(["### E2 — mean ADE (m)", ""])
        table(dataset, [(f"k={k} s", "E2", dict(k=k), "ADE") for k in AGES])
        lines.extend(["### E3 — combined conditions (m)", "",
                      "Nominal: sigma=0 m, pi=0, k=0 s. Moderate: sigma=10 m, pi=0.2, L=5 s, k=1 s. "
                      "Severe: sigma=20 m, pi=0.4, L=5 s, k=3 s.", ""])
        table(dataset, [(f"{c['condition']} {metric}", "E3", dict(condition=c["condition"]), metric)
                        for c in CONDITIONS for metric in ("ADE", "FDE")])
    lines.extend(["## Generator checks", "",
                  "Loss fractions use indices 0..58 (59 eligible reports); index 59 is always received. "
                  "Mean bursts pool all within-window lost runs, including truncated boundary runs. "
                  "Window truncation can reduce their mean below L, especially at L=10 s. "
                  "A mean burst of 0 at pi=0 denotes no lost runs.", "",
                  "| Dataset | pi | L (s) | Realized loss | Realized mean burst (s) |",
                  "|:--|--:|--:|--:|--:|"])
    for c in checks:
        lines.append(f"| {c['dataset']} | {c['pi']:.1f} | {c['L']} | {c['realized_loss']:.6f} | {c['realized_mean_burst']:.4f} |")
    lines.append("")
    output.with_name("link_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    a = arguments(__doc__, "link_impairment.json")
    device = a.device
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print(f"Device {device}; FP32; TF32 off; start={started}", flush=True)
    result, checks, audit, counts = [], [], [], {}
    for d in DATASETS:
        for horizon in (30, 60):
            w = Windows(a.data / d, "test", horizon)
            idx = w.origins[:, None] + np.arange(60 + horizon)
            xyz = np.asarray(w.positions[idx], dtype=np.float64)
            clean, truth = xyz[:, :60], xyz[:, 60:]
            counts[d] = len(clean)
            inputs = []
            if horizon == 30:
                for L in BURST_LENGTHS:
                    for pi in LOSS_RATES:
                        hist, check = gilbert_elliott(clean, pi, L, np.random.default_rng(2026))
                        checks.append(dict(dataset=d, **check))
                        print("generator", d, check, flush=True)
                        inputs.append(("E1", dict(pi=pi, L=L), hist))
            else:
                # E2 levels have no random draws or changes to the observed history.
                inputs.append(("E2", {}, clean))
                for params in CONDITIONS:
                    rng = np.random.default_rng(2026)
                    hist = clean if params["sigma"] == 0 else clean + rng.normal(0., params["sigma"], clean.shape)
                    hist, _ = gilbert_elliott(hist, params["pi"], params["L"], rng)
                    inputs.append(("E3", dict(params), hist))

            def evaluate(method, seed, predict, reference):
                print(d, method, seed, f"H={horizon} start", flush=True)
                clean_pred = predict(clean)
                assert_reference(scores(clean_pred, truth)["ADE"], reference,
                                 (d, method, seed, horizon, "full clean"), audit)
                for experiment, params, hist in inputs:
                    pred = clean_pred if hist is clean else predict(hist)
                    if experiment == "E1":
                        s = scores(pred, truth)
                        if params["pi"] == 0:
                            assert_reference(s["ADE"], reference, (d, method, seed, params), audit)
                        result.append(dict(experiment=experiment, dataset=d, method=method, seed=seed,
                                           params=dict(params), **s))
                    else:
                        ages = AGES if experiment == "E2" else (params["k"],)
                        for k in ages:
                            s = scores(pred[:, k:k + 30], truth[:, k:k + 30])
                            result.append(dict(experiment=experiment, dataset=d, method=method, seed=seed,
                                               params=dict(params, k=k), **s))
                print(d, method, seed, f"H={horizon} done; elapsed={time.perf_counter() - start:.1f}s", flush=True)

            for method in ("CV", "CA"):
                win = next(r["fit_window"] for r in RUNS if r["study"] == "main" and r["dataset"] == d
                           and r["horizon"] == horizon and r["method"] == method)
                evaluate(method, None, lambda hist: kinematic(hist, horizon, method, win),
                         reference(d, horizon, method))
            for rec in learned_records(d, horizon):
                model = load_model(rec, device)

                def predict(hist):
                    std = ((hist - w.mean) / w.scale).astype(np.float32)
                    return neural(model, std, device, w, horizon)

                evaluate(rec["method"], rec["seed"], predict, rec["reported"]["ADE"])
                del model
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            del inputs, clean, truth, xyz, idx, w
    assert len(result) == 2 * (2 + 2 * len(SEEDS)) * (18 + 6 + 3) and len(checks) == 36, (len(result), len(checks))
    keys = [(r["experiment"], r["dataset"], r["method"], r["seed"], json.dumps(r["params"], sort_keys=True))
            for r in result]
    assert len(keys) == len(set(keys))
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
    a.out.with_name("link_impairment_checks.json").write_text(json.dumps(checks, indent=1, allow_nan=False), encoding="utf-8")
    compare_records(result, "link_impairment.json", ("experiment", "dataset", "method", "seed", "params"))
    expected = json.loads((RESULTS / "link_impairment_checks.json").read_text(encoding="utf-8"))
    assert checks == expected, "generator checks differ"
    elapsed = time.perf_counter() - start
    write_summary(result, checks, elapsed, audit, counts, started, a.out)
    print(f"DONE: {len(result)} results, {len(checks)} generator checks, {len(audit)} sanity comparisons; "
          f"runtime={time.perf_counter() - start:.2f}s; no assertions relaxed", flush=True)


if __name__ == "__main__":
    main()
