"""Replay real reception masks/ages on frozen H=30/60 benchmark checkpoints.

Writes only real_trace.json and real_trace_summary.md. No training or data edits.
"""
import sys
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from link_impairment import DATASETS, Windows, assert_reference, kinematic, scores
from link_common import (ROOT, RESULTS, METHODS, RUNS, SEEDS, arguments, compare_records,
                         learned_records, load_model, neural, reference)
from real_trace_masks import EXPLANATION, array_digest, pool_stats


def apply_masks(hist, mask):
    """Same np.interp treatment, axes and boundary behavior as gilbert_elliott."""
    assert mask.shape == (len(hist), 60) and mask[:, 59].all()
    assert np.isin(mask, (0, 1)).all()
    drop = ~mask[:, :59].astype(bool)
    if not drop.any():
        return hist
    out = hist.copy()
    t = np.arange(60)
    for i in np.flatnonzero(drop.any(axis=1)):
        keep = np.r_[~drop[i], True]
        for a in range(3):
            out[i, :, a] = np.interp(t, t[keep], hist[i, keep, a])
    assert np.array_equal(out[:, 59], hist[:, 59])
    assert np.array_equal(out[mask.astype(bool)], hist[mask.astype(bool)])
    assert np.isfinite(out).all()
    return out


def variable_age_slices(pred, truth, age):
    assert pred.shape == truth.shape and pred.shape[1] == 60
    assert len(age) == len(pred) and (age >= 0).all() and (age <= 30).all()
    row = np.arange(len(age))[:, None]
    lead = age[:, None].astype(np.int64) + np.arange(30)
    return pred[row, lead], truth[row, lead]


def sanity_checks():
    # Nonlinear histories expose interpolation and leading-loss boundary behavior.
    hist = (np.arange(2 * 60 * 3, dtype=np.float64).reshape(2, 60, 3) ** 2)
    mask = np.ones((2, 60), dtype=np.uint8)
    assert apply_masks(hist, mask) is hist
    mask[0, :2] = 0
    mask[0, 10:13] = 0
    filled = apply_masks(hist, mask)
    assert np.array_equal(filled[0, :2], np.repeat(hist[0, 2:3], 2, axis=0))
    assert np.array_equal(filled[0, 11], (hist[0, 9] + hist[0, 13]) / 2)
    assert np.array_equal(filled[1], hist[1])
    for age in (np.array([0, 0]), np.array([0, 30]), np.array([3, 5])):
        p, t = variable_age_slices(hist, hist + 1, age)
        for i, k in enumerate(age):
            assert np.array_equal(p[i], hist[i, k:k+30])
            assert np.array_equal(t[i], hist[i, k:k+30] + 1)
        assert np.array_equal(np.linalg.norm(p - t, axis=-1), np.full((2, 30), np.sqrt(3)))


def assignment(pool_mask, pool_age, n, experiment):
    seed = 2026 if experiment == "RL" else 2027
    eligible = np.arange(len(pool_age), dtype=np.int64)
    if experiment == "RLA":
        eligible = eligible[pool_age <= 30]
    assert len(eligible)
    rng = np.random.default_rng(seed)
    indices = eligible[rng.integers(0, len(eligible), size=n)]
    mask, age = pool_mask[indices], pool_age[indices]
    loss_count = (mask[:, :59] == 0).sum(axis=1)
    # Exact upper decile: high loss first, stable original-window order for ties.
    size = math.ceil(n / 10)
    ordered = np.argsort(-loss_count, kind="stable")
    worst = np.zeros(n, dtype=bool)
    worst[ordered[:size]] = True
    cutoff = int(loss_count[ordered[size-1]])
    subsets = dict(all=np.ones(n, dtype=bool), worst_decile=worst)
    if experiment == "RLA":
        assert (age <= 30).all()
        subsets["k_ge_3"] = age >= 3
    assert all(s.any() for s in subsets.values())
    statistics = dict(
        experiment=experiment, seed=seed, sampling="uniform with replacement",
        n_windows=n, eligible_pool_size=len(eligible),
        excluded_fraction=float(1 - len(eligible) / len(pool_age)),
        n_unique_masks_sampled=int(np.unique(indices).size),
        indices=indices.tolist(), indices_sha256=array_digest(indices),
        stats=pool_stats(mask, age),
        subsets={name: dict(n=int(s.sum()), fraction=float(s.mean()),
                            stats=pool_stats(mask[s], age[s])) for name, s in subsets.items()},
        worst_decile=dict(rule="ceil(N/10) highest loss counts; stable window-order tie break",
                          cutoff_lost_slots=cutoff, cutoff_loss_fraction=cutoff / 59,
                          n_at_cutoff=int((loss_count == cutoff).sum()),
                          n_at_cutoff_selected=int(((loss_count == cutoff) & worst).sum()),
                          window_indices=np.flatnonzero(worst).tolist()),
    )
    return mask, age, subsets, statistics


COLUMNS = (
    ("Nominal H30", "nominal_H30", "all", None),
    ("RL", "RL", "all", None),
    ("RL change (%)", "RL", "all", "nominal_H30"),
    ("RL worst 10%", "RL", "worst_decile", None),
    ("H60 nominal [0:30]", "nominal_H60", "all", None),
    ("RLA", "RLA", "all", None),
    ("RLA change (%)", "RLA", "all", "nominal_H60"),
    ("RLA worst 10%", "RLA", "worst_decile", None),
    ("RLA k>=3", "RLA", "k_ge_3", None),
)


def aggregated(rows, dataset, method, experiment, subset, metric):
    selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method
                and r["experiment"] == experiment and r["subset"] == subset]
    expected = [None] if method in METHODS[:2] else list(SEEDS)
    assert sorted((r["seed"] for r in selected), key=str) == sorted(expected, key=str)
    return np.asarray([r[metric] for r in selected])


def summary(document):
    rows, meta = document["records"], document["metadata"]
    lines = ["# Real ADS-B reception replay", "",
             f"Started UTC: {meta['started_utc']}. Evaluation runtime: "
             f"{meta['runtime_seconds']:.2f} s ({meta['runtime_seconds']/60:.2f} min).", "",
             "Frozen test-split checkpoints; FP32; TF32 off. "
             "No packages installed, training, or changes to inputs. Learned cells show "
             "three-seed mean ± sample SD (42/123/456, ddof=1), in metres. "
             "CV/CA are deterministic (det.). Bold is the smallest unrounded mean per column; "
             "lower is better, including percentage changes. Changes are ratios of seed means.", "",
             "Each dataset uses a fresh default_rng(2026) for RL and default_rng(2027) for "
             "RLA; uniform mask draws with replacement are shared by every method/seed. "
             "Assignments (pool indices), hashes and subset diagnostics are in real_trace.json. "
             "RL ignores sampled ages and scores H=30. RLA conditions the pool on k<=30, "
             "uses H=60 checkpoints, and scores each unchanged future slice [k:k+30], "
             "i.e. leads k+1 through k+30 after the latest report; FDE uses lead k+30. "
             "History ends at the latest report; missing positions use the exact np.interp "
             "treatment in link_impairment.py, including constant fill before the first keep.", "",
             "Worst decile is selected separately per experiment from assigned loss counts "
             "over slots 0..58: ceil(N/10) highest counts, with ties broken by original test-window "
             "order. Subsets and inputs are identical for all methods/seeds. Nominal columns "
             "use all test windows; the two percentage columns compare all-window RL/RLA "
             "against the corresponding nominal column, respectively.", "",
             f"Strict reference checks: {len(meta['reference_relative_errors'])} passed with "
             f"relative error <1e-5; maximum {max(meta['reference_relative_errors']):.3e}. "
             "Checks cover clean H30 full ADE and clean H60 full-future ADE for every "
             "method/seed. H60 nominal [0:30] is measured separately. No assertions relaxed. "
             "Mask parsing reproduces the original full-file calibration exactly.", "",
             "Pool interpretation: " + EXPLANATION, ""]
    rankings = {}
    for dataset in DATASETS:
        pool = document["pools"][dataset]
        s, c = pool["stats"], pool["calibration"]
        assignments = document["assignments"][dataset]
        lines += [f"## {dataset}", "",
                  f"Pool: {s['N']:,} masks from {pool['n_files']} files / {c['n_tracks']:,} tracks; "
                  f"mean loss {s['mean_loss_slots_0_58']:.4%}; mean burst "
                  f"{s['mean_burst_length']:.4f} s; any loss {s['fraction_masks_any_loss']:.4%}. "
                  f"Calibration pooled slots: {c['pi']:.4%} loss, "
                  f"{c['mean_burst_length_reports']:.4f} s mean burst. "
                  f"Age>30: {s['fraction_age_gt_30']:.6%} (RLA excluded fraction).", "",
                  "Pool integer age percentiles (s): " + ", ".join(
                      f"{k}={v:g}" for k, v in s["age_percentiles"].items()) + ". "
                  "Fractions: " + ", ".join(f"{k}={v:.4%}" for k, v in s["age_fractions"].items()) + ".", ""]
        for exp in ("RL", "RLA"):
            a = assignments[exp]
            counts = ", ".join(f"{k}: {v['n']:,}" for k, v in a["subsets"].items())
            lines += [f"{exp} assignments — {counts}; loss "
                      f"{a['stats']['mean_loss_slots_0_58']:.4%}, burst "
                      f"{a['stats']['mean_burst_length']:.4f} s; worst-decile cutoff "
                      f"{a['worst_decile']['cutoff_lost_slots']}/59 lost slots; excluded pool "
                      f"{a['excluded_fraction']:.6%}.", ""]
        rankings[dataset] = {}
        for metric in ("ADE", "FDE"):
            lines += [f"### {metric} (m; percentage columns in %)", ""]
            values, sds = {}, {}
            for method in METHODS:
                for j, (_, exp, sub, baseline) in enumerate(COLUMNS):
                    v = aggregated(rows, dataset, method, exp, sub, metric)
                    if baseline:
                        b = aggregated(rows, dataset, method, baseline, "all", metric)
                        values[method, j] = float(100 * (v.mean() / b.mean() - 1))
                        sds[method, j] = None
                    else:
                        values[method, j] = float(v.mean())
                        sds[method, j] = float(v.std(ddof=1)) if len(v) > 1 else None
            best = [min(values[m, j] for m in METHODS) for j in range(len(COLUMNS))]
            lines += ["| Method | " + " | ".join(x[0] for x in COLUMNS) + " |",
                      "|:--|" + "--:|" * len(COLUMNS)]
            for method in METHODS:
                cells = []
                for j, (_, _, _, baseline) in enumerate(COLUMNS):
                    value, sd = values[method, j], sds[method, j]
                    cell = f"{value:+.2f}" if baseline else (
                        f"{value:.2f} ± {sd:.2f}" if sd is not None else f"{value:.2f} (det.)")
                    cells.append(f"**{cell}**" if value == best[j] else cell)
                lines.append("| " + method + " | " + " | ".join(cells) + " |")
            lines.append("")
            rankings[dataset][metric] = {}
            for j, (heading, _, _, _) in enumerate(COLUMNS):
                rank = 1 + sum(values[m, j] < values["PA-TSMixer", j] for m in METHODS)
                winners = [m for m in METHODS if values[m, j] == best[j]]
                competitor = min((m for m in METHODS if m != "PA-TSMixer"), key=lambda m: values[m, j])
                rankings[dataset][metric][heading] = dict(
                    pa_rank=rank, pa_value=values["PA-TSMixer", j], best_methods=winners,
                    best_value=best[j], best_competitor=competitor,
                    best_competitor_value=values[competitor, j])
                lines.append(f"- {heading}: PA-TSMixer {values['PA-TSMixer', j]:.4f}, rank "
                             f"{rank}/{len(METHODS)}; best {', '.join(winners)} ({best[j]:.4f}); "
                             f"best competitor {competitor} ({values[competitor, j]:.4f}).")
            lines.append("")
    lines += ["## Implementation and scope", "",
              "Existing parser, retained-track/slot construction, test-window loader, normalization, "
              "frozen-checkpoint configuration, forward pass, kinematic fit windows, scoring and "
              "reference assertions are imported unchanged. The existing parser's recovery of "
              "malformed/empty files is retained and recorded in the NPZ metadata and JSON. "
              "Recomputed released-model results are written to the selected output directory. "
              "No corrections to existing scripts or results were required. "
              "This is reception-pattern replay onto processed test trajectories; it does not "
              "claim trajectory-matched raw reports, independent RF packet loss, or an independent "
              "held-out reception campaign. Overlapping masks remain in the empirical pool, "
              "as requested; seed SD describes model seeds, not reception-sampling uncertainty.", ""]
    document["rankings"] = rankings
    return "\n".join(lines)


def main():
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    sanity_checks()
    cli = arguments(__doc__, "real_trace.json")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = cli.device
    print(f"Device {device}; FP32; TF32 off; {started}", flush=True)
    rows, audit, pools, assignments, counts = [], [], {}, {}, {}
    reference_details, sources = [], []
    for dataset in DATASETS:
        with np.load(RESULTS / f"real_masks_{dataset}.npz", allow_pickle=False) as z:
            mask_pool, age_pool = z["mask"], z["age"]
            pools[dataset] = json.loads(str(z["metadata_json"].item()))
        assert pool_stats(mask_pool, age_pool) == pools[dataset]["stats"]
        assignments[dataset] = {}
        original_origins = None
        for horizon, experiment in ((30, "RL"), (60, "RLA")):
            w = Windows(cli.data / dataset, "test", horizon)
            if original_origins is None:
                original_origins = w.origins.copy()
            else:
                assert np.array_equal(w.origins, original_origins)
            idx = w.origins[:, None] + np.arange(60 + horizon)
            xyz = np.asarray(w.positions[idx], dtype=np.float64)
            clean, truth = xyz[:, :60], xyz[:, 60:]
            counts[dataset] = len(clean)
            masks, ages, subsets, a = assignment(mask_pool, age_pool, len(clean), experiment)
            assignments[dataset][experiment] = a
            impaired = apply_masks(clean, masks)
            print(dataset, experiment, "assignment", json.dumps({
                "n": len(clean), "seed": a["seed"], "stats": a["stats"],
                "subsets": {s: int(v.sum()) for s, v in subsets.items()},
                "excluded_fraction": a["excluded_fraction"]}), flush=True)

            def evaluate(method, seed, predict, reference):
                clean_pred = predict(clean)
                measured = scores(clean_pred, truth)["ADE"]
                assert_reference(measured, reference, (dataset, method, seed, horizon, "full clean"), audit)
                reference_details.append(dict(dataset=dataset, method=method, seed=seed,
                                              horizon=horizon, ADE=measured, reference_ADE=reference,
                                              relative_error=audit[-1]))
                nominal = scores(clean_pred[:, :30], truth[:, :30])
                rows.append(dict(experiment=f"nominal_H{horizon}", subset="all", dataset=dataset,
                                 method=method, seed=seed, params=dict(horizon=horizon, k=0),
                                 n_windows=len(clean), **nominal))
                pred = predict(impaired)
                p, target = (pred, truth) if horizon == 30 else variable_age_slices(pred, truth, ages)
                for name, selected in subsets.items():
                    rows.append(dict(experiment=experiment, subset=name, dataset=dataset,
                                     method=method, seed=seed,
                                     params=dict(horizon=horizon, assignment_seed=a["seed"],
                                                 k="per_window" if horizon == 60 else 0),
                                     n_windows=int(selected.sum()), **scores(p[selected], target[selected])))
                print(f"{dataset} H={horizon} {method} seed={seed}: "
                      f"nominal ADE={nominal['ADE']:.6f}; {experiment} ADE={scores(p,target)['ADE']:.6f}; "
                      f"elapsed={time.perf_counter()-start:.1f}s", flush=True)

            for method in METHODS[:2]:
                win = next(r["fit_window"] for r in RUNS if r["study"] == "main"
                           and r["dataset"] == dataset and r["horizon"] == horizon and r["method"] == method)
                evaluate(method, None, lambda hist: kinematic(hist, horizon, method, win),
                         reference(dataset, horizon, method))
            for rec in learned_records(dataset, horizon):
                model = load_model(rec, device)
                assert not model.training
                assert all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point())

                def predict(hist):
                    std = ((hist - w.mean) / w.scale).astype(np.float32)
                    return neural(model, std, device, w, horizon)

                sources.append(dict(dataset=dataset, horizon=horizon, method=rec["method"],
                                    seed=rec["seed"], checkpoint="checkpoints/" + rec["file"]))
                evaluate(rec["method"], rec["seed"], predict, rec["reported"]["ADE"])
                del model
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            del w, idx, xyz, clean, truth, impaired, masks
        del mask_pool, age_pool
    assert len(rows) == 2 * (2 + 2 * len(SEEDS)) * 7 and len(audit) == 2 * 2 * (2 + 2 * len(SEEDS)), (len(rows), len(audit))
    keys = [(r["experiment"], r["subset"], r["dataset"], r["method"], r["seed"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    document = dict(
        metadata=dict(started_utc=started, runtime_seconds=time.perf_counter()-start,
                      device=str(device), precision="FP32",
                      tf32=False, python=sys.version, numpy_version=np.__version__,
                      torch_version=torch.__version__, test_window_counts=counts,
                      reference_relative_errors=audit, reference_checks=reference_details,
                      checkpoints=sources, assertions_relaxed=False,
                      script_sha256={name: hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                                     for name in ("real_trace.py", "real_trace_masks.py",
                                                  "link_impairment.py", "link_calibration.py")}),
        pools=pools, assignments=assignments, records=rows)
    compare_records(rows, "real_trace.json", ("experiment", "subset", "dataset", "method", "seed"))
    expected = json.loads((RESULTS / "real_trace.json").read_text(encoding="utf-8"))
    assert assignments == expected["assignments"], "replay assignments differ"
    markdown = summary(document)
    cli.out.write_text(json.dumps(document, indent=1, allow_nan=False), encoding="utf-8")
    cli.out.with_name("real_trace_summary.md").write_text(markdown, encoding="utf-8")
    print(f"DONE {len(rows)} records; {len(audit)} strict reference checks; "
          f"max relative error={max(audit):.3e}; runtime={time.perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
