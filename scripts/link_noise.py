"""Robustness of the trained H=30 models (seeds 42/123/456) and CV/CA to degraded input histories (no retraining).

Noise: zero-mean Gaussian position noise with standard deviation sigma (m) on every observed point and axis.
Missing: a fraction p of the observed points, excluding the most recent one, is removed at random and filled by
linear interpolation in time (the first point, if removed, takes the nearest retained value).
Targets are always the clean future positions. One fixed perturbation per window and level (seed 2026) is shared
by all released models (PA-TSMixer, TSMixer, CV, CA; Fig. 6, Section VI-D). Writes robustness.json with test ADE/FDE per model, seed, level.
"""
import json
import numpy as np
import torch
from link_common import (RUNS, arguments, compare_records, learned_records, load_model, neural)
from pa_tsmixer.data import Windows
from kinematic_baselines import predict as kinematic

NOISE = [0, 5, 10, 20, 40]
MISSING = [0, .1, .2, .3, .4]
H = 30


def perturb(hist, kind, level, rng):
    if level == 0:
        return hist
    if kind == "noise":
        return hist + rng.normal(0., level, hist.shape)
    out = hist.copy()
    t = np.arange(60)
    drop = rng.random((len(hist), 59)) < level          # never drop the most recent point
    for i in np.flatnonzero(drop.any(axis=1)):
        keep = np.r_[~drop[i], True]
        for a in range(3):
            out[i, :, a] = np.interp(t, t[keep], hist[i, keep, a])
    return out


def scores(pred, truth):
    d = np.linalg.norm(pred - truth, axis=-1)
    return dict(ADE=float(d.mean()), FDE=float(d[:, -1].mean()))


def main():
    a = arguments(__doc__, "robustness.json")
    device = a.device
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    result = []
    for d in ("trajair", "tartanaviation"):
        w = Windows(a.data / d, "test", H)
        idx = w.origins[:, None] + np.arange(60 + H)
        xyz = np.asarray(w.positions[idx], dtype=np.float64)
        clean, truth = xyz[:, :60], xyz[:, 60:]
        inputs = {}
        for kind, levels in (("noise", NOISE), ("missing", MISSING)):
            for lv in levels:
                inputs[(kind, lv)] = perturb(clean, kind, lv, np.random.default_rng(2026))
        for method in ("CV", "CA"):
            win = next(r["fit_window"] for r in RUNS if r["study"] == "main" and r["dataset"] == d
                       and r["horizon"] == H and r["method"] == method)
            for (kind, lv), hist in inputs.items():
                result.append(dict(dataset=d, method=method, seed=None, kind=kind, level=lv,
                                   **scores(kinematic(hist, H, method, win), truth)))
        for rec in learned_records(d, H):
            model = load_model(rec, device)
            for (kind, lv), hist in inputs.items():
                std = ((hist - w.mean) / w.scale).astype(np.float32)
                s = scores(neural(model, std, device, w, H), truth)
                if lv == 0:
                    ref = rec["reported"]["ADE"]          # saved test ADE of this run
                    assert abs(s["ADE"] - ref) / ref < 1e-5, (rec["file"], s, ref)
                result.append(dict(dataset=d, method=rec["method"], seed=rec["seed"], kind=kind, level=lv, **s))
            print(d, rec["method"], rec["seed"], flush=True)
    a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    compare_records(result, "robustness.json", ("dataset", "method", "seed", "kind", "level"))


if __name__ == "__main__":
    main()
