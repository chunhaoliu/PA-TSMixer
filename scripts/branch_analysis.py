"""Learned scale weights and test-time branch removal for the 24 main PA-TSMixer checkpoints (Fig. 7).

For each checkpoint: the softmax scale weights of each mixing block, and the test ADE with one temporal branch
silenced (its output set to zero in both the weighted sum and the fusion input, in every block), overall and per
history-derived motion group. No model is retrained.

    python scripts/branch_analysis.py --data data          # writes results/branch_analysis.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pa_tsmixer.data import Windows
from pa_tsmixer.models import PATSMixer
from pa_tsmixer.motion import EXPECTED, groups

BRANCHES = ("fine", "medium", "coarse")


def forward_masked(temporal, x, keep):
    length = x.shape[-1]
    fine = temporal.fine(x) * keep[0]
    med = F.interpolate(temporal.med(F.avg_pool1d(x, 2, 2)), size=length, mode="linear", align_corners=False) * keep[1]
    coarse = F.interpolate(temporal.coarse(F.avg_pool1d(x, 4, 4)), size=length, mode="linear",
                           align_corners=False) * keep[2]
    w = F.softmax(temporal.scale_w, dim=0)
    return w[0] * fine + w[1] * med + w[2] * coarse + temporal.fuse(torch.cat([fine, med, coarse], dim=1))


@torch.inference_mode()
def window_ade(model, windows, device, keep):
    for block in model.model:
        block.temporal.forward = lambda x, t=block.temporal: forward_masked(t, x, keep)
    out = []
    for start in range(0, len(windows), 4096):
        history, truth = windows.batch(start, min(len(windows), start + 4096))
        pred = windows.to_metres(model(torch.from_numpy(history).to(device)).cpu().numpy())
        out.append(np.linalg.norm(pred - truth, axis=-1).mean(axis=1))
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=ROOT / "results/branch_analysis.json")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    index = [e for e in json.loads((ROOT / "checkpoints/index.json").read_text(encoding="utf-8"))
             if e["study"] == "main" and e["architecture"] == "PA-TSMixer"]
    cache, result = {}, []
    for e in index:
        d, h = e["dataset"], e["horizon"]
        if (d, h) not in cache:
            w = Windows(a.data / d, "test", h)
            masks = groups(np.asarray(w.positions[w.origins[:, None] + np.arange(60)], dtype=np.float64))
            assert {k: int(v.sum()) for k, v in masks.items()} == EXPECTED[d], "motion groups differ from the paper"
            cache[(d, h)] = (w, masks)
        w, masks = cache[(d, h)]
        model = PATSMixer(h)
        model.load_state_dict(torch.load(ROOT / "checkpoints" / e["file"], map_location="cpu", weights_only=True))
        model.eval().to(a.device)
        weights = [F.softmax(b.temporal.scale_w.detach().cpu(), 0).tolist() for b in model.model]
        full = window_ade(model, w, a.device, (1., 1., 1.))
        row = dict(dataset=d, horizon=h, seed=e["seed"], weights=weights, ADE=float(full.mean()),
                   groups={g: float(full[m].mean()) for g, m in masks.items()}, removed={})
        for i, name in enumerate(BRANCHES):
            keep = [1., 1., 1.]
            keep[i] = 0.
            ade = window_ade(model, w, a.device, tuple(keep))
            row["removed"][name] = dict(ADE=float(ade.mean()), groups={g: float(ade[m].mean()) for g, m in masks.items()})
        result.append(row)
        inc = {k: f"+{100 * (v['ADE'] / row['ADE'] - 1):.1f}%" for k, v in row["removed"].items()}
        print(f"{d:15s} H={h:2d} seed {e['seed']:3d}  block-1 weights {np.round(weights[0], 3)}  removal {inc}",
              flush=True)
    a.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for d in ("trajair", "tartanaviation"):
        coarse = [np.mean([100 * (r["removed"]["coarse"]["ADE"] / r["ADE"] - 1) for r in result
                           if r["dataset"] == d and r["horizon"] == h]) for h in (10, 20, 30, 60)]
        print(f"{d}: ADE increase without the quarter-resolution branch, H=10/20/30/60:",
              " / ".join(f"{c:.1f}%" for c in coarse))


if __name__ == "__main__":
    main()
