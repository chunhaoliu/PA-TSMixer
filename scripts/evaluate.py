"""Evaluate released checkpoints on the test split and compare with the values reported in the paper.

    python scripts/evaluate.py --data data                       # all 96 checkpoints
    python scripts/evaluate.py --data data --study main --dataset trajair --horizon 30

ADE and FDE are 3D displacement errors in metres; MSE is the mean squared XYZ coordinate error (m^2).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pa_tsmixer.data import Windows
from pa_tsmixer.models import build


@torch.inference_mode()
def evaluate(model, windows, device, batch=1024):
    model.eval().to(device)
    n, h = len(windows), windows.pred_len
    ade = fde = sq = 0.
    for start in range(0, n, batch):
        history, truth = windows.batch(start, min(n, start + batch))
        out = model(torch.from_numpy(history).to(device)).float().cpu().numpy()
        error = windows.to_metres(out) - truth
        distance = np.linalg.norm(error, axis=-1)
        ade += distance.sum(); fde += distance[:, -1].sum(); sq += np.sum(error * error)
    return dict(ADE=ade / (n * h), FDE=fde / n, MSE=sq / (n * h * 3))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--study", choices=["main", "ablation"])
    p.add_argument("--dataset", choices=["trajair", "tartanaviation"])
    p.add_argument("--horizon", type=int, choices=[10, 20, 30, 60])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tolerance", type=float, default=1e-6, help="maximum allowed relative difference")
    p.add_argument("--output", type=Path, help="optional JSON file for the recomputed metrics")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    index = json.loads((ROOT / "checkpoints/index.json").read_text(encoding="utf-8"))
    selected = [e for e in index if (a.study in (None, e["study"])) and (a.dataset in (None, e["dataset"]))
                and (a.horizon in (None, e["horizon"]))]
    cache, rows, worst = {}, [], 0.
    for e in selected:
        key = (e["dataset"], e["horizon"])
        if key not in cache:
            cache[key] = Windows(a.data / e["dataset"], "test", e["horizon"])
        model = build(e["architecture"], e["horizon"], **e["kwargs"])
        model.load_state_dict(torch.load(ROOT / "checkpoints" / e["file"], map_location="cpu", weights_only=True))
        m = evaluate(model, cache[key], a.device)
        diff = max(abs(m[k] - e["reported"][k]) / abs(e["reported"][k]) for k in m)
        worst = max(worst, diff)
        rows.append(dict(file=e["file"], **m, max_relative_difference=diff))
        print(f"{e['file']:58s} ADE {m['ADE']:8.2f}  FDE {m['FDE']:8.2f}  MSE {m['MSE']:10.1f}  rel. diff {diff:.1e}",
              flush=True)
    if a.output:
        a.output.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} checkpoints; largest relative difference from the reported values {worst:.1e}")
    return 0 if worst <= a.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
