"""Constant-velocity (CV) and constant-acceleration (CA) extrapolation baselines on the test split.

Each method fits a least-squares velocity (CV) or velocity and acceleration (CA) to the last ``window`` observed
positions, relative to the last observation, and extrapolates it over the horizon. The window was selected on
the validation split; the selected values are stored in results/runs.json and reused here.

    python scripts/kinematic_baselines.py --data data
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pa_tsmixer.data import HISTORY, Windows


def predict(history, horizon, method, window):
    """history: [N, 60, 3] physical metres -> forecast [N, horizon, 3]."""
    t = np.arange(1 - window, 1, dtype=np.float64)
    design = t[:, None] if method == "CV" else np.column_stack((t, .5 * t * t))
    coef = np.einsum("ak,nkd->nad", np.linalg.pinv(design), history[:, -window:] - history[:, -1:, :])
    f = np.arange(1, horizon + 1, dtype=np.float64)
    future = f[:, None] if method == "CV" else np.column_stack((f, .5 * f * f))
    return history[:, -1:, :] + np.einsum("ha,nad->nhd", future, coef)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--tolerance", type=float, default=1e-9, help="maximum allowed relative difference")
    a = p.parse_args()
    runs = [r for r in json.loads((ROOT / "results/runs.json").read_text(encoding="utf-8"))
            if r["study"] == "main" and r["method"] in ("CV", "CA")]
    worst = 0.
    for r in runs:
        w = Windows(a.data / r["dataset"], "test", r["horizon"])
        idx = w.origins[:, None] + np.arange(HISTORY + r["horizon"])
        xyz = np.asarray(w.positions[idx], dtype=np.float64)
        error = predict(xyz[:, :HISTORY], r["horizon"], r["method"], r["fit_window"]) - xyz[:, HISTORY:]
        d = np.linalg.norm(error, axis=-1)
        m = dict(ADE=d.mean(), FDE=d[:, -1].mean(), MSE=np.mean(error * error))
        diff = max(abs(m[k] - r[k]) / r[k] for k in m)
        worst = max(worst, diff)
        print(f"{r['dataset']:15s} H={r['horizon']:2d} {r['method']} (window {r['fit_window']:2d})  "
              f"ADE {m['ADE']:8.2f}  FDE {m['FDE']:8.2f}  MSE {m['MSE']:10.1f}  rel. diff {diff:.1e}")
    print(f"{len(runs)} runs; largest relative difference from the reported values {worst:.1e}")
    return 0 if worst <= a.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
