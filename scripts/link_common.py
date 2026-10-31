"""Shared public-checkpoint loading and strict comparisons for the link experiments."""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/link"
sys.path.insert(0, str(ROOT))
from pa_tsmixer.models import build

METHODS = ("CV", "CA", "TSMixer", "PA-TSMixer")
SEEDS = (42, 123, 456)
RUNS = json.loads((ROOT / "results/runs.json").read_text(encoding="utf-8"))


def arguments(description, filename):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=RESULTS / "recomputed" / filename,
                   help="recomputed JSON; published all-method results are retained separately")
    a = p.parse_args()
    if a.out.resolve().parent == RESULTS.resolve():
        p.error("--out must use a separate directory to preserve the published result files")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return a


def learned_records(dataset, horizon):
    index = json.loads((ROOT / "checkpoints/index.json").read_text(encoding="utf-8"))
    records = [e for e in index if e["study"] == "main" and e["dataset"] == dataset
               and e["horizon"] == horizon]
    actual = [(e["method"], e["seed"]) for e in records]
    assert len(actual) == 6 and set(actual) == {(m, s) for m in METHODS[2:] for s in SEEDS}, actual
    return sorted(records, key=lambda e: (METHODS.index(e["method"]), e["seed"]))


def load_model(record, device):
    model = build(record["architecture"], record["horizon"], **record["kwargs"]).float()
    assert model.projection.in_features == 60 and model.projection.out_features == record["horizon"]
    model.load_state_dict(torch.load(ROOT / "checkpoints" / record["file"],
                                    map_location="cpu", weights_only=True))
    return model.eval().to(device)


def reference(dataset, horizon, method, seed=None):
    return next(r["ADE"] for r in RUNS if r["study"] == "main" and r["dataset"] == dataset
                and r["horizon"] == horizon and r["method"] == method and r["seed"] == seed)


@torch.inference_mode()
def neural(model, hist_std, device, windows, horizon):
    preds = []
    for s in range(0, len(hist_std), 4096):
        x = torch.from_numpy(hist_std[s:s + 4096]).float().to(device)
        preds.append(model(x)[:, -horizon:].float().cpu().numpy())
    return windows.to_metres(np.concatenate(preds))


def compare_records(rows, filename, fields):
    """Require all released-method rows and both full-precision metrics to agree."""
    expected = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
    if isinstance(expected, dict):
        expected = expected["records"]
    expected = [r for r in expected if r["method"] in METHODS]
    key = lambda r: tuple(json.dumps(r[f], sort_keys=True) for f in fields)
    actual = {key(r): r for r in rows}
    published = {key(r): r for r in expected}
    assert len(actual) == len(rows) and len(published) == len(expected)
    assert actual.keys() == published.keys(), "missing or extra released-model rows"
    worst = 0.
    for k, row in actual.items():
        for metric in ("ADE", "FDE"):
            ref = published[k][metric]
            diff = abs(row[metric] - ref) / abs(ref) if ref else abs(row[metric] - ref)
            assert diff < 1e-5, (k, metric, row[metric], ref, diff)
            worst = max(worst, diff)
    print(f"{filename}: {len(rows)} released-model rows; max ADE/FDE relative difference {worst:.12e}",
          flush=True)
    return worst
