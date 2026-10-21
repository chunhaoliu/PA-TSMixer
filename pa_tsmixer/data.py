"""Processed XYZ benchmark used in the paper: construction from the public sources and window access.

Protocol (Section V-A): published processed XYZ in kilometres -> metres; per-aircraft one-second segments,
cut at time gaps and at gross jumps (>300 m horizontal or >100 m vertical in one second); segments shorter
than 120 points are discarded. Windows of 60 observed + 60 future points start every 15 points; identical
windows of the same aircraft are removed. Aircraft are assigned to train/val/test (70/15/15) by a fixed
SHA-256 rule. Normalization uses the per-axis mean and standard deviation of the training points.
All horizons share the same window origins.
"""
import collections, hashlib, io, json, zipfile
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
HISTORY, MAX_FUTURE, STRIDE = 60, 60, 15
WINDOW = HISTORY + MAX_FUTURE


def split_code(aircraft_id):
    u = int.from_bytes(hashlib.sha256(("TVT-aircraft-split-42|" + str(aircraft_id)).encode()).digest()[:8], "big") / 2**64
    return 0 if u < .70 else 1 if u < .85 else 2


def _sources(name, source):
    """Yield the raw files in the order used for the paper build."""
    if name == "trajair":       # directory of the TrajAir '111_days' processed text files
        for p in sorted(Path(source).rglob("*.txt")):
            payload = p.read_bytes()
            yield np.loadtxt(io.StringIO(payload.decode("utf-8")), dtype=np.float64, ndmin=2) if payload.strip() else None
    else:                       # TartanAviation KBTP processed.zip
        with zipfile.ZipFile(source) as z:
            for member in z.infolist():
                payload = z.read(member.filename)
                if not payload.strip():
                    yield None
                    continue
                text = io.TextIOWrapper(io.BytesIO(payload), encoding="utf-8", newline=None)
                yield np.loadtxt(text, dtype=np.float64, ndmin=2, delimiter=",")


def build(name, source, output):
    """Write positions.npy, segments.npy, origins.npy and scaler.json for one source."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    points, segments, origins, seen = [], [], [], set()
    total, counts = 0, collections.Counter()
    train_sum, train_sq, train_n = np.zeros(3), np.zeros(3), 0
    for source_index, a in enumerate(_sources(name, source)):
        if a is None:
            continue
        for aid_value in np.unique(a[:, 1]):
            aid = int(aid_value)
            t = a[a[:, 1] == aid_value]
            t = t[np.argsort(t[:, 0], kind="stable")]
            dt = np.diff(t[:, 0])
            xyz = t[:, 2:5] * 1000.
            horizontal = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
            vertical = np.abs(np.diff(xyz[:, 2]))
            breaks = (dt != 1) | (horizontal > 300.) | (vertical > 100.)
            bounds = np.r_[0, np.flatnonzero(breaks) + 1, len(t)]
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                length = int(hi - lo)
                if length < WINDOW:
                    continue
                seg = xyz[lo:hi].astype(np.float32)
                code = split_code(aid)
                segments.append((total, length, aid, code, source_index, int(t[lo, 0])))
                points.append(seg)
                if code == 0:
                    q = seg.astype(np.float64)
                    train_sum += q.sum(axis=0)
                    train_sq += (q * q).sum(axis=0)
                    train_n += length
                for offset in range(0, length - WINDOW + 1, STRIDE):
                    key = hashlib.sha256(str(aid).encode() + b"|" + seg[offset:offset + WINDOW].tobytes()).digest()
                    if key in seen:
                        continue
                    seen.add(key)
                    origins.append((total + offset, len(segments) - 1, code))
                    counts[SPLITS[code]] += 1
                total += length
    np.save(output / "positions.npy", np.concatenate(points), allow_pickle=False)
    np.save(output / "segments.npy", np.asarray(segments, dtype=np.int64), allow_pickle=False)
    np.save(output / "origins.npy", np.asarray(origins, dtype=np.int64), allow_pickle=False)
    mean = train_sum / train_n
    var = np.maximum(train_sq / train_n - mean * mean, 0.)
    scale = np.sqrt(var)
    scale[scale < 1e-8] = 1.
    scaler = dict(mean=mean.tolist(), var=var.tolist(), scale=scale.tolist(), n_samples_seen=train_n,
                  n_features_in=3, split="train", fit_scope="retained source points, not repeated overlapping windows")
    (output / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    return dict(points=total, segments=len(segments), windows=dict(counts))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Windows:
    """Standardized history/target windows of one split for horizon ``pred_len``."""

    def __init__(self, directory, split, pred_len):
        directory = Path(directory)
        self.positions = np.load(directory / "positions.npy", mmap_mode="r")
        scaler = json.loads((directory / "scaler.json").read_text(encoding="utf-8"))
        self.mean, self.scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        origins = np.load(directory / "origins.npy")
        self.origins = origins[origins[:, 2] == SPLITS.index(split), 0]
        self.pred_len = pred_len

    def __len__(self):
        return len(self.origins)

    def batch(self, start, stop):
        """Return standardized history [n, 60, 3] (float32) and physical future [n, H, 3] (float64, metres)."""
        idx = self.origins[start:stop, None] + np.arange(HISTORY + self.pred_len)
        xyz = np.asarray(self.positions[idx], dtype=np.float64)
        history = ((xyz[:, :HISTORY] - self.mean) / self.scale).astype(np.float32)
        return history, xyz[:, HISTORY:]

    def to_metres(self, standardized):
        return standardized.astype(np.float64) * self.scale + self.mean
