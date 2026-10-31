"""Build real 1-Hz reception-history/age pools from all calibrated KBTP logs.

Only the two real_masks_*.npz files are written. Run with Python -B.
True/1 means received. Tracks and source files are never joined.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from link_calibration import (RESULTS, Distribution, PooledSlots, raw_arguments, source_name,
                              discover_files, iter_tracks, read_file, track_slots)

DATASETS = ("trajair", "tartanaviation")
RADII = dict(trajair=6.0, tartanaviation=11.0)
EXPLANATION = (
    "Calibration pools every retained track slot once; this pool samples now slots "
    "59,74,89,... and shifts each history to its latest received slot j. Histories "
    "overlap, require j>=59, can repeat for different now slots, and measure loss "
    "only in slots 0..58. Thus track weighting, boundaries, and conditioning on a "
    "received endpoint differ. Bursts are pooled within-history lost runs, including "
    "left-boundary truncation. Integer slot age n-j differs from calibration's "
    "continuous, right-endpoint-exclusive window age and logged snapshot Age. "
    "These are empirical reception gaps, not identified RF packet losses."
)


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pool_stats(mask, age):
    assert mask.ndim == 2 and mask.shape[1] == 60 and len(mask) == len(age) > 0
    assert np.isin(mask, (0, 1)).all() and mask[:, -1].all()
    assert np.issubdtype(age.dtype, np.integer) and (age >= 0).all()
    drop = ~mask[:, :59].astype(bool)
    starts = drop & ~np.concatenate((np.zeros((len(drop), 1), bool), drop[:, :-1]), axis=1)
    lost, runs = int(drop.sum()), int(starts.sum())
    ages, counts = np.unique(age, return_counts=True)
    return dict(
        N=len(mask), mean_loss_slots_0_58=float(drop.mean()),
        lost_slots=lost, loss_bursts=runs,
        mean_burst_length=float(lost / runs) if runs else 0.0,
        fraction_masks_any_loss=float(drop.any(axis=1).mean()),
        age_mean=float(age.mean()), age_max=int(age.max()),
        age_percentiles={f"p{q:g}": float(np.percentile(age, q))
                         for q in (0, 25, 50, 75, 90, 95, 99, 99.9, 100)},
        age_histogram={str(int(k)): int(n) for k, n in zip(ages, counts)},
        age_fractions={f"ge_{k}": float((age >= k).mean()) for k in (1, 2, 3, 5)},
        fraction_age_gt_30=float((age > 30).mean()),
        mask_sha256=array_digest(mask), age_sha256=array_digest(age),
    )


def build_pool(dataset, raw_root, out, calibration_path):
    started = time.perf_counter()
    files = discover_files(raw_root, dataset, None)
    pooled = PooledSlots()
    mask_chunks, age_chunks, id_chunks, now_chunks, report_chunks = [], [], [], [], []
    track_file, track_aircraft, track_start, track_end = [], [], [], []
    diagnostics = []
    n_candidates = n_skipped = 0
    # read_file needs collectors for its original diagnostics; avoid retaining rows.
    for file_index, path in enumerate(files):
        reports, diagnostic = read_file(
            path, dataset, RADII[dataset], Distribution((50,)), Distribution((50,)))
        diagnostic["file"] = source_name(path, dataset)
        diagnostics.append(diagnostic)
        for aircraft, group in reports.groupby("ID", sort=False):
            for track in iter_tracks(group["timestamp"].to_numpy(dtype=np.float64), 30.0):
                info = track_slots(track)
                pooled.add(track, info)
                _, received, n_slots, _, _ = info
                tid = len(track_file)
                track_file.append(file_index)
                track_aircraft.append(str(aircraft))
                track_start.append(float(track[0]))
                track_end.append(float(track[-1]))
                rec = np.zeros(n_slots, dtype=bool)
                rec[received] = True
                now = np.arange(59, n_slots, 15, dtype=np.int64)
                n_candidates += len(now)
                latest = received[np.searchsorted(received, now, side="right") - 1]
                valid = latest >= 59
                n_skipped += int((~valid).sum())
                now, latest = now[valid], latest[valid]
                if not len(now):
                    continue
                age = now - latest
                assert age.max() <= np.iinfo(np.int16).max
                masks = rec[latest[:, None] + np.arange(-59, 1)].astype(np.uint8)
                assert masks[:, -1].all()
                mask_chunks.append(masks)
                age_chunks.append(age.astype(np.int16))
                id_chunks.append(np.full(len(now), tid, dtype=np.int32))
                now_chunks.append(now.astype(np.int32))
                report_chunks.append(latest.astype(np.int32))
        if (file_index + 1) % 10 == 0 or file_index + 1 == len(files):
            print(f"{dataset}: {file_index + 1}/{len(files)} files; "
                  f"tracks={pooled.n_tracks:,}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    mask, age = np.concatenate(mask_chunks), np.concatenate(age_chunks)
    stats = pool_stats(mask, age)
    calibration = pooled.summarize()
    # Check exact reproduction of the original full-file parsing/track population.
    reference = json.loads(calibration_path.read_text(encoding="utf-8"))["datasets"][dataset]
    for key in ("n_tracks", "n_reports", "n_slots", "n_lost_slots", "n_loss_bursts"):
        assert calibration[key] == reference[key], (dataset, key, calibration[key], reference[key])
    assert calibration["pi"] == reference["pi"]
    assert calibration["mean_burst_length_reports"] == reference["mean_burst_length_reports"]
    assert len(files) == reference["n_files"]
    metadata = dict(
        dataset=dataset, created_utc=datetime.now(timezone.utc).isoformat(),
        radius_km=RADII[dataset], altitude_ceiling_ft_msl=7500, split_gap_gt_s=30,
        minimum_track_duration_s=60, stride_s=15, first_now_slot=59,
        n_files=len(files), n_candidates=n_candidates, n_skipped_latest_lt_59=n_skipped,
        stats=stats, calibration=calibration, calibration_exact_checks_passed=True,
        file_diagnostics=diagnostics, explanation=EXPLANATION,
        runtime_seconds=time.perf_counter()-started,
    )
    output = out / f"real_masks_{dataset}.npz"
    np.savez_compressed(
        output, mask=mask, age=age, track_id=np.concatenate(id_chunks),
        now_slot=np.concatenate(now_chunks), report_slot=np.concatenate(report_chunks),
        source_files=np.asarray([source_name(p, dataset) for p in files]),
        track_file_index=np.asarray(track_file, dtype=np.int32),
        track_aircraft_id=np.asarray(track_aircraft),
        track_start_epoch_s=np.asarray(track_start), track_end_epoch_s=np.asarray(track_end),
        metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)),
    )
    with np.load(output, allow_pickle=False) as actual, np.load(RESULTS / output.name, allow_pickle=False) as expected:
        assert actual.files == expected.files
        for key in actual.files:
            if key == "metadata_json":
                x, y = (json.loads(str(z[key].item())) for z in (actual, expected))
                for transient in ("created_utc", "runtime_seconds"):
                    x.pop(transient); y.pop(transient)
                assert x == y, (dataset, key)
            else:
                assert np.array_equal(actual[key], expected[key]), (dataset, key)
    print(f"{dataset}: all pool arrays identical; metadata identical except creation time/runtime.", flush=True)
    print(dataset, json.dumps(stats, allow_nan=False), flush=True)
    print(f"Saved {output}; exact calibration checks passed; "
          f"runtime={time.perf_counter()-started:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    raw_arguments(parser)
    parser.add_argument("--out", type=Path, default=RESULTS / "recomputed", help="output directory for both NPZ pools")
    parser.add_argument("--calibration", type=Path, default=RESULTS / "link_calibration.json")
    args = parser.parse_args()
    if args.out.resolve() == RESULTS.resolve():
        parser.error("--out must use a separate directory to preserve the published mask pools")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for dataset in DATASETS:
        build_pool(dataset, args.raw_trajair if dataset == "trajair" else args.raw_tartan,
                   args.out, args.calibration)
    print(EXPLANATION, flush=True)
    print(f"DONE pools; runtime={time.perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
