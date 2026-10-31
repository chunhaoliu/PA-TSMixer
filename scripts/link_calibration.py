"""Measure ADS-B report loss and staleness in raw receiver-table snapshots.

Sources: aircraft-main/data/trajair/111_days/111_days/raw_data/*_adsb/*.csv
and aircraft-main/data/TartanAviation_raw/*/*.csv. Repeated (ID, Date + Time)
reports are deduplicated AFTER the KBTP distance/altitude filter. Each file/ID
is independent; gaps strictly greater than split_s split tracks, and only tracks
spanning at least 60 seconds contribute to report/slot/window statistics.

Slots are [t0+i, t0+i+1), including i=floor(last_report-t0). Full 60-second
windows start every 15 slots and end no later than the last report timestamp;
their right endpoint is excluded when finding the last report for end age.
Logged Age and Range sanity statistics use all valid in-region snapshot rows,
including rows belonging to tracks subsequently excluded by the duration rule.
Exact pooled percentiles use in-memory arrays; only the requested outputs are written.

These are empirical gaps in received position reports, not independently
identified RF packet losses: source reporting cadence, receiver/table behavior,
region exits and track censoring can all affect the estimates. No tracks or
bursts are joined across files, IDs, region gaps exceeding split_s, or tracks.
"""

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/link"
AIRPORT_LAT = 40.7769
AIRPORT_LON = -79.9497
FIELD_ELEVATION_FT = 1248.0  # Metadata only; the altitude ceiling is MSL.
ALTITUDE_CEILING_FT = 7500.0
EARTH_RADIUS_KM = 6371.0088
MIN_TRACK_S = 60.0
WINDOW_S = 60
STRIDE_S = 15
SENSITIVITY_SPLITS = (15.0, 30.0, 60.0)
GRID_PI = (0.1, 0.2, 0.3, 0.4, 0.5)
GRID_L = (1, 5, 10)
AGE_THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
WINDOW_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.5)
BURST_LABELS = ("1", "2", "3", "4", "5", "6-10", "11-20", ">20")
QUOTED_TRIPLE = re.compile(
    r"""^\s*\[\s*[uU]?['"](\d+)['"]\s*,\s*"""
    r"""[uU]?['"](\d+)['"]\s*,\s*[uU]?['"](\d+(?:\.\d+)?)['"]\s*\]\s*$"""
)
CLOCK_TRIPLE = re.compile(r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")


def threshold_key(value):
    return f"gt_{value:g}"


class Distribution:
    """Collect finite float64 chunks; calculate exact pooled percentiles."""

    def __init__(self, percentiles, thresholds=(), include_max=False):
        self.chunks = []
        self.percentiles = percentiles
        self.thresholds = thresholds
        self.include_max = include_max
        self.n = 0
        self.exceed = np.zeros(len(thresholds), dtype=np.int64)
        self.maximum = None

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        assert np.isfinite(values).all()
        if not values.size:
            return
        self.chunks.append(values.copy())
        self.n += int(values.size)
        for index, threshold in enumerate(self.thresholds):
            self.exceed[index] += np.count_nonzero(values > threshold)
        maximum = float(values.max())
        self.maximum = maximum if self.maximum is None else max(self.maximum, maximum)

    def summarize(self):
        percentiles = {f"p{q:g}": None for q in self.percentiles}
        if self.n:
            values = np.concatenate(self.chunks)
            self.chunks.clear()
            quantiles = np.percentile(values, self.percentiles, overwrite_input=True)
            percentiles = {
                f"p{q:g}": float(value)
                for q, value in zip(self.percentiles, quantiles)
            }
        result = {"n": self.n, "percentiles": percentiles}
        if self.include_max:
            result["max"] = self.maximum
        if self.thresholds:
            result["fractions"] = {
                threshold_key(threshold): float(count / self.n) if self.n else None
                for threshold, count in zip(self.thresholds, self.exceed)
            }
        return result


class Distributions:
    """Own pooled arrays without creating scratch files outside the allowed outputs."""

    def __init__(self):
        self.distributions = {}

    def __enter__(self):
        return self

    def make(self, name, percentiles, thresholds=(), include_max=False):
        distribution = Distribution(percentiles, thresholds, include_max)
        self.distributions[name] = distribution
        return distribution

    def __exit__(self, exc_type, exc_value, traceback):
        self.distributions.clear()


def parse_timestamps(frame, dataset):
    """Return epoch seconds (float64) and a validity mask, without row loops."""
    if dataset == "tartanaviation":
        clock = frame["Time"].str.extract(QUOTED_TRIPLE)
        date_parts = frame["Date"].str.extract(QUOTED_TRIPLE)
        date_string = date_parts[0] + "-" + date_parts[1] + "-" + date_parts[2]
        dates = pd.to_datetime(date_string, format="%Y-%m-%d", errors="coerce")
    else:
        clock = frame["Time"].str.extract(CLOCK_TRIPLE)
        dates = pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="coerce")
    components = clock.apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    hour, minute, second = components.T
    valid = (
        dates.notna().to_numpy()
        & np.isfinite(components).all(axis=1)
        & (hour >= 0) & (hour < 24)
        & (minute >= 0) & (minute < 60)
        & (second >= 0) & (second < 60)
    )
    timestamps = np.full(len(frame), np.nan, dtype=np.float64)
    # Use explicit nanosecond resolution before converting to float epoch seconds.
    day_seconds = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e9
    timestamps[valid] = day_seconds[valid] + (
        hour[valid] * 3600.0 + minute[valid] * 60.0 + second[valid]
    )
    return timestamps, valid


def great_circle_km(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    reference_lat = math.radians(AIRPORT_LAT)
    reference_lon = math.radians(AIRPORT_LON)
    haversine = (
        np.sin((lat - reference_lat) / 2.0) ** 2
        + np.cos(lat) * math.cos(reference_lat)
        * np.sin((lon - reference_lon) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def read_file(path, dataset, radius_km, logged_age, range_ratios):
    """Read only needed columns (+ optional Range for the requested sanity check)."""
    needed = ["ID", "Time", "Date", "Altitude", "Lat", "Lon", "Age"]
    try:
        header = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError:
        print(f"[{dataset}] Empty source file: {path}", flush=True)
        return pd.DataFrame({"ID": pd.Series(dtype="string"),
                             "timestamp": pd.Series(dtype=np.float64)}), {
            "file": str(path), "empty_file": True, "n_logged_rows": 0,
            "n_skipped_malformed_rows": 0, "parser_recovery": None,
            "n_invalid_timestamp_rows": 0, "n_in_region_logged_rows": 0,
            "n_invalid_in_region_age_rows": 0,
            "n_unique_in_region_reports_before_track_filter": 0,
            "range_column_present": False, "n_range_ratios": 0,
            "median_range_to_distance_km_ratio": None,
        }
    missing = sorted(set(needed) - set(header))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    columns = needed + (["Range"] if "Range" in header else [])
    read_options = {
        "usecols": columns,
        "dtype": {"ID": "string", "Time": "string", "Date": "string"},
    }
    parser_recovery = None
    n_skipped_malformed_rows = 0
    try:
        frame = pd.read_csv(path, low_memory=False, **read_options)
    except pd.errors.ParserError as error:
        # Some source CSVs end inside a quoted field. Preserve readable records;
        # Python's parser can skip the incomplete record without editing inputs.
        if "EOF inside string" not in str(error):
            raise
        frame = pd.read_csv(path, engine="python", on_bad_lines="skip", **read_options)
        with path.open(encoding="utf-8") as handle:
            n_source_rows = sum(1 for _ in handle) - 1
        n_skipped_malformed_rows = n_source_rows - len(frame)
        assert n_skipped_malformed_rows >= 1
        parser_recovery = str(error)
        print(f"[{dataset}] Recovered {path}: skipped "
              f"{n_skipped_malformed_rows} malformed row(s)", flush=True)
    timestamps, valid_time = parse_timestamps(frame, dataset)
    numeric = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        for name in columns if name not in ("ID", "Time", "Date")
    }
    lat, lon, altitude = (numeric[name] for name in ("Lat", "Lon", "Altitude"))
    ids = frame["ID"].str.strip()
    valid = (
        valid_time & ids.notna().to_numpy() & ids.ne("").fillna(False).to_numpy(dtype=bool)
        & np.isfinite(lat) & np.isfinite(lon) & np.isfinite(altitude)
        & (lat != 0) & (lon != 0) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    )
    distances = great_circle_km(lat, lon)
    in_region = valid & (distances <= radius_km) & (altitude <= ALTITUDE_CEILING_FT)
    ages = numeric["Age"][in_region]
    valid_age = np.isfinite(ages) & (ages >= 0)
    logged_age.add(ages[valid_age])
    ratio_median = None
    ratio_count = 0
    if "Range" in numeric:
        ranges = numeric["Range"]
        ratio_mask = in_region & np.isfinite(ranges) & (ranges > 0) & (distances > 0)
        ratios = ranges[ratio_mask] / distances[ratio_mask]
        ratios = ratios[np.isfinite(ratios)]
        range_ratios.add(ratios)
        ratio_count = int(ratios.size)
        if ratio_count:
            ratio_median = float(np.median(ratios))
    reports = pd.DataFrame({"ID": ids[in_region], "timestamp": timestamps[in_region]})
    reports = reports.drop_duplicates(["ID", "timestamp"]).sort_values(["ID", "timestamp"])
    diagnostics = {
        "file": str(path),
        "empty_file": False,
        "n_logged_rows": len(frame),
        "n_skipped_malformed_rows": n_skipped_malformed_rows,
        "parser_recovery": parser_recovery,
        "n_invalid_timestamp_rows": int((~valid_time).sum()),
        "n_in_region_logged_rows": int(in_region.sum()),
        "n_invalid_in_region_age_rows": int((~valid_age).sum()),
        "n_unique_in_region_reports_before_track_filter": len(reports),
        "range_column_present": "Range" in numeric,
        "n_range_ratios": ratio_count,
        "median_range_to_distance_km_ratio": ratio_median,
    }
    return reports, diagnostics


def iter_tracks(times, split_s):
    intervals = np.diff(times)
    assert (intervals >= 0).all(), "Negative inter-report interval after sorting"
    cuts = np.flatnonzero(intervals > split_s) + 1
    for track in np.split(times, cuts):
        if track.size >= 2 and track[-1] - track[0] >= MIN_TRACK_S:
            yield track


def track_slots(track):
    relative = track - track[0]
    received_indices = np.unique(np.floor(relative).astype(np.int64))
    n_slots = int(math.floor(float(relative[-1]))) + 1
    gaps = np.diff(received_indices) - 1
    bursts = gaps[gaps > 0]
    received = int(received_indices.size)
    lost = n_slots - received
    assert n_slots == received + lost
    assert int(bursts.sum()) == lost, "Burst lengths do not sum to lost slots"
    assert 0.0 <= lost / n_slots <= 1.0
    return relative, received_indices, n_slots, lost, bursts


class PooledSlots:
    def __init__(self):
        self.n_tracks = 0
        self.n_reports = 0
        self.track_seconds = 0.0
        self.slots = 0
        self.received = 0
        self.lost = 0
        self.n_bursts = 0
        self.histogram = np.zeros(len(BURST_LABELS), dtype=np.int64)

    def add(self, track, slot_info):
        _, received_indices, n_slots, lost, bursts = slot_info
        self.n_tracks += 1
        self.n_reports += int(track.size)
        self.track_seconds += float(track[-1] - track[0])
        self.slots += n_slots
        self.received += int(received_indices.size)
        self.lost += lost
        self.n_bursts += int(bursts.size)
        # Right-closed bins: 1,2,3,4,5,6..10,11..20,>20.
        bins = np.searchsorted([1, 2, 3, 4, 5, 10, 20], bursts, side="left")
        self.histogram += np.bincount(bins, minlength=len(BURST_LABELS))

    def summarize(self):
        assert self.slots == self.received + self.lost
        assert int(self.histogram.sum()) == self.n_bursts
        pi = self.lost / self.slots if self.slots else None
        mean_burst = self.lost / self.n_bursts if self.n_bursts else None
        assert pi is None or 0.0 <= pi <= 1.0
        if mean_burst is not None:
            p_bg = 1.0 / mean_burst
            p_gb = pi / (mean_burst * (1.0 - pi)) if pi < 1 else None
        else:
            p_bg = None  # Unidentifiable without any observed loss runs.
            p_gb = 0.0 if pi == 0 else None
        assert p_bg is None or 0.0 <= p_bg <= 1.0
        assert p_gb is None or 0.0 <= p_gb <= 1.0
        return {
            "n_tracks": self.n_tracks,
            "total_track_hours": self.track_seconds / 3600.0,
            "n_reports": self.n_reports,
            "n_slots": self.slots,
            "n_received_slots": self.received,
            "n_lost_slots": self.lost,
            "pi": pi,
            "n_loss_bursts": self.n_bursts,
            "mean_burst_length_reports": mean_burst,
            "burst_length_histogram": {
                label: int(count) for label, count in zip(BURST_LABELS, self.histogram)
            },
            "ge_fit": {"p_bg": p_bg, "p_gb": p_gb},
        }


def add_windows(slot_info, loss_distribution, burst_distribution, end_age_distribution):
    relative, received_indices, n_slots, _, _ = slot_info
    last_start = math.floor(float(relative[-1]) - WINDOW_S)
    starts = np.arange(0, last_start + 1, STRIDE_S, dtype=np.int64)
    if not starts.size:
        return 0
    lost = np.ones(n_slots, dtype=bool)
    lost[received_indices] = False
    cumulative = np.concatenate(([0], np.cumsum(lost, dtype=np.int64)))
    loss_distribution.add((cumulative[starts + WINDOW_S] - cumulative[starts]) / WINDOW_S)
    # Clip the running loss streak to the window's left boundary.
    indices = np.arange(n_slots, dtype=np.int64)
    last_received = np.maximum.accumulate(np.where(lost, -1, indices))
    streak = indices - last_received
    streak_windows = np.lib.stride_tricks.sliding_window_view(streak, WINDOW_S)[starts]
    longest = np.minimum(streak_windows, np.arange(1, WINDOW_S + 1)).max(axis=1)
    burst_distribution.add(longest)
    ends = starts + WINDOW_S
    last_reports = np.searchsorted(relative, ends, side="left") - 1
    has_report = (last_reports >= 0) & (relative[np.maximum(last_reports, 0)] >= starts)
    # Empty windows are possible with user-selected split thresholds above 60 s.
    # Their within-window age is undefined; count them explicitly, without imputation.
    ages = ends[has_report] - relative[last_reports[has_report]]
    assert (ages >= 0).all()
    end_age_distribution.add(ages)
    return int((~has_report).sum())


def grid_coverage(window_loss):
    result = {"n_tested_pi_L_cells": len(GRID_PI) * len(GRID_L)}
    for percentile in ("p90", "p99"):
        cutoff = window_loss["percentiles"][percentile]
        count = sum(pi <= cutoff for pi in GRID_PI) * len(GRID_L) if cutoff is not None else None
        result[percentile] = {
            "window_loss_cutoff": cutoff,
            "n_cells_with_pi_le_cutoff": count,
            "fraction_cells_with_pi_le_cutoff": count / result["n_tested_pi_L_cells"]
            if count is not None else None,
        }
    result["interpretation"] = (
        "Counts all 15 (pi,L) cells by pi only; it does not establish coverage of "
        "the measured burst-length distribution or of joint loss/staleness behavior."
    )
    return result


def source_name(path, dataset):
    """Portable source provenance; raw logs themselves are not redistributed."""
    prefix = ("data/raw/trajair/111_days/111_days/raw_data" if dataset == "trajair"
              else "data/raw/TartanAviation_raw")
    return f"{prefix}/{path.parent.name}/{path.name}"


def discover_files(raw_root, dataset, max_files):
    pattern = "*_adsb/*.csv" if dataset == "trajair" else "*/*.csv"
    files = sorted(raw_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {dataset} CSV files found under {raw_root}")
    return files[:max_files] if max_files is not None else files


def raw_arguments(parser):
    parser.add_argument("--raw-trajair", type=Path,
                        default=ROOT / "data/raw/trajair/111_days/111_days/raw_data")
    parser.add_argument("--raw-tartan", type=Path, default=ROOT / "data/raw/TartanAviation_raw")


def analyze_dataset(dataset, files, radius_km, split_s):
    main_pool = PooledSlots()
    sensitivity = {value: PooledSlots() for value in SENSITIVITY_SPLITS if value != split_s}
    file_diagnostics = []
    n_empty_windows = 0
    with Distributions() as scratch:
        intervals = scratch.make("intervals", (50, 90, 99, 99.9), (1.5, 2.5, 5.5), True)
        logged_age = scratch.make("logged_age", (50, 90, 99), AGE_THRESHOLDS)
        end_age = scratch.make("window_end_age", (50, 90, 99), AGE_THRESHOLDS)
        window_loss = scratch.make("window_loss", (50, 90, 99), WINDOW_THRESHOLDS)
        window_burst = scratch.make("window_burst", (50, 90, 99))
        track_loss = scratch.make("track_loss", (10, 50, 90))
        range_ratios = scratch.make("range_ratios", (50,))
        for file_index, path in enumerate(files, 1):
            reports, diagnostics = read_file(path, dataset, radius_km, logged_age, range_ratios)
            diagnostics["file"] = source_name(path, dataset)
            file_diagnostics.append(diagnostics)
            for _, group in reports.groupby("ID", sort=False):
                times = group["timestamp"].to_numpy(dtype=np.float64)
                for track in iter_tracks(times, split_s):
                    slot_info = track_slots(track)
                    main_pool.add(track, slot_info)
                    intervals.add(np.diff(track))
                    track_loss.add([slot_info[3] / slot_info[2]])
                    n_empty_windows += add_windows(slot_info, window_loss, window_burst, end_age)
                for threshold, pool in sensitivity.items():
                    for track in iter_tracks(times, threshold):
                        pool.add(track, track_slots(track))
            # Parsed reports are reused only for this file's sensitivity calculations.
            del reports
            if file_index % 20 == 0:
                pi = main_pool.lost / main_pool.slots if main_pool.slots else 0.0
                print(
                    f"[{dataset}] {file_index}/{len(files)} files; "
                    f"{main_pool.n_tracks:,} tracks; pooled pi={pi:.4f}", flush=True
                )
        print(f"[{dataset}] Computing exact pooled percentiles...", flush=True)
        result = main_pool.summarize()
        result.update({
            "n_files": len(files),
            "radius_km": radius_km,
            "split_s": split_s,
            "inter_report_interval_s": intervals.summarize(),
            "window_loss_fraction": window_loss.summarize(),
            "window_longest_burst_reports": window_burst.summarize(),
            "logged_row_age_s": logged_age.summarize(),
            "window_end_age_s": end_age.summarize(),
            "n_windows_without_reports": n_empty_windows,
            "per_track_loss_rate": track_loss.summarize(),
        })
        ratios = range_ratios.summarize()
        file_medians = [
            item["median_range_to_distance_km_ratio"] for item in file_diagnostics
            if item["median_range_to_distance_km_ratio"] is not None
        ]
        result["range_sanity"] = {
            "n_ratios": ratios["n"],
            "pooled_median_range_to_distance_km_ratio": ratios["percentiles"]["p50"],
            "median_of_file_median_ratios": float(np.median(file_medians)) if file_medians else None,
            "unit_reference_ratios": {"km": 1.0, "nautical_miles": 1.0 / 1.852,
                                      "statute_miles": 1.0 / 1.609344},
            "note": "Range / computed distance in km; positive finite values only; diagnostic only.",
        }
        result["split_sensitivity"] = {
            f"{threshold:g}": main_pool.summarize() if threshold == split_s
            else sensitivity[threshold].summarize()
            for threshold in SENSITIVITY_SPLITS
        }
        result["grid_fairness"] = grid_coverage(result["window_loss_fraction"])
        result["files"] = file_diagnostics
        result["n_empty_files"] = sum(item["empty_file"] for item in file_diagnostics)
        result["n_skipped_malformed_rows"] = sum(
            item["n_skipped_malformed_rows"] for item in file_diagnostics
        )
        result["n_in_region_logged_rows"] = sum(
            item["n_in_region_logged_rows"] for item in file_diagnostics
        )
        result["n_unique_in_region_reports_before_track_filter"] = sum(
            item["n_unique_in_region_reports_before_track_filter"] for item in file_diagnostics
        )
        assert result["window_loss_fraction"]["n"] == result["window_end_age_s"]["n"] + n_empty_windows
        assert result["window_loss_fraction"]["n"] == result["window_longest_burst_reports"]["n"]
        assert result["inter_report_interval_s"]["n"] == result["n_reports"] - result["n_tracks"]
        return result


def fmt(value, percent=False):
    if value is None:
        return "n/a"
    return f"{value:.2%}" if percent else f"{value:,.4g}"


def percentile_text(distribution, keys=("p50", "p90", "p99"), percent=False):
    return " / ".join(fmt(distribution["percentiles"][key], percent) for key in keys)


def fraction_text(distribution, thresholds):
    return "; ".join(
        f">{threshold:g}: {fmt(distribution['fractions'][threshold_key(threshold)], True)}"
        for threshold in thresholds
    )


def make_markdown(document):
    lines = [
        "# Raw ADS-B link calibration", "",
        "Report/slot metrics use unique in-region reports in tracks spanning >=60 s. "
        "Logged-row Age includes repeated snapshots and short tracks. "
        "All loss rates below are fractions of 1-Hz slots, not packet-level loss estimates.", "",
        f"Files are selected in sorted path order; --max-files applies separately to each dataset "
        f"(limit: {document['parameters']['max_files_per_dataset']}).", "",
    ]
    for name, result in document["datasets"].items():
        rows = [
            ("Region / split", f"{result['radius_km']:g} km / {result['split_s']:g} s"),
            ("Files / retained tracks / unique reports",
             f"{result['n_files']:,} / {result['n_tracks']:,} / {result['n_reports']:,}"),
            ("Empty source files (included in file count)", f"{result['n_empty_files']:,}"),
            ("Malformed source rows skipped", f"{result['n_skipped_malformed_rows']:,}"),
            ("Track hours", fmt(result["total_track_hours"])),
            ("Slots: all / received / lost",
             f"{result['n_slots']:,} / {result['n_received_slots']:,} / {result['n_lost_slots']:,}"),
            ("Pooled pi / mean burst L (reports)",
             f"{fmt(result['pi'], True)} / {fmt(result['mean_burst_length_reports'])}"),
            ("GE p_bg / p_gb", f"{fmt(result['ge_fit']['p_bg'])} / {fmt(result['ge_fit']['p_gb'])}"),
            ("Burst counts (length: count)", "; ".join(
                f"{key}: {value:,}" for key, value in result["burst_length_histogram"].items())),
            ("Inter-report s: p50 / p90 / p99 / p99.9 / max",
             percentile_text(result["inter_report_interval_s"], ("p50", "p90", "p99", "p99.9"))
             + " / " + fmt(result["inter_report_interval_s"]["max"])),
            ("Inter-report interval fractions", fraction_text(result["inter_report_interval_s"], (1.5, 2.5, 5.5))),
            ("60-s windows", f"{result['window_loss_fraction']['n']:,}"),
            ("Windows without reports (excluded from window-end age)",
             f"{result['n_windows_without_reports']:,}"),
            ("Window loss: p50 / p90 / p99", percentile_text(result["window_loss_fraction"], percent=True)),
            ("Window loss fractions", fraction_text(result["window_loss_fraction"], WINDOW_THRESHOLDS)),
            ("Window longest burst: p50 / p90 / p99", percentile_text(result["window_longest_burst_reports"])),
            ("Logged Age s: p50 / p90 / p99", percentile_text(result["logged_row_age_s"])),
            ("Logged Age fractions", fraction_text(result["logged_row_age_s"], AGE_THRESHOLDS)),
            ("Window-end age s: p50 / p90 / p99", percentile_text(result["window_end_age_s"])),
            ("Window-end age fractions", fraction_text(result["window_end_age_s"], AGE_THRESHOLDS)),
            ("Track pi: p10 / p50 / p90", percentile_text(result["per_track_loss_rate"], ("p10", "p50", "p90"), True)),
            ("Range / distance-km: pooled median / median of file medians",
             fmt(result["range_sanity"]["pooled_median_range_to_distance_km_ratio"])
             + " / " + fmt(result["range_sanity"]["median_of_file_median_ratios"])),
        ]
        for threshold, sensitivity in result["split_sensitivity"].items():
            rows.append((f"Split {threshold} s: pi / L / retained tracks",
                         f"{fmt(sensitivity['pi'], True)} / "
                         f"{fmt(sensitivity['mean_burst_length_reports'])} / {sensitivity['n_tracks']:,}"))
        for percentile in ("p90", "p99"):
            coverage = result["grid_fairness"][percentile]
            rows.append((f"Tested (pi,L) cells with pi <= window-loss {percentile}",
                         f"{coverage['n_cells_with_pi_le_cutoff']} / 15 "
                         f"({fmt(coverage['fraction_cells_with_pi_le_cutoff'], True)})"))
        lines.extend([f"## {name}", "", "| Metric | Value |", "|---|---|"])
        lines.extend(f"| {label} | {value} |" for label, value in rows)
        lines.append("")
    lines.extend([
        "Range units can be checked against ratios 1 (km), 0.5400 (nautical miles), "
        "or 0.6214 (statute miles). Per-file medians and row counts are in the JSON.", "",
        "Full windows are [start, start+60), with stride 15 s, contained within the "
        "first-to-last-report span. Window-end age excludes reports exactly at the "
        "right endpoint. The final track slot contains the last report. Track hours "
        "sum first-to-last-report durations, not slot counts. Windows without reports "
        "(possible with split >60 s) have undefined within-window age and are counted "
        "separately rather than included in the age distribution.", "",
        "Gaps above the split threshold, short tracks, region filtering and file "
        "boundaries censor the observations. Source cadence and logger behavior "
        "also contribute to apparent loss. Grid coverage compares pi only; it does "
        "not validate the grid's burst lengths or joint staleness coverage. "
        "No-loss L and p_bg are n/a because no loss runs were observed. "
        "Missing/nonfinite/negative logged Age is excluded from age statistics. "
        "Files ending inside a quoted field are recovered with the Python CSV parser; "
        "skipped malformed row counts and recovery details are recorded per file. "
        "Empty source files contribute zero observations and remain in the file count.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("trajair", "tartanaviation", "both"), default="both")
    raw_arguments(parser)
    parser.add_argument("--max-files", type=int, help="Read the first N sorted files per dataset.")
    parser.add_argument("--radius-km", type=float, help="Override both dataset-specific radii.")
    parser.add_argument("--split-s", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=RESULTS / "recomputed/link_calibration.json")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")
    if not math.isfinite(args.split_s) or args.split_s <= 0:
        parser.error("--split-s must be positive and finite")
    if args.radius_km is not None and (not math.isfinite(args.radius_km) or args.radius_km <= 0):
        parser.error("--radius-km must be positive and finite")
    output = args.out.resolve()
    if output.parent == RESULTS.resolve():
        parser.error("--out must use a separate directory to preserve the published result files")
    markdown = output.with_name("link_calibration.md")
    if output == markdown or output == Path(__file__).resolve():
        parser.error("--out must differ from the script path and link_calibration.md")
    datasets = ("trajair", "tartanaviation") if args.dataset == "both" else (args.dataset,)
    # Resolve all inputs before creating scratch/output files; fail on missing data.
    files = {name: discover_files(args.raw_trajair if name == "trajair" else args.raw_tartan,
                                 name, args.max_files) for name in datasets}
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "parameters": {
            "data_root": "data/raw",  # Canonical source labels, independent of local input/output locations.
            "dataset": args.dataset,
            "max_files_per_dataset": args.max_files,
            "radius_km": {name: args.radius_km if args.radius_km is not None else
                          (6.0 if name == "trajair" else 11.0) for name in datasets},
            "split_s": args.split_s,
            "sensitivity_split_s": list(SENSITIVITY_SPLITS),
            "minimum_track_duration_s": MIN_TRACK_S,
            "slot_s": 1,
            "window_s": WINDOW_S,
            "window_stride_s": STRIDE_S,
            "airport": {"name": "KBTP", "lat": AIRPORT_LAT, "lon": AIRPORT_LON,
                        "field_elevation_ft": FIELD_ELEVATION_FT},
            "altitude_ceiling_ft_msl": ALTITUDE_CEILING_FT,
            "earth_radius_km": EARTH_RADIUS_KM,
            "tested_grid": {"pi": list(GRID_PI), "L_reports_at_1_hz": list(GRID_L),
                            "staleness_s": list(range(6))},
            "percentile_method": "Exact NumPy linear percentiles, pooled observations",
            "window_definition": "[t0+15*j, t0+15*j+60), end <= last report timestamp",
            "empty_window_age_policy": "Undefined; exclude from age distribution and count separately",
            "logged_age_policy": "All valid in-region snapshot rows; exclude missing/nonfinite/negative Age",
            "report_count_policy": "Unique in-region reports in retained tracks only",
            "no_loss_policy": "L and p_bg null; p_gb=0 when slots exist but no loss occurs",
            "file_boundary_policy": "Never join tracks across files, even for the same ID/date",
            "malformed_csv_policy": "On EOF inside string, use Python parser and count skipped malformed rows",
            "empty_file_policy": "Include in source file count, contribute zero observations",
            "percentile_storage": "In-memory arrays; no scratch files",
            "out": "results/link/link_calibration.json",
        },
        "datasets": {},
    }
    for name in datasets:
        print(f"[{name}] Processing {len(files[name]):,} files...", flush=True)
        document["datasets"][name] = analyze_dataset(
            name, files[name], document["parameters"]["radius_km"][name], args.split_s
        )
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown.write_text(make_markdown(document), encoding="utf-8")
    if args.dataset == "both" and args.max_files is None and args.radius_km is None and args.split_s == 30:
        expected = json.loads((RESULTS / "link_calibration.json").read_text(encoding="utf-8"))
        assert document == expected, "calibration differs from published result"
        print("Calibration JSON identical (including portable source provenance).", flush=True)
    print("Calibration complete:", flush=True)
    for name, result in document["datasets"].items():
        print(
            f"  {name}: {result['n_files']:,} files, {result['n_tracks']:,} tracks, "
            f"{result['total_track_hours']:.2f} h, {result['n_reports']:,} reports; "
            f"pi={fmt(result['pi'], True)}, L={fmt(result['mean_burst_length_reports'])}; "
            f"window-loss p90/p99="
            f"{percentile_text(result['window_loss_fraction'], ('p90', 'p99'), True)}",
            flush=True,
        )
    print(f"JSON: {output}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
