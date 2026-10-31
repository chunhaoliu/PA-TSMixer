# Raw ADS-B link calibration

Report/slot metrics use unique in-region reports in tracks spanning >=60 s. Logged-row Age includes repeated snapshots and short tracks. All loss rates below are fractions of 1-Hz slots, not packet-level loss estimates.

Files are selected in sorted path order; --max-files applies separately to each dataset (limit: None).

## trajair

| Metric | Value |
|---|---|
| Region / split | 6 km / 30 s |
| Files / retained tracks / unique reports | 111 / 7,489 / 2,197,610 |
| Empty source files (included in file count) | 2 |
| Malformed source rows skipped | 0 |
| Track hours | 731.2 |
| Slots: all / received / lost | 2,636,112 / 2,068,623 / 567,489 |
| Pooled pi / mean burst L (reports) | 21.53% / 1.309 |
| GE p_bg / p_gb | 0.7639 / 0.2096 |
| Burst counts (length: count) | 1: 379,677; 2: 33,812; 3: 7,535; 4: 3,522; 5: 2,149; 6-10: 4,314; 11-20: 1,974; >20: 515 |
| Inter-report s: p50 / p90 / p99 / p99.9 / max | 1.068 / 1.614 / 3.289 / 12.22 / 29.94 |
| Inter-report interval fractions | >1.5: 13.01%; >2.5: 2.02%; >5.5: 0.41% |
| 60-s windows | 149,267 |
| Windows without reports (excluded from window-end age) | 0 |
| Window loss: p50 / p90 / p99 | 16.67% / 38.33% / 65.00% |
| Window loss fractions | >0: 100.00%; >0.1: 73.41%; >0.2: 41.15%; >0.3: 21.95%; >0.5: 2.90% |
| Window longest burst: p50 / p90 / p99 | 1 / 6 / 19 |
| Logged Age s: p50 / p90 / p99 | 0.853 / 6.45 / 53.38 |
| Logged Age fractions | >1: 36.68%; >2: 14.30%; >3: 12.25%; >5: 10.65% |
| Window-end age s: p50 / p90 / p99 | 0.613 / 1.343 / 7.519 |
| Window-end age fractions | >1: 21.12%; >2: 4.72%; >3: 2.81%; >5: 1.68% |
| Track pi: p10 / p50 / p90 | 10.43% / 18.35% / 36.13% |
| Range / distance-km: pooled median / median of file medians | 1.007 / 1.007 |
| Split 15 s: pi / L / retained tracks | 20.58% / 1.243 / 8,055 |
| Split 30 s: pi / L / retained tracks | 21.53% / 1.309 / 7,489 |
| Split 60 s: pi / L / retained tracks | 22.31% / 1.366 / 7,259 |
| Tested (pi,L) cells with pi <= window-loss p90 | 9 / 15 (60.00%) |
| Tested (pi,L) cells with pi <= window-loss p99 | 15 / 15 (100.00%) |

## tartanaviation

| Metric | Value |
|---|---|
| Region / split | 11 km / 30 s |
| Files / retained tracks / unique reports | 246 / 18,989 / 6,049,471 |
| Empty source files (included in file count) | 19 |
| Malformed source rows skipped | 56 |
| Track hours | 2,108 |
| Slots: all / received / lost | 7,604,832 / 5,946,431 / 1,658,401 |
| Pooled pi / mean burst L (reports) | 21.81% / 1.241 |
| GE p_bg / p_gb | 0.8055 / 0.2247 |
| Burst counts (length: count) | 1: 1,194,841; 2: 104,616; 3: 15,084; 4: 4,861; 5: 2,826; 6-10: 6,425; 11-20: 5,684; >20: 1,574 |
| Inter-report s: p50 / p90 / p99 / p99.9 / max | 1 / 2 / 3 / 13.57 / 30 |
| Inter-report interval fractions | >1.5: 20.40%; >2.5: 2.20%; >5.5: 0.27% |
| 60-s windows | 439,951 |
| Windows without reports (excluded from window-end age) | 0 |
| Window loss: p50 / p90 / p99 | 18.33% / 36.67% / 68.33% |
| Window loss fractions | >0: 100.00%; >0.1: 86.27%; >0.2: 39.62%; >0.3: 19.18%; >0.5: 2.73% |
| Window longest burst: p50 / p90 / p99 | 1 / 3 / 19 |
| Logged Age s: p50 / p90 / p99 | 0.9302 / 5.383 / 52.73 |
| Logged Age fractions | >1: 45.16%; >2: 19.11%; >3: 12.54%; >5: 10.17% |
| Window-end age s: p50 / p90 / p99 | 1 / 2 / 8.936 |
| Window-end age fractions | >1: 21.80%; >2: 4.20%; >3: 2.33%; >5: 1.61% |
| Track pi: p10 / p50 / p90 | 12.11% / 19.07% / 36.46% |
| Range / distance-km: pooled median / median of file medians | 1.005 / 1.007 |
| Split 15 s: pi / L / retained tracks | 20.59% / 1.162 / 19,973 |
| Split 30 s: pi / L / retained tracks | 21.81% / 1.241 / 18,989 |
| Split 60 s: pi / L / retained tracks | 22.62% / 1.298 / 18,363 |
| Tested (pi,L) cells with pi <= window-loss p90 | 9 / 15 (60.00%) |
| Tested (pi,L) cells with pi <= window-loss p99 | 15 / 15 (100.00%) |

Range units can be checked against ratios 1 (km), 0.5400 (nautical miles), or 0.6214 (statute miles). Per-file medians and row counts are in the JSON.

Full windows are [start, start+60), with stride 15 s, contained within the first-to-last-report span. Window-end age excludes reports exactly at the right endpoint. The final track slot contains the last report. Track hours sum first-to-last-report durations, not slot counts. Windows without reports (possible with split >60 s) have undefined within-window age and are counted separately rather than included in the age distribution.

Gaps above the split threshold, short tracks, region filtering and file boundaries censor the observations. Source cadence and logger behavior also contribute to apparent loss. Grid coverage compares pi only; it does not validate the grid's burst lengths or joint staleness coverage. No-loss L and p_bg are n/a because no loss runs were observed. Missing/nonfinite/negative logged Age is excluded from age statistics. Files ending inside a quoted field are recovered with the Python CSV parser; skipped malformed row counts and recovery details are recorded per file. Empty source files contribute zero observations and remain in the file count.
