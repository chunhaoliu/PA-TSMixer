# Real ADS-B reception replay

Started UTC: 2026-10-31T12:06:22.888496+00:00. Evaluation runtime: 257.53 s (4.29 min).

Frozen test-split checkpoints; physical GPU 1 only; FP32; TF32 off. No packages installed, training, or changes to inputs. Learned cells show three-seed mean ± sample SD (42/123/456, ddof=1), in metres. CV/CA are deterministic (det.). Bold is the smallest unrounded mean per column; lower is better, including percentage changes. Changes are ratios of seed means.

Each dataset uses a fresh default_rng(2026) for RL and default_rng(2027) for RLA; uniform mask draws with replacement are shared by every method/seed. Assignments (pool indices), hashes and subset diagnostics are in real_trace.json. RL ignores sampled ages and scores H=30. RLA conditions the pool on k<=30, uses H=60 checkpoints, and scores each unchanged future slice [k:k+30], i.e. leads k+1 through k+30 after the latest report; FDE uses lead k+30. History ends at the latest report; missing positions use the exact np.interp treatment in link_impairment.py, including constant fill before the first keep.

Worst decile is selected separately per experiment from assigned loss counts over slots 0..58: ceil(N/10) highest counts, with ties broken by original test-window order. Subsets and inputs are identical for all methods/seeds. Nominal columns use all test windows; the two percentage columns compare all-window RL/RLA against the corresponding nominal column, respectively.

Strict reference checks: 104 passed with relative error <1e-5; maximum 2.197e-06. Checks cover clean H30 full ADE and clean H60 full-future ADE for every method/seed. H60 nominal [0:30] is measured separately. No assertions relaxed. Mask parsing reproduces the original full-file calibration exactly.

Pool interpretation: Calibration pools every retained track slot once; this pool samples now slots 59,74,89,... and shifts each history to its latest received slot j. Histories overlap, require j>=59, can repeat for different now slots, and measure loss only in slots 0..58. Thus track weighting, boundaries, and conditioning on a received endpoint differ. Bursts are pooled within-history lost runs, including left-boundary truncation. Integer slot age n-j differs from calibration's continuous, right-endpoint-exclusive window age and logged snapshot Age. These are empirical reception gaps, not identified RF packet losses.

## trajair

Pool: 148,168 masks from 111 files / 7,489 tracks; mean loss 20.6921%; mean burst 1.2621 s; any loss 100.0000%. Calibration pooled slots: 21.5275% loss, 1.3091 s mean burst. Age>30: 0.000000% (RLA excluded fraction).

Pool integer age percentiles (s): p0=0, p25=0, p50=0, p75=0, p90=1, p95=1, p99=7, p99.9=19, p100=28. Fractions: ge_1=20.2169%, ge_2=4.5111%, ge_3=2.6794%, ge_5=1.6016%.

RL assignments — all: 21,346, worst_decile: 2,135; loss 20.5650%, burst 1.2600 s; worst-decile cutoff 22/59 lost slots; excluded pool 0.000000%.

RLA assignments — all: 21,346, worst_decile: 2,135, k_ge_3: 574; loss 20.5342%, burst 1.2624 s; worst-decile cutoff 22/59 lost slots; excluded pool 0.000000%.

### ADE (m; percentage columns in %)

| Method | Nominal H30 | RL | RL change (%) | RL worst 10% | H60 nominal [0:30] | RLA | RLA change (%) | RLA worst 10% | RLA k>=3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV | 175.45 (det.) | 175.69 (det.) | +0.14 | 183.42 (det.) | 175.45 (det.) | 181.56 (det.) | +3.48 | 203.71 (det.) | 302.83 (det.) |
| CA | 218.45 (det.) | 217.99 (det.) | -0.21 | 227.94 (det.) | 218.45 (det.) | 226.62 (det.) | +3.74 | 251.58 (det.) | 388.84 (det.) |
| CNN-GRU | 121.46 ± 3.00 | 121.42 ± 3.00 | -0.03 | 123.93 ± 2.95 | 122.75 ± 1.24 | 124.68 ± 1.22 | **+1.57** | 133.19 ± 1.83 | 173.36 ± 0.97 |
| FlightBERT++ | 105.56 ± 0.46 | 104.62 ± 0.64 | **-0.89** | 106.57 ± 0.74 | 109.79 ± 0.59 | 111.81 ± 0.63 | +1.84 | 121.90 ± 0.38 | 177.26 ± 5.39 |
| DLinear | 179.90 ± 0.10 | 180.14 ± 0.09 | +0.13 | 184.32 ± 0.08 | 180.11 ± 0.70 | 185.85 ± 0.73 | +3.18 | 205.07 ± 0.98 | 289.21 ± 1.13 |
| Transformer | 126.21 ± 1.70 | 126.20 ± 1.70 | -0.01 | 130.18 ± 1.20 | 141.80 ± 1.68 | 144.28 ± 1.69 | +1.75 | 151.15 ± 1.75 | 197.35 ± 2.05 |
| PatchTST | 179.94 ± 9.90 | 179.91 ± 9.89 | -0.02 | 184.72 ± 9.05 | 178.25 ± 6.94 | 182.56 ± 6.88 | +2.42 | 200.40 ± 6.05 | 264.65 ± 4.66 |
| TimeMixer | 142.16 ± 5.02 | 142.25 ± 4.86 | +0.07 | 148.35 ± 4.27 | 147.14 ± 2.76 | 152.17 ± 2.74 | +3.42 | 171.41 ± 3.23 | 249.70 ± 3.91 |
| TSMixer | 120.34 ± 2.89 | 120.47 ± 2.87 | +0.11 | 124.54 ± 2.96 | 137.23 ± 0.68 | 139.78 ± 0.64 | +1.86 | 147.93 ± 0.36 | 192.91 ± 1.20 |
| PA-TSMixer | **100.23 ± 0.47** | **100.37 ± 0.48** | +0.15 | **104.95 ± 0.40** | **103.59 ± 1.31** | **106.58 ± 1.36** | +2.89 | **115.13 ± 1.53** | **167.39 ± 2.18** |

- Nominal H30: PA-TSMixer 100.2279, rank 1/10; best PA-TSMixer (100.2279); best competitor FlightBERT++ (105.5594).
- RL: PA-TSMixer 100.3744, rank 1/10; best PA-TSMixer (100.3744); best competitor FlightBERT++ (104.6217).
- RL change (%): PA-TSMixer 0.1462, rank 10/10; best FlightBERT++ (-0.8883); best competitor FlightBERT++ (-0.8883).
- RL worst 10%: PA-TSMixer 104.9461, rank 1/10; best PA-TSMixer (104.9461); best competitor FlightBERT++ (106.5701).
- H60 nominal [0:30]: PA-TSMixer 103.5879, rank 1/10; best PA-TSMixer (103.5879); best competitor FlightBERT++ (109.7936).
- RLA: PA-TSMixer 106.5844, rank 1/10; best PA-TSMixer (106.5844); best competitor FlightBERT++ (111.8125).
- RLA change (%): PA-TSMixer 2.8927, rank 6/10; best CNN-GRU (1.5689); best competitor CNN-GRU (1.5689).
- RLA worst 10%: PA-TSMixer 115.1307, rank 1/10; best PA-TSMixer (115.1307); best competitor FlightBERT++ (121.9023).
- RLA k>=3: PA-TSMixer 167.3930, rank 1/10; best PA-TSMixer (167.3930); best competitor CNN-GRU (173.3621).

### FDE (m; percentage columns in %)

| Method | Nominal H30 | RL | RL change (%) | RL worst 10% | H60 nominal [0:30] | RLA | RLA change (%) | RLA worst 10% | RLA k>=3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV | 418.49 (det.) | 419.08 (det.) | +0.14 | 438.37 (det.) | 418.49 (det.) | 427.73 (det.) | +2.21 | 468.57 (det.) | 607.06 (det.) |
| CA | 565.56 (det.) | 564.08 (det.) | -0.26 | 587.40 (det.) | 565.56 (det.) | 578.77 (det.) | +2.34 | 622.95 (det.) | 838.22 (det.) |
| CNN-GRU | 222.12 ± 1.74 | 222.24 ± 1.75 | +0.06 | 227.20 ± 2.42 | **212.36 ± 2.23** | **215.63 ± 2.16** | +1.54 | **227.61 ± 3.06** | **294.88 ± 1.54** |
| FlightBERT++ | 224.32 ± 1.55 | 222.51 ± 1.86 | **-0.80** | **225.28 ± 1.77** | 231.00 ± 1.32 | 232.64 ± 1.41 | **+0.71** | 247.52 ± 1.00 | 318.06 ± 8.98 |
| DLinear | 402.17 ± 0.64 | 402.56 ± 0.64 | +0.10 | 411.90 ± 0.55 | 401.36 ± 1.56 | 408.93 ± 1.55 | +1.89 | 439.05 ± 1.76 | 540.81 ± 2.02 |
| Transformer | 237.66 ± 2.42 | 237.63 ± 2.40 | -0.01 | 245.26 ± 2.48 | 245.41 ± 4.18 | 249.05 ± 4.30 | +1.48 | 257.37 ± 3.20 | 324.94 ± 5.74 |
| PatchTST | 361.26 ± 15.36 | 361.55 ± 15.30 | +0.08 | 373.15 ± 12.94 | 345.63 ± 2.62 | 352.46 ± 2.78 | +1.98 | 382.84 ± 3.43 | 463.43 ± 5.11 |
| TimeMixer | 334.43 ± 8.25 | 334.72 ± 8.10 | +0.09 | 347.95 ± 7.33 | 332.35 ± 5.28 | 339.95 ± 5.18 | +2.29 | 371.87 ± 6.06 | 483.20 ± 3.46 |
| TSMixer | 254.74 ± 5.83 | 255.09 ± 5.79 | +0.13 | 264.19 ± 6.01 | 238.53 ± 1.32 | 242.43 ± 1.36 | +1.64 | 254.96 ± 1.40 | 321.94 ± 2.78 |
| PA-TSMixer | **220.45 ± 0.91** | **220.73 ± 0.92** | +0.13 | 228.46 ± 1.17 | 215.13 ± 2.82 | 219.02 ± 2.88 | +1.81 | 231.54 ± 2.78 | 302.51 ± 2.58 |

- Nominal H30: PA-TSMixer 220.4534, rank 1/10; best PA-TSMixer (220.4534); best competitor CNN-GRU (222.1156).
- RL: PA-TSMixer 220.7347, rank 1/10; best PA-TSMixer (220.7347); best competitor CNN-GRU (222.2415).
- RL change (%): PA-TSMixer 0.1276, rank 8/10; best FlightBERT++ (-0.8033); best competitor FlightBERT++ (-0.8033).
- RL worst 10%: PA-TSMixer 228.4612, rank 3/10; best FlightBERT++ (225.2751); best competitor FlightBERT++ (225.2751).
- H60 nominal [0:30]: PA-TSMixer 215.1256, rank 2/10; best CNN-GRU (212.3577); best competitor CNN-GRU (212.3577).
- RLA: PA-TSMixer 219.0219, rank 2/10; best CNN-GRU (215.6312); best competitor CNN-GRU (215.6312).
- RLA change (%): PA-TSMixer 1.8111, rank 5/10; best FlightBERT++ (0.7123); best competitor FlightBERT++ (0.7123).
- RLA worst 10%: PA-TSMixer 231.5379, rank 2/10; best CNN-GRU (227.6144); best competitor CNN-GRU (227.6144).
- RLA k>=3: PA-TSMixer 302.5136, rank 2/10; best CNN-GRU (294.8816); best competitor CNN-GRU (294.8816).

## tartanaviation

Pool: 437,062 masks from 246 files / 18,989 tracks; mean loss 21.2638%; mean burst 1.2043 s; any loss 100.0000%. Calibration pooled slots: 21.8072% loss, 1.2414 s mean burst. Age>30: 0.000000% (RLA excluded fraction).

Pool integer age percentiles (s): p0=0, p25=0, p50=0, p75=0, p90=1, p95=1, p99=8, p99.9=20, p100=29. Fractions: ge_1=20.9924%, ge_2=4.0184%, ge_3=2.2283%, ge_5=1.5410%.

RL assignments — all: 57,009, worst_decile: 5,701; loss 21.3557%, burst 1.2067 s; worst-decile cutoff 22/59 lost slots; excluded pool 0.000000%.

RLA assignments — all: 57,009, worst_decile: 5,701, k_ge_3: 1,276; loss 21.2226%, burst 1.2008 s; worst-decile cutoff 22/59 lost slots; excluded pool 0.000000%.

### ADE (m; percentage columns in %)

| Method | Nominal H30 | RL | RL change (%) | RL worst 10% | H60 nominal [0:30] | RLA | RLA change (%) | RLA worst 10% | RLA k>=3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV | 158.27 (det.) | 158.65 (det.) | +0.24 | 160.23 (det.) | 158.27 (det.) | 163.91 (det.) | +3.56 | 179.79 (det.) | 284.45 (det.) |
| CA | 222.00 (det.) | 221.90 (det.) | -0.05 | 218.16 (det.) | 222.00 (det.) | 230.25 (det.) | +3.71 | 251.41 (det.) | 436.02 (det.) |
| CNN-GRU | 212.41 ± 6.93 | 212.55 ± 6.97 | +0.06 | 211.33 ± 8.09 | 217.26 ± 12.23 | 219.18 ± 12.28 | +0.89 | 226.53 ± 12.55 | 264.15 ± 10.70 |
| FlightBERT++ | 127.01 ± 1.61 | 125.04 ± 1.26 | **-1.55** | 125.18 ± 1.51 | 136.78 ± 7.06 | 136.98 ± 6.00 | **+0.15** | 142.74 ± 2.93 | 217.20 ± 7.79 |
| DLinear | 190.43 ± 0.45 | 190.60 ± 0.44 | +0.09 | 191.25 ± 0.43 | 191.02 ± 1.37 | 196.98 ± 1.37 | +3.12 | 213.20 ± 1.24 | 325.02 ± 0.82 |
| Transformer | 153.39 ± 7.69 | 153.45 ± 7.70 | +0.04 | 153.98 ± 7.73 | 167.55 ± 0.53 | 170.15 ± 0.55 | +1.55 | 177.88 ± 1.16 | 227.29 ± 2.15 |
| PatchTST | 262.41 ± 14.72 | 262.19 ± 14.58 | -0.08 | 262.65 ± 13.97 | 246.72 ± 15.51 | 251.15 ± 14.89 | +1.79 | 265.42 ± 13.93 | 356.15 ± 8.64 |
| TimeMixer | 155.87 ± 17.06 | 156.03 ± 16.85 | +0.10 | 158.76 ± 15.20 | 157.05 ± 2.35 | 161.59 ± 2.28 | +2.89 | 175.58 ± 0.95 | 270.29 ± 5.20 |
| TSMixer | 128.91 ± 8.04 | 129.13 ± 7.99 | +0.18 | 130.40 ± 7.31 | 152.57 ± 7.96 | 155.23 ± 7.89 | +1.74 | 162.35 ± 7.58 | 214.16 ± 6.24 |
| PA-TSMixer | **100.54 ± 2.09** | **100.79 ± 2.09** | +0.25 | **102.40 ± 1.85** | **105.23 ± 0.20** | **108.26 ± 0.21** | +2.88 | **116.18 ± 0.35** | **171.87 ± 0.30** |

- Nominal H30: PA-TSMixer 100.5413, rank 1/10; best PA-TSMixer (100.5413); best competitor FlightBERT++ (127.0079).
- RL: PA-TSMixer 100.7880, rank 1/10; best PA-TSMixer (100.7880); best competitor FlightBERT++ (125.0398).
- RL change (%): PA-TSMixer 0.2454, rank 10/10; best FlightBERT++ (-1.5495); best competitor FlightBERT++ (-1.5495).
- RL worst 10%: PA-TSMixer 102.3983, rank 1/10; best PA-TSMixer (102.3983); best competitor FlightBERT++ (125.1834).
- H60 nominal [0:30]: PA-TSMixer 105.2292, rank 1/10; best PA-TSMixer (105.2292); best competitor FlightBERT++ (136.7798).
- RLA: PA-TSMixer 108.2621, rank 1/10; best PA-TSMixer (108.2621); best competitor FlightBERT++ (136.9798).
- RLA change (%): PA-TSMixer 2.8822, rank 6/10; best FlightBERT++ (0.1462); best competitor FlightBERT++ (0.1462).
- RLA worst 10%: PA-TSMixer 116.1784, rank 1/10; best PA-TSMixer (116.1784); best competitor FlightBERT++ (142.7357).
- RLA k>=3: PA-TSMixer 171.8682, rank 1/10; best PA-TSMixer (171.8682); best competitor TSMixer (214.1632).

### FDE (m; percentage columns in %)

| Method | Nominal H30 | RL | RL change (%) | RL worst 10% | H60 nominal [0:30] | RLA | RLA change (%) | RLA worst 10% | RLA k>=3 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV | 365.64 (det.) | 366.38 (det.) | +0.20 | 367.70 (det.) | 365.64 (det.) | 373.95 (det.) | +2.27 | 397.80 (det.) | 537.28 (det.) |
| CA | 558.05 (det.) | 557.54 (det.) | -0.09 | 544.99 (det.) | 558.05 (det.) | 571.13 (det.) | +2.35 | 601.96 (det.) | 896.88 (det.) |
| CNN-GRU | 308.58 ± 14.05 | 308.93 ± 14.10 | +0.11 | 308.19 ± 14.59 | 296.92 ± 14.26 | 300.40 ± 14.32 | +1.17 | 311.33 ± 15.05 | 371.48 ± 10.97 |
| FlightBERT++ | 264.19 ± 4.58 | 260.44 ± 3.96 | **-1.42** | 258.85 ± 2.33 | 282.26 ± 16.08 | 280.09 ± 13.85 | **-0.77** | 284.15 ± 8.49 | 371.51 ± 13.63 |
| DLinear | 419.77 ± 2.99 | 420.03 ± 2.98 | +0.06 | 419.45 ± 2.79 | 417.69 ± 1.87 | 425.60 ± 1.80 | +1.89 | 447.23 ± 1.98 | 584.64 ± 0.70 |
| Transformer | 260.03 ± 14.63 | 260.25 ± 14.65 | +0.08 | 259.32 ± 15.63 | 276.33 ± 2.07 | 280.21 ± 2.11 | +1.41 | 290.89 ± 2.41 | 351.89 ± 4.04 |
| PatchTST | 429.68 ± 7.33 | 429.54 ± 7.23 | -0.03 | 433.01 ± 7.47 | 427.49 ± 4.48 | 434.27 ± 4.72 | +1.59 | 455.45 ± 5.47 | 573.93 ± 3.74 |
| TimeMixer | 346.61 ± 23.96 | 346.99 ± 23.72 | +0.11 | 351.66 ± 20.74 | 344.03 ± 4.87 | 350.62 ± 4.95 | +1.92 | 372.84 ± 4.26 | 499.27 ± 11.95 |
| TSMixer | 258.33 ± 8.86 | 258.81 ± 8.76 | +0.19 | 261.02 ± 7.38 | 250.92 ± 5.34 | 255.08 ± 5.33 | +1.66 | 265.47 ± 5.86 | 335.67 ± 5.23 |
| PA-TSMixer | **214.85 ± 4.47** | **215.33 ± 4.48** | +0.22 | **217.16 ± 3.72** | **213.87 ± 0.91** | **217.83 ± 1.02** | +1.85 | **227.92 ± 1.03** | **294.01 ± 1.67** |

- Nominal H30: PA-TSMixer 214.8511, rank 1/10; best PA-TSMixer (214.8511); best competitor TSMixer (258.3302).
- RL: PA-TSMixer 215.3269, rank 1/10; best PA-TSMixer (215.3269); best competitor TSMixer (258.8142).
- RL change (%): PA-TSMixer 0.2214, rank 10/10; best FlightBERT++ (-1.4182); best competitor FlightBERT++ (-1.4182).
- RL worst 10%: PA-TSMixer 217.1643, rank 1/10; best PA-TSMixer (217.1643); best competitor FlightBERT++ (258.8543).
- H60 nominal [0:30]: PA-TSMixer 213.8721, rank 1/10; best PA-TSMixer (213.8721); best competitor TSMixer (250.9185).
- RLA: PA-TSMixer 217.8296, rank 1/10; best PA-TSMixer (217.8296); best competitor TSMixer (255.0757).
- RLA change (%): PA-TSMixer 1.8504, rank 6/10; best FlightBERT++ (-0.7694); best competitor FlightBERT++ (-0.7694).
- RLA worst 10%: PA-TSMixer 227.9186, rank 1/10; best PA-TSMixer (227.9186); best competitor TSMixer (265.4734).
- RLA k>=3: PA-TSMixer 294.0115, rank 1/10; best PA-TSMixer (294.0115); best competitor TSMixer (335.6710).

## Implementation and scope

Existing parser, retained-track/slot construction, test-window loader, normalization, frozen-checkpoint configuration, forward pass, kinematic fit windows, scoring and reference assertions are imported unchanged. The existing parser's recovery of malformed/empty files is retained and recorded in the NPZ metadata and JSON. Only new replay scripts/outputs and Work_Archive/real_trace_*.log are written. No corrections to existing scripts or results were required. This is reception-pattern replay onto processed test trajectories; it does not claim trajectory-matched raw reports, independent RF packet loss, or an independent held-out reception campaign. Overlapping masks remain in the empirical pool, as requested; seed SD describes model seeds, not reception-sampling uncertainty.
