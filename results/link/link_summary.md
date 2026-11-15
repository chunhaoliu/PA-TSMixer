# ADS-B link-impairment evaluation

Started (UTC): 2026-11-15T04:06:03.501040+00:00. Evaluation runtime: 85.67 s (1.43 min).

Frozen checkpoints; test split; FP32; TF32 disabled. Recomputed released-model rows. Targets remain clean. A fresh np.random.default_rng(2026) is used per dataset, experiment and level; each resulting input array is shared across methods and seeds.

Learned-model cells are mean ± sample standard deviation across seeds 42/123/456 (ddof=1), in metres. CV/CA are deterministic single evaluations, marked (det.); their sample standard deviation is not defined. Bold marks the smallest mean per column.

Sanity assertions: 80 reference comparisons passed at relative tolerance 1e-5; largest relative error 1.741e-08. No assertions relaxed. H=30 clean/pi=0 ADE and every learned H=60 full-future ADE were checked. All realized eligible-report loss fractions passed absolute tolerance 0.02.

Test windows: trajair: 21,346; tartanaviation: 57,009.

E1 uses H=30 checkpoints. E2 and E3 use H=60 checkpoints and score the unchanged future slice [k:k+30], i.e. leads k+1 through k+30 after the last report. Nominal E3 uses L=5 as an inert convention when pi=0.

## trajair

### E1 — mean ADE (m), L=1 s

| Method | pi=0% | pi=10% | pi=20% | pi=30% | pi=40% | pi=50% |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 175.45 (det.) | 175.47 (det.) | 175.50 (det.) | 175.48 (det.) | 175.50 (det.) | 175.47 (det.) |
| CA | 218.45 (det.) | 218.32 (det.) | 218.18 (det.) | 218.07 (det.) | 217.95 (det.) | 218.07 (det.) |
| TSMixer | 120.34 ± 2.89 | 120.33 ± 2.89 | 120.30 ± 2.87 | 120.29 ± 2.87 | 120.25 ± 2.85 | 120.26 ± 2.84 |
| PA-TSMixer | **100.23 ± 0.47** | **100.22 ± 0.46** | **100.20 ± 0.46** | **100.21 ± 0.45** | **100.18 ± 0.46** | **100.16 ± 0.47** |

### E1 — mean ADE (m), L=5 s

| Method | pi=0% | pi=10% | pi=20% | pi=30% | pi=40% | pi=50% |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 175.45 (det.) | 176.36 (det.) | 177.28 (det.) | 177.96 (det.) | 178.93 (det.) | 180.01 (det.) |
| CA | 218.45 (det.) | 217.67 (det.) | 217.00 (det.) | 216.47 (det.) | 215.85 (det.) | 214.95 (det.) |
| TSMixer | 120.34 ± 2.89 | 121.00 ± 2.85 | 121.77 ± 2.83 | 122.38 ± 2.80 | 122.97 ± 2.77 | 123.99 ± 2.67 |
| PA-TSMixer | **100.23 ± 0.47** | **100.87 ± 0.49** | **101.68 ± 0.50** | **102.23 ± 0.50** | **102.80 ± 0.52** | **103.97 ± 0.52** |

### E1 — mean ADE (m), L=10 s

| Method | pi=0% | pi=10% | pi=20% | pi=30% | pi=40% | pi=50% |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 175.45 (det.) | 177.79 (det.) | 180.15 (det.) | 183.31 (det.) | 185.77 (det.) | 188.13 (det.) |
| CA | 218.45 (det.) | 219.39 (det.) | 219.77 (det.) | 221.12 (det.) | 222.81 (det.) | 224.09 (det.) |
| TSMixer | 120.34 ± 2.89 | 122.63 ± 2.88 | 124.94 ± 2.76 | 127.79 ± 2.79 | 130.29 ± 2.66 | 132.82 ± 2.65 |
| PA-TSMixer | **100.23 ± 0.47** | **102.66 ± 0.53** | **105.01 ± 0.60** | **108.02 ± 0.66** | **110.94 ± 0.70** | **113.29 ± 0.76** |

### E2 — mean ADE (m)

| Method | k=0 s | k=1 s | k=2 s | k=3 s | k=4 s | k=5 s |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 175.45 (det.) | 189.78 (det.) | 204.65 (det.) | 220.06 (det.) | 236.03 (det.) | 252.56 (det.) |
| CA | 218.45 (det.) | 238.12 (det.) | 258.77 (det.) | 280.44 (det.) | 303.16 (det.) | 326.93 (det.) |
| TSMixer | 137.23 ± 0.68 | 142.80 ± 0.61 | 148.67 ± 0.56 | 154.83 ± 0.54 | 161.29 ± 0.54 | 168.03 ± 0.56 |
| PA-TSMixer | **95.31 ± 0.68** | **101.87 ± 0.71** | **108.51 ± 0.74** | **115.29 ± 0.78** | **122.20 ± 0.81** | **129.26 ± 0.85** |

### E3 — combined conditions (m)

Nominal: sigma=0 m, pi=0, k=0 s. Moderate: sigma=10 m, pi=0.2, L=5 s, k=1 s. Severe: sigma=20 m, pi=0.4, L=5 s, k=3 s.

| Method | nominal ADE | nominal FDE | moderate ADE | moderate FDE | severe ADE | severe FDE |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 175.45 (det.) | 418.49 (det.) | 236.98 (det.) | 514.66 (det.) | 350.25 (det.) | 685.53 (det.) |
| CA | 218.45 (det.) | 565.56 (det.) | 284.98 (det.) | 692.10 (det.) | 434.84 (det.) | 975.40 (det.) |
| TSMixer | 137.23 ± 0.68 | 238.53 ± 1.32 | 152.29 ± 1.13 | 264.89 ± 1.91 | 192.95 ± 2.41 | 330.79 ± 3.09 |
| PA-TSMixer | **95.31 ± 0.68** | **199.03 ± 1.46** | **111.36 ± 1.22** | **220.80 ± 2.58** | **151.30 ± 2.43** | **277.66 ± 5.12** |

## tartanaviation

### E1 — mean ADE (m), L=1 s

| Method | pi=0% | pi=10% | pi=20% | pi=30% | pi=40% | pi=50% |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 158.27 (det.) | 158.32 (det.) | 158.36 (det.) | 158.42 (det.) | 158.48 (det.) | 158.49 (det.) |
| CA | 222.00 (det.) | 221.91 (det.) | 221.77 (det.) | 221.71 (det.) | 221.62 (det.) | 221.48 (det.) |
| TSMixer | 128.91 ± 8.04 | 128.92 ± 8.04 | 128.89 ± 8.04 | 128.91 ± 8.03 | 128.89 ± 8.01 | 128.93 ± 8.01 |
| PA-TSMixer | **100.54 ± 2.09** | **100.55 ± 2.09** | **100.54 ± 2.09** | **100.53 ± 2.10** | **100.52 ± 2.10** | **100.59 ± 2.08** |

### E1 — mean ADE (m), L=5 s

| Method | pi=0% | pi=10% | pi=20% | pi=30% | pi=40% | pi=50% |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 158.27 (det.) | 159.10 (det.) | 159.98 (det.) | 160.80 (det.) | 161.53 (det.) | 162.28 (det.) |
| CA | 222.00 (det.) | 220.97 (det.) | 219.61 (det.) | 218.48 (det.) | 217.39 (det.) | 216.42 (det.) |
| TSMixer | 128.91 ± 8.04 | 129.46 ± 7.93 | 130.04 ± 7.79 | 130.55 ± 7.69 | 131.12 ± 7.59 | 131.64 ± 7.43 |
| PA-TSMixer | **100.54 ± 2.09** | **101.12 ± 2.07** | **101.76 ± 2.04** | **102.32 ± 2.00** | **102.89 ± 2.05** | **103.43 ± 2.02** |

### E1 — mean ADE (m), L=10 s

| Method | pi=0% | pi=10% | pi=20% | pi=30% | pi=40% | pi=50% |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 158.27 (det.) | 160.49 (det.) | 162.66 (det.) | 164.61 (det.) | 167.04 (det.) | 169.06 (det.) |
| CA | 222.00 (det.) | 221.30 (det.) | 220.89 (det.) | 220.37 (det.) | 219.60 (det.) | 218.88 (det.) |
| TSMixer | 128.91 ± 8.04 | 130.77 ± 7.77 | 132.80 ± 7.48 | 134.52 ± 7.22 | 136.55 ± 6.93 | 138.38 ± 6.73 |
| PA-TSMixer | **100.54 ± 2.09** | **102.59 ± 2.03** | **104.85 ± 1.95** | **106.71 ± 1.87** | **109.00 ± 1.83** | **110.94 ± 1.82** |

### E2 — mean ADE (m)

| Method | k=0 s | k=1 s | k=2 s | k=3 s | k=4 s | k=5 s |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 158.27 (det.) | 170.66 (det.) | 183.43 (det.) | 196.64 (det.) | 210.29 (det.) | 224.39 (det.) |
| CA | 222.00 (det.) | 241.28 (det.) | 261.43 (det.) | 282.50 (det.) | 304.54 (det.) | 327.54 (det.) |
| TSMixer | 152.57 ± 7.96 | 157.99 ± 7.76 | 163.71 ± 7.56 | 169.71 ± 7.35 | 176.00 ± 7.14 | 182.57 ± 6.93 |
| PA-TSMixer | **97.36 ± 1.77** | **103.77 ± 1.84** | **110.21 ± 1.90** | **116.75 ± 1.96** | **123.42 ± 2.01** | **130.22 ± 2.06** |

### E3 — combined conditions (m)

Nominal: sigma=0 m, pi=0, k=0 s. Moderate: sigma=10 m, pi=0.2, L=5 s, k=1 s. Severe: sigma=20 m, pi=0.4, L=5 s, k=3 s.

| Method | nominal ADE | nominal FDE | moderate ADE | moderate FDE | severe ADE | severe FDE |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 158.27 (det.) | 365.64 (det.) | 194.82 (det.) | 420.01 (det.) | 263.70 (det.) | 519.67 (det.) |
| CA | 222.00 (det.) | 558.05 (det.) | 292.10 (det.) | 693.40 (det.) | 441.44 (det.) | 975.85 (det.) |
| TSMixer | 152.57 ± 7.96 | 250.92 ± 5.34 | 166.33 ± 6.99 | 275.40 ± 4.06 | 204.31 ± 4.78 | 337.23 ± 2.51 |
| PA-TSMixer | **97.36 ± 1.77** | **197.22 ± 2.49** | **115.45 ± 2.05** | **222.88 ± 3.10** | **157.41 ± 3.89** | **283.30 ± 6.73** |

## Generator checks

Loss fractions use indices 0..58 (59 eligible reports); index 59 is always received. Mean bursts pool all within-window lost runs, including truncated boundary runs. Window truncation can reduce their mean below L, especially at L=10 s. A mean burst of 0 at pi=0 denotes no lost runs.

| Dataset | pi | L (s) | Realized loss | Realized mean burst (s) |
|:--|--:|--:|--:|--:|
| trajair | 0.0 | 1 | 0.000000 | 0.0000 |
| trajair | 0.1 | 1 | 0.100125 | 1.0000 |
| trajair | 0.2 | 1 | 0.200252 | 1.0000 |
| trajair | 0.3 | 1 | 0.300125 | 1.0000 |
| trajair | 0.4 | 1 | 0.400297 | 1.0000 |
| trajair | 0.5 | 1 | 0.499995 | 1.0000 |
| trajair | 0.0 | 5 | 0.000000 | 0.0000 |
| trajair | 0.1 | 5 | 0.100620 | 4.7389 |
| trajair | 0.2 | 5 | 0.200995 | 4.6953 |
| trajair | 0.3 | 5 | 0.300058 | 4.6822 |
| trajair | 0.4 | 5 | 0.400262 | 4.6793 |
| trajair | 0.5 | 5 | 0.501282 | 4.6866 |
| trajair | 0.0 | 10 | 0.000000 | 0.0000 |
| trajair | 0.1 | 10 | 0.099017 | 8.6330 |
| trajair | 0.2 | 10 | 0.199007 | 8.6772 |
| trajair | 0.3 | 10 | 0.299237 | 8.6399 |
| trajair | 0.4 | 10 | 0.398597 | 8.6514 |
| trajair | 0.5 | 10 | 0.498984 | 8.6559 |
| tartanaviation | 0.0 | 1 | 0.000000 | 0.0000 |
| tartanaviation | 0.1 | 1 | 0.099998 | 1.0000 |
| tartanaviation | 0.2 | 1 | 0.200228 | 1.0000 |
| tartanaviation | 0.3 | 1 | 0.300274 | 1.0000 |
| tartanaviation | 0.4 | 1 | 0.400048 | 1.0000 |
| tartanaviation | 0.5 | 1 | 0.500052 | 1.0000 |
| tartanaviation | 0.0 | 5 | 0.000000 | 0.0000 |
| tartanaviation | 0.1 | 5 | 0.099510 | 4.6774 |
| tartanaviation | 0.2 | 5 | 0.200290 | 4.6824 |
| tartanaviation | 0.3 | 5 | 0.300063 | 4.6804 |
| tartanaviation | 0.4 | 5 | 0.399745 | 4.6806 |
| tartanaviation | 0.5 | 5 | 0.500833 | 4.6890 |
| tartanaviation | 0.0 | 10 | 0.000000 | 0.0000 |
| tartanaviation | 0.1 | 10 | 0.100252 | 8.7020 |
| tartanaviation | 0.2 | 10 | 0.199679 | 8.6707 |
| tartanaviation | 0.3 | 10 | 0.299817 | 8.6723 |
| tartanaviation | 0.4 | 10 | 0.399804 | 8.6731 |
| tartanaviation | 0.5 | 10 | 0.499725 | 8.6788 |
