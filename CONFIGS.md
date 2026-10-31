# Protocol and configurations

*Lightweight Multi-Scale Trajectory Prediction for Connected Aircraft Under ADS-B Link Impairments*,
submitted to IEEE Transactions on Vehicular Technology.

## Data and evaluation

| Item | Setting |
|---|---|
| Input / output | XYZ positions in metres from the published processed coordinates (km x 1000) |
| Sampling | 1 s; 60 observed points; horizons H = 10, 20, 30, 60 s |
| Segments | per aircraft, cut at time gaps, at > 300 m horizontal or > 100 m vertical displacement in one second; segments shorter than 120 points are dropped |
| Windows | 120 valid points (60 + 60), stride 15, identical windows of the same aircraft removed; all horizons share the same origins |
| Split | aircraft-level, SHA-256 rule `TVT-aircraft-split-42|<id>`, 70/15/15 (not the official or a chronological split) |
| Normalization | per-axis mean and standard deviation of the training points |
| Metrics | ADE and FDE: 3D displacement (m); MSE: mean squared XYZ coordinate error (m^2), on the test split |

## Training (all learned models; Section V-C)

AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, cosine schedule, at most 30 epochs with early-stopping
patience 8, loss: MSE on standardized coordinates, checkpoint: lowest validation 3D ADE in metres, FP32 with TF32
disabled, seeds 42, 123 and 456. `scripts/train.py` implements these settings for PA-TSMixer and the TSMixer
controls.

## Model settings

| Model | Settings |
|---|---|
| PA-TSMixer | 2 blocks, d = 128; temporal branch widths 128 / 64 / 32 on 60 / 30 / 15 steps; branch dropout 0.1 / 0.15 / 0.2; channel MLP width 128 |
| TSMixer | 2 blocks, d = 128 |
| DLinear, Transformer, PatchTST, TimeMixer | d_model 128, 2 encoder layers, d_ff 256, 4 heads, dropout 0.1, label length 30, PatchTST patch length 16 |
| CNN-GRU | ported from the official implementation (Shafienya and Regan, 2022): spatial input = 60 standardized XYZ positions, temporal input = 60 per-second displacements of the history, direct H x 3 output, MSE loss, learning rate 1e-3; PCA compression and the image branch omitted |
| FlightBERT++ | ported from the official implementation (Guo et al., AAAI 2024): binary attributes of absolute positions (1 m) and per-second displacements, 4 encoder and 4 decoder layers, embedding 128, 4 heads, dropout 0.1, bitwise cross-entropy, learning rate 1e-4; decoded displacements accumulated from the last observation |
| CV / CA | least-squares fit to the last w observed points, w chosen on the validation split (stored as `fit_window` in `results/runs.json`) |

The aviation-specific baselines are reported through their per-run results (`results/runs.json`); their training
code is part of the full pipeline, available on request.

## Capacity and component controls (Table IV)

PA-TSMixer adds its forecast to the last observed position (`last_residual=True`); the TSMixer baseline of Table II
outputs positions directly. The controls separate the two elements at matched size.

| Control | Change from PA-TSMixer |
|---|---|
| TSMixer, matched size | single-scale TSMixer with d = 167 (44,708 parameters), same residual output |
| PA, equal weights | scale logits fixed at zero (weights 1/3); the 1x1 fusion convolution is kept |
| PA, uniform widths | width 96 in all three temporal branches (44,800 parameters) |
| PA, direct output | the same model without the residual output (`last_residual=False`) |
| TSMixer, matched, direct | matched-size TSMixer without the residual output |

## ADS-B link impairments (Section VI-D; Fig. 6 and Table V)

`scripts/link_noise.py` uses H=30 checkpoints, Gaussian position noise sigma = 0/5/10/20/40 m on all history
positions and axes, and independent loss p = 0/0.1/0.2/0.3/0.4 over the first 59 reports. `link_impairment.py`
uses stationary Gilbert–Elliott loss pi = 0/0.1/0.2/0.3/0.4/0.5, mean burst L = 1/5/10 reports (H=30),
staleness k = 0..5 s (H=60, scoring future slice [k:k+30]), and combined nominal/moderate/severe conditions:
(sigma, pi, L, k) = (0, 0, 5, 0) / (10, 0.2, 5, 1) / (20, 0.4, 5, 3).
Loss always retains the latest report; missing positions use `np.interp`, with constant fill before the first
retained report. A fresh `default_rng(2026)` per dataset/level produces inputs shared by all methods and seeds;
combined conditions draw noise before loss. Targets stay clean, models remain frozen, FP32 and TF32 off.

`link_calibration.py` deduplicates raw in-region reports after filtering to 6 km (TrajAir) or 11 km
(TartanAviation) around KBTP, with altitude <=7500 ft MSL. Tracks split at gaps >30 s and must span >=60 s;
calibration includes split sensitivities 15/30/60 s. Raw receiver logs come from TrajAir's `raw_data`
(<https://theairlab.org/trajair/>) and the TartanAviation repository's raw ADS-B logs
(<https://github.com/castacks/TartanAviation>); they are not redistributed.

`real_trace_masks.py` samples now slots 59,74,89,... per retained track and shifts each 60-slot history to its
latest received report. `real_trace.py` draws masks uniformly with replacement: seed 2026 for loss-only RL
(H=30), seed 2027 for loss-plus-age RLA (H=60, pool conditioned on k<=30, scoring [k:k+30]). Worst-decile
subsets use ceil(N/10) highest loss counts with stable window-order tie breaks; RLA also reports k>=3.
These are empirical reception gaps, not independently identified RF packet losses.

The scripts evaluate the released PA-TSMixer/TSMixer checkpoints and CV/CA, with strict clean-reference and
published ADE/FDE checks (<1e-5 relative difference). `results/link/` retains the original full-precision
ten-method results and derived mask pools; recomputed outputs default to its ignored `recomputed/` subdirectory.
Tables II–IV remain in `results/paper_tables.json`; Table V is printed by `scripts/make_link_table.py`.
Branch contributions are Fig. 5, motion groups Fig. 4, stepwise errors Fig. 3, and accuracy–latency Fig. 7;
computational cost is Table VI and Section VI-E.
