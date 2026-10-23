# Protocol and configurations

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

## Training (all learned models)

AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, cosine schedule, at most 30 epochs with early-stopping
patience 8, loss: MSE on standardized coordinates, checkpoint: lowest validation 3D ADE in metres, FP32 with TF32
disabled, seeds 42, 123 and 456.

## Model settings

| Model | Settings |
|---|---|
| PA-TSMixer | 2 blocks, d = 128; temporal branch widths 128 / 64 / 32 on 60 / 30 / 15 steps; branch dropout 0.1 / 0.15 / 0.2; channel MLP width 128 |
| TSMixer | 2 blocks, d = 128 |
| DLinear, Transformer, PatchTST, TimeMixer | d_model 128, 2 encoder layers, d_ff 256, 4 heads, dropout 0.1, label length 30, PatchTST patch length 16 |
| CV / CA | least-squares fit to the last w observed points, w chosen on the validation split (stored as `fit_window` in `results/runs.json`) |

## Capacity and component controls (Table V)

PA-TSMixer adds its forecast to the last observed position (`last_residual=True`); the TSMixer baseline of Table II
outputs positions directly. The controls separate the two elements at matched size.

| Control | Change from PA-TSMixer |
|---|---|
| TSMixer, matched size | single-scale TSMixer with d = 167 (44,708 parameters), same residual output |
| PA, equal weights | scale logits fixed at zero (weights 1/3); the 1x1 fusion convolution is kept |
| PA, uniform widths | width 96 in all three temporal branches (44,800 parameters) |
| PA, direct output | the same model without the residual output (`last_residual=False`) |
| TSMixer, matched, direct | matched-size TSMixer without the residual output |
