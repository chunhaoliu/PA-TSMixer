# PA-TSMixer: lightweight multi-scale temporal mixing for aircraft trajectory prediction

Code, checkpoints and evaluation outputs for

> C. Liu, P. Wu, S. He, G. Pan, and J. H. Park, *Lightweight Multi-Scale Temporal Mixing for Short-Term
> Aircraft Trajectory Prediction in Terminal Airspace*, submitted to IEEE Transactions on Vehicular Technology.

This release contains the PA-TSMixer model and its single-scale parent, the construction of the processed
XYZ benchmark from the public TrajAir and TartanAviation (KBTP) files, a training script that retrains
PA-TSMixer and the TSMixer controls with the settings of the paper, the trained PA-TSMixer and TSMixer
checkpoints of the main and capacity-control studies, the constant-velocity and constant-acceleration
baselines, and the per-run test results of every compared method together with a script that recomputes
the result tables. The configurations of all baselines are listed in `CONFIGS.md`.

## Contents

| Path | What |
|---|---|
| `pa_tsmixer/models.py` | PA-TSMixer (full-, half- and quarter-resolution temporal branches, learned scale fusion; Section IV) and TSMixer; options for the equal-weight and uniform-width controls |
| `pa_tsmixer/data.py` | processed-benchmark construction (segmentation, aircraft-level split, windows, normalization; Section V-A) and test-window access |
| `pa_tsmixer/data_manifest.json` | SHA-256 of the processed arrays used in the paper |
| `scripts/prepare_data.py` | builds the processed data from the public source files and checks it byte for byte against the paper build |
| `scripts/train.py` | trains PA-TSMixer or a TSMixer control with the paper settings (AdamW, cosine schedule, early stopping on validation ADE) and evaluates it on the test split |
| `scripts/evaluate.py` | evaluates the released checkpoints on the test split and compares with the reported ADE, FDE and MSE |
| `scripts/kinematic_baselines.py` | CV and CA baselines with the validation-selected fitting windows |
| `scripts/reproduce_tables.py` | recomputes every printed cell of Tables II, III and V from `results/runs.json` |
| `pa_tsmixer/motion.py` | history-only motion groups (climb/descent/level, turning/non-turning/low speed; Section VI-C) |
| `scripts/branch_analysis.py` | learned scale weights and test-time removal of each temporal branch for the 24 main PA-TSMixer models (Fig. 8) |
| `results/branch_analysis.json` | output of `branch_analysis.py`: weights per block and ADE with each branch removed, overall and per motion group |
| `checkpoints/` | 108 checkpoints: PA-TSMixer and TSMixer, 2 sources x 4 horizons x 3 seeds (main study), and the five capacity/component controls of Table V at H = 30 and 60 (`index.json` lists settings, SHA-256 and reported metrics) |
| `results/runs.json` | per-run test metrics of all methods: ADE, FDE, MSE, horizontal ADE, altitude MAE and ADE per motion group |
| `results/paper_tables.json` | the cells of Tables II, III and V as printed in the paper |
| `CONFIGS.md` | protocol, training settings and baseline configurations |

## Usage

Python 3.12; `pip install -r requirements.txt`.

```bash
python -m unittest discover -s tests -t .
python scripts/reproduce_tables.py
python scripts/prepare_data.py --trajair <TrajAir 111_days folder> --tartan-zip <TartanAviation kbtp/processed.zip> --output data
python scripts/evaluate.py --data data
python scripts/kinematic_baselines.py --data data
python scripts/branch_analysis.py --data data
python scripts/train.py --data data --dataset trajair --horizon 10 --seed 42
```

`reproduce_tables.py` needs no data and exits with status 0 when all 537 printed cells are reproduced.
`prepare_data.py` reports for each processed array whether it is identical to the paper build (it takes about
a minute). `evaluate.py` and `kinematic_baselines.py` then recompute the test metrics of the released models and
baselines; on our machine the largest relative difference from the reported values is below 1e-7.
`branch_analysis.py` first checks that the motion groups match the window counts of the paper and then
recomputes the scale analysis of Fig. 8 (a few minutes on a GPU).
`train.py` writes the selected checkpoint and the training history to `runs/`; options select TSMixer
(`--model TSMixer`), the width (`--d-model 167` for the matched-size control) and the output
(`--last-residual` / `--no-last-residual`). GPU training is not bitwise deterministic: retraining PA-TSMixer on
TrajAir at H = 10 with seeds 42, 123 and 456 gave a mean test ADE of 33.57 m, against 33.68 +/- 0.37 m reported
(about 2-3 minutes per run on an RTX 5090).

The source data are not redistributed here: TrajAir (111-day processed trajectories) is available from
<https://theairlab.org/trajair/> and TartanAviation (KBTP processed ADS-B trajectories) from
<https://github.com/castacks/TartanAviation>.

### Using the model

```python
import torch
from pa_tsmixer.models import PATSMixer
from pa_tsmixer.data import Windows

model = PATSMixer(pred_len=30)
model.load_state_dict(torch.load("checkpoints/main_trajair_H30_pa-tsmixer_s42.pt", weights_only=True))
test = Windows("data/trajair", "test", pred_len=30)
history, future = test.batch(0, 8)             # standardized [8, 60, 3]; metres [8, 30, 3]
forecast = test.to_metres(model.eval()(torch.from_numpy(history)).detach().numpy())   # metres [8, 30, 3]
```

## License

MIT, see `LICENSE`.
