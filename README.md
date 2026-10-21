# PA-TSMixer: multi-scale temporal mixing for aircraft trajectory prediction (core release)

Code, checkpoints and evaluation outputs for

> C. Liu, P. Wu, S. He, G. Pan, and J. H. Park, *PA-TSMixer: Multi-Scale Temporal Mixing for Aircraft
> Trajectory Prediction*, submitted to IEEE Transactions on Vehicular Technology.

This core release contains the PA-TSMixer model and its single-scale parent, the construction of the processed
XYZ benchmark from the public TrajAir and TartanAviation (KBTP) files, the trained PA-TSMixer and TSMixer
checkpoints of the main and capacity-control studies, the constant-velocity and constant-acceleration
baselines, and the per-run test results of every compared method together with a script that recomputes
the result tables. The full training and experiment pipeline and the checkpoints of the other learned
baselines are not included; they are available from the corresponding authors on reasonable request.

## Contents

| Path | What |
|---|---|
| `pa_tsmixer/models.py` | PA-TSMixer (full-, half- and quarter-resolution temporal branches, learned scale fusion; Section IV) and TSMixer; options for the equal-weight and uniform-width controls |
| `pa_tsmixer/data.py` | processed-benchmark construction (segmentation, aircraft-level split, windows, normalization; Section V-A) and test-window access |
| `pa_tsmixer/data_manifest.json` | SHA-256 of the processed arrays used in the paper |
| `scripts/prepare_data.py` | builds the processed data from the public source files and checks it byte for byte against the paper build |
| `scripts/evaluate.py` | evaluates the released checkpoints on the test split and compares with the reported ADE, FDE and MSE |
| `scripts/kinematic_baselines.py` | CV and CA baselines with the validation-selected fitting windows |
| `scripts/reproduce_tables.py` | recomputes every printed cell of Tables II, III and V from `results/runs.json` |
| `checkpoints/` | 96 checkpoints: PA-TSMixer and TSMixer, 2 sources x 4 horizons x 3 seeds (main study), and the four capacity/component controls at H = 30 and 60 (`index.json` lists settings, SHA-256 and reported metrics) |
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
```

`reproduce_tables.py` needs no data and exits with status 0 when all 421 printed cells are reproduced.
`prepare_data.py` reports for each processed array whether it is identical to the paper build (it takes about
a minute). `evaluate.py` and `kinematic_baselines.py` then recompute the test metrics of the released models and
baselines; on our machine the largest relative difference from the reported values is below 1e-7.

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
