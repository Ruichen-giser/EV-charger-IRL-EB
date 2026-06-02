# EV-charger-IRL

Inverse reinforcement learning for EV charging station deployment on 2 km county grids. The pipeline has three standalone packages:

| Package | Role |
|---------|------|
| **IRL_data** | Raw GIS/EVCS → prepared pickle + grid feature `.npz` |
| **IRL_BC** | Behavior cloning (BC) with SimpleGridCNN |
| **IRL_IQ** | IQ-Learn with the same CNN backbone |

## Repository layout

```text
EV-charger-IRL/
├── data/              # raw datasets (see data/README.md)
├── outputs/           # generated artifacts (gitignored)
├── IRL_data/          # data preparation
├── IRL_BC/            # BC training & evaluation
└── IRL_IQ/            # IQ-Learn training & evaluation
```

## Quick start

### 1. Install dependencies

Each package has its own `requirements.txt`. For the full pipeline:

```bash
pip install -r IRL_data/requirements.txt
pip install -r IRL_BC/requirements.txt
# IRL_IQ uses the same deps as IRL_BC
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Put raw files under `data/` (see [data/README.md](data/README.md)), then from the repo root:

```bash
cd IRL_data
python main.py
```

Outputs:

- `outputs/prepared_data/prepared_irl_dataset.pkl`
- `outputs/prepared_data/grid_tensors/<County>_grid_features.npz`

### 3. Train IRL models

Behavior cloning:

```bash
cd IRL_BC
python main.py --county Los_Angeles
```

IQ-Learn:

```bash
cd IRL_IQ
python main.py --county Los_Angeles
```

Both trainers read grid tensors from `outputs/prepared_data/grid_tensors/` by default and write checkpoints to `outputs/bc_output/` or `outputs/iq_output/`.

### MDP modes

Use `--mdp-mode legacy` (default) or `--mdp-mode repeat`. See package READMEs for details.

## Optional utilities

```bash
# Feature distribution plots
python IRL_data/feature_statistics.py

# Expert first-step visualization (BC package)
python IRL_BC/scripts/visualize_expert_first_step.py
```

## Citation

If you use this code, please cite the associated paper (link TBD).
