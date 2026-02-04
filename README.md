# Aero surrogate (DrivAerML summary CSVs)

This project trains regression surrogates to predict aerodynamic coefficients (cd, cl, clf, clr, cs) from DrivAer geometry parameters.

## Data files (place these in `data/`)
- `geo_parameters_all.csv`
- `force_mom_constref_all.csv`

These are small "summary" files from the DrivAerML dataset.

## Quick start
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/01_inspect.py
python scripts/02_train_baselines.py
python scripts/03_group_split_eval.py
```

## Outputs
- `reports/metrics_baselines.csv`
- plots in `reports/figures/`
