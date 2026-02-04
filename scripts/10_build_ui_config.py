import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import joblib

from src.utils import load_and_merge

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl", "clf", "clr"]
EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]

# If you still want the p90 thresholds shown in UI, keep these.
# (They were calibrated for cd/cl using the earlier approach; we can recalibrate for 4 outputs later.)
CD_STD_P90 = 0.001444
CL_STD_P90 = 0.002911

def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def main():
    df = load_and_merge(str(GEO), str(FORCES))
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].copy()

    baseline = X.mean(axis=0)
    q05 = X.quantile(0.05)
    q95 = X.quantile(0.95)

    # load 4-target ensemble
    models = []
    for m in range(30):
        models.append(joblib.load(MODELS_DIR / f"ridge4_ensemble_{m:02d}.joblib"))

    X_base = pd.DataFrame([baseline.to_dict()])[feature_cols]
    mean, std = ensemble_predict(models, X_base)

    baseline_outputs = {t: float(mean[0, i]) for i, t in enumerate(TARGETS)}
    baseline_unc = {t: float(std[0, i]) for i, t in enumerate(TARGETS)}

    ui_config = {
        "feature_cols": feature_cols,
        "targets": TARGETS,
        "baseline": {k: float(baseline[k]) for k in feature_cols},
        "slider_min": {k: float(q05[k]) for k in feature_cols},
        "slider_max": {k: float(q95[k]) for k in feature_cols},
        "baseline_outputs": baseline_outputs,
        "baseline_uncertainty": baseline_unc,
        "uncertainty_thresholds": {
            "cd_std_p90": CD_STD_P90,
            "cl_std_p90": CL_STD_P90
        },
        "notes": {
            "baseline_definition": "Dataset mean of geometry parameters",
            "slider_limits": "5th–95th percentile of dataset"
        }
    }

    outpath = MODELS_DIR / "ui_config_4targets.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(ui_config, f, indent=2)

    print("Wrote:", outpath)

if __name__ == "__main__":
    main()
