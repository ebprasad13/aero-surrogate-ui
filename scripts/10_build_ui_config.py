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

TARGETS_FULL = ["cd", "cl", "clf", "clr"]
EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]

# Your calibrated p90 thresholds (from uncertainty_thresholds_p90.csv)
CD_STD_P90 = 0.001444
CL_STD_P90 = 0.002911

def load_ensemble_models(n_models=30):
    models = []
    for m in range(n_models):
        path = MODELS_DIR / f"ridge_ensemble_{m:02d}.joblib"
        models.append(joblib.load(path))
    return models

def ensemble_predict(models, X):
    preds = np.stack([mdl.predict(X) for mdl in models], axis=0)  # (n_models, n_samples, n_targets)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

def main():
    df = load_and_merge(str(GEO), str(FORCES))

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].copy()

    # Baseline = dataset mean (reference car)
    baseline = X.mean(axis=0)

    # Slider limits = 5th–95th percentile (robust range)
    q05 = X.quantile(0.05)
    q95 = X.quantile(0.95)

    # Load ensemble and compute baseline outputs
    models = load_ensemble_models(n_models=30)

    X_base = pd.DataFrame([baseline.to_dict()])[feature_cols]
    pred_mean, pred_std = ensemble_predict(models, X_base)

    # NOTE: your ensemble was trained to output cd/cl only OR cd/cl for the saved model set.
    # If your saved ensemble predicts ONLY cd/cl, baseline_outputs will contain those 2.
    # We'll still store placeholders for clf/clr if absent.
    baseline_outputs = {}
    if pred_mean.shape[1] == 2:
        baseline_outputs["cd"] = float(pred_mean[0, 0])
        baseline_outputs["cl"] = float(pred_mean[0, 1])
    else:
        # If you later saved an ensemble that outputs 4 targets
        for i, t in enumerate(TARGETS_FULL[:pred_mean.shape[1]]):
            baseline_outputs[t] = float(pred_mean[0, i])

    ui_config = {
        "feature_cols": feature_cols,
        "baseline": {k: float(baseline[k]) for k in feature_cols},
        "slider_min": {k: float(q05[k]) for k in feature_cols},
        "slider_max": {k: float(q95[k]) for k in feature_cols},
        "baseline_outputs": baseline_outputs,
        "uncertainty_thresholds": {
            "cd_std_p90": CD_STD_P90,
            "cl_std_p90": CL_STD_P90
        },
        "notes": {
            "baseline_definition": "Dataset mean of geometry parameters",
            "slider_limits": "5th–95th percentile of dataset"
        }
    }

    outpath = MODELS_DIR / "ui_config.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(ui_config, f, indent=2)

    print("Wrote:", outpath)

if __name__ == "__main__":
    main()
