import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

from src.utils import load_and_merge

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORT_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl"]
EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]

N_MODELS = 30
RIDGE_ALPHA = 1.0
SEED = 42

def main():
    df = load_and_merge(str(GEO), str(FORCES))

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].reset_index(drop=True)
    y = df[TARGETS].reset_index(drop=True)

    # Calibration split (NOT the final test set). We use this only to set the 90th percentile thresholds.
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.15, random_state=SEED
    )

    rng = np.random.default_rng(SEED)

    # ---- Train ensemble on TRAIN portion ----
    cal_preds = np.zeros((N_MODELS, len(X_cal), len(TARGETS)), dtype=np.float64)

    for m in range(N_MODELS):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        Xb = X_train.iloc[idx]
        yb = y_train.iloc[idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED + m))
        ])
        model.fit(Xb, yb)

        cal_preds[m] = model.predict(X_cal)

    cal_std = cal_preds.std(axis=0)  # (n_cal, 2) std for cd/cl

    # 90th percentile thresholds (Option A)
    thr_cd = float(np.quantile(cal_std[:, 0], 0.90))
    thr_cl = float(np.quantile(cal_std[:, 1], 0.90))

    thresholds = {"cd_std_p90": thr_cd, "cl_std_p90": thr_cl}
    print("=== Calibrated uncertainty thresholds (90th percentile) ===")
    print(thresholds)

    # ---- Retrain a FINAL ensemble on ALL data (train + calibration) and SAVE it ----
    # This is what we'll use for prediction going forward.
    X_all = pd.concat([X_train, X_cal], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_train, y_cal], axis=0).reset_index(drop=True)

    # reset rng to keep reproducibility
    rng = np.random.default_rng(SEED)

    model_paths = []
    for m in range(N_MODELS):
        idx = rng.integers(0, len(X_all), size=len(X_all))
        Xb = X_all.iloc[idx]
        yb = y_all.iloc[idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED + m))
        ])
        model.fit(Xb, yb)

        path = MODELS_DIR / f"ridge_ensemble_{m:02d}.joblib"
        joblib.dump(model, path)
        model_paths.append(str(path))

    metadata = {
        "targets": TARGETS,
        "feature_cols": feature_cols,
        "n_models": N_MODELS,
        "ridge_alpha": RIDGE_ALPHA,
        "seed": SEED,
        "thresholds": thresholds,
        "model_files": model_paths
    }

    with open(MODELS_DIR / "ensemble_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved ensemble + metadata to:")
    print("- models/ridge_ensemble_XX.joblib")
    print("- models/ensemble_metadata.json")

    # Save thresholds into a report-friendly csv too
    pd.DataFrame([thresholds]).to_csv(REPORT_DIR / "uncertainty_thresholds_p90.csv", index=False)
    print("- reports/uncertainty_thresholds_p90.csv")

if __name__ == "__main__":
    main()
