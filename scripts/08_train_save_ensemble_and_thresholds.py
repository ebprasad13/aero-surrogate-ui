import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.utils import load_and_merge

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORT_DIR = Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl", "clf", "clr"]
EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]

N_MODELS = 30
RIDGE_ALPHA = 1.0
SEED = 42

def main():
    df = load_and_merge(str(GEO), str(FORCES))
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].reset_index(drop=True)
    y = df[TARGETS].reset_index(drop=True)

    rng = np.random.default_rng(SEED)

    model_files = []
    for m in range(N_MODELS):
        idx = rng.integers(0, len(X), size=len(X))  # bootstrap all data
        Xb = X.iloc[idx]
        yb = y.iloc[idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=SEED + m))
        ])
        model.fit(Xb, yb)

        path = MODELS_DIR / f"ridge4_ensemble_{m:02d}.joblib"
        joblib.dump(model, path)
        model_files.append(str(path))

    meta = {
        "targets": TARGETS,
        "feature_cols": feature_cols,
        "n_models": N_MODELS,
        "ridge_alpha": RIDGE_ALPHA,
        "seed": SEED,
        "model_files": model_files,
    }

    with open(MODELS_DIR / "ensemble4_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved 4-target ensemble + metadata:")
    print("- models/ridge4_ensemble_XX.joblib")
    print("- models/ensemble4_metadata.json")

if __name__ == "__main__":
    main()
