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
from sklearn.model_selection import train_test_split

from src.utils import load_and_merge

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl", "clf", "clr"]
EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]

N_MODELS = 30
TOP_K = 8          # choose 6, 7, or 8
SEED = 42

def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def main():
    df = load_and_merge(str(GEO), str(FORCES))
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols].reset_index(drop=True)
    y = df[TARGETS].reset_index(drop=True)

    # ---- Load your saved 4-target ensemble models ----
    model_paths = [MODELS_DIR / f"ridge4_ensemble_{m:02d}.joblib" for m in range(N_MODELS)]
    missing = [str(p) for p in model_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ensemble model files. Upload/commit these first:\n" + "\n".join(missing)
        )
    models = [joblib.load(p) for p in model_paths]

    # ---- Baseline = dataset mean ----
    baseline = X.mean(axis=0)

    # Slider limits = 5th–95th percentile
    q05 = X.quantile(0.05)
    q95 = X.quantile(0.95)

    # ---- Baseline outputs (predicted at baseline geometry) ----
    X_base = pd.DataFrame([baseline.to_dict()])[feature_cols]
    base_mean, base_std = ensemble_predict(models, X_base)

    baseline_outputs = {t: float(base_mean[0, i]) for i, t in enumerate(TARGETS)}
    baseline_unc = {t: float(base_std[0, i]) for i, t in enumerate(TARGETS)}

    # ---- Compute p90 uncertainty thresholds for ALL 4 targets ----
    # We use a calibration split, but *we don’t retrain models* — we just measure ensemble disagreement on cal.
    X_train, X_cal = train_test_split(X, test_size=0.15, random_state=SEED)

    _, cal_std = ensemble_predict(models, X_cal)
    thresholds = {f"{t}_std_p90": float(np.quantile(cal_std[:, i], 0.90)) for i, t in enumerate(TARGETS)}

    # ---- Choose top K influential parameters (for a clean HR UI) ----
    # Fit ONE Ridge model (scaled) to get standardized coefficients, then rank features by sum(|coef|) across targets.
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=SEED))
    ])
    ridge.fit(X, y)
    coef = ridge.named_steps["ridge"].coef_  # shape (n_targets, n_features)
    importance = np.sum(np.abs(coef), axis=0)  # aggregate across targets
    top_idx = np.argsort(importance)[::-1][:TOP_K]
    slider_features = [feature_cols[i] for i in top_idx]

    ui_config = {
        "feature_cols": feature_cols,
        "slider_features": slider_features,  # 👈 only these will appear as sliders
        "targets": TARGETS,
        "baseline": {k: float(baseline[k]) for k in feature_cols},
        "slider_min": {k: float(q05[k]) for k in feature_cols},
        "slider_max": {k: float(q95[k]) for k in feature_cols},
        "baseline_outputs": baseline_outputs,
        "baseline_uncertainty": baseline_unc,
        "uncertainty_thresholds": thresholds,
        "notes": {
            "baseline_definition": "Dataset mean of geometry parameters",
            "slider_limits": "5th–95th percentile of dataset",
            "threshold_definition": "90th percentile of ensemble std on a calibration split",
            "feature_ranking": "Top K by sum(|standardized Ridge coefficients|) across cd/cl/clf/clr"
        }
    }

    outpath = MODELS_DIR / "ui_config_4targets.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(ui_config, f, indent=2)

    print("Wrote:", outpath)
    print("Top slider features:", slider_features)
    print("p90 thresholds:", thresholds)

if __name__ == "__main__":
    main()
