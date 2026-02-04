import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import load_and_merge

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl", "clf", "clr"]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    df = load_and_merge(str(GEO), str(FORCES))

    # Features = geometry parameters only
    EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]  # drop cs completely
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols]
    y = df[TARGETS]

    # 1) Create a FINAL test set we will not touch for model selection
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2) Cross-validation on trainval only
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # sklearn cross_validate expects scorers where "higher is better", so RMSE is negative
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }

    cv_results = cross_validate(
        model, X_trainval, y_trainval,
        cv=cv, scoring=scoring, return_train_score=False
    )

    cv_summary = {
        "cv_rmse_mean": -np.mean(cv_results["test_rmse"]),
        "cv_rmse_std":  np.std(-cv_results["test_rmse"]),
        "cv_mae_mean":  -np.mean(cv_results["test_mae"]),
        "cv_mae_std":   np.std(-cv_results["test_mae"]),
        "cv_r2_mean":   np.mean(cv_results["test_r2"]),
        "cv_r2_std":    np.std(cv_results["test_r2"]),
    }

    print("=== 5-fold CV on train/val (Ridge) ===")
    for k, v in cv_summary.items():
        print(f"{k}: {v:.6f}")

    # 3) Fit on all trainval, then evaluate once on FINAL test
    model.fit(X_trainval, y_trainval)
    pred = model.predict(X_test)

    # Per-target final test metrics
    rows = []
    for i, t in enumerate(TARGETS):
        yt = y_test.iloc[:, i].values
        yp = pred[:, i]
        rows.append({
            "target": t,
            "test_MAE": mean_absolute_error(yt, yp),
            "test_RMSE": rmse(yt, yp),
            "test_R2": r2_score(yt, yp)
        })

    final_df = pd.DataFrame(rows)
    final_df.to_csv(REPORT_DIR / "metrics_final_test_ridge.csv", index=False)

    print("\n=== FINAL TEST (untouched) metrics saved to reports/metrics_final_test_ridge.csv ===")
    print(final_df)

if __name__ == "__main__":
    main()
