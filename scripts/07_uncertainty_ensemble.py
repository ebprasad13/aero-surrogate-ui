import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from src.utils import load_and_merge

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl"]
EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]

N_MODELS = 30
RIDGE_ALPHA = 1.0
RANDOM_SEED = 42

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    df = load_and_merge(str(GEO), str(FORCES))

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].reset_index(drop=True)
    y = df[TARGETS].reset_index(drop=True)

    # Holdout test set (same philosophy as before)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    rng = np.random.default_rng(RANDOM_SEED)

    preds = np.zeros((N_MODELS, len(X_test), len(TARGETS)), dtype=np.float64)

    for m in range(N_MODELS):
        # Bootstrap sample indices from trainval
        idx = rng.integers(0, len(X_trainval), size=len(X_trainval))
        Xb = X_trainval.iloc[idx]
        yb = y_trainval.iloc[idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED + m))
        ])

        model.fit(Xb, yb)
        preds[m] = model.predict(X_test)

    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)  # uncertainty proxy

    # Build results table
    out = pd.DataFrame({
        "cd_true": y_test["cd"].values,
        "cl_true": y_test["cl"].values,
        "cd_pred_mean": pred_mean[:, 0],
        "cl_pred_mean": pred_mean[:, 1],
        "cd_pred_std": pred_std[:, 0],
        "cl_pred_std": pred_std[:, 1],
    })
    out["cd_abs_err"] = (out["cd_pred_mean"] - out["cd_true"]).abs()
    out["cl_abs_err"] = (out["cl_pred_mean"] - out["cl_true"]).abs()

    out.to_csv(REPORT_DIR / "uncertainty_ensemble_results.csv", index=False)
    print("Saved results to reports/uncertainty_ensemble_results.csv")

    # Headline metrics using mean prediction
    print("\n=== Ensemble mean performance on test set ===")
    print(f"cd RMSE: {rmse(out['cd_true'], out['cd_pred_mean']):.6f}")
    print(f"cl RMSE: {rmse(out['cl_true'], out['cl_pred_mean']):.6f}")

    # Plot uncertainty vs abs error
    for t in ["cd", "cl"]:
        plt.figure()
        plt.scatter(out[f"{t}_pred_std"], out[f"{t}_abs_err"], s=18)
        plt.xlabel(f"{t} predicted std (uncertainty)")
        plt.ylabel(f"{t} absolute error")
        plt.title(f"Uncertainty vs error ({t}) - ensemble Ridge")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"uncertainty_vs_error_{t}.png", dpi=160)
        plt.close()

    # Optional: correlation numbers (quick sanity)
    cd_corr = np.corrcoef(out["cd_pred_std"], out["cd_abs_err"])[0, 1]
    cl_corr = np.corrcoef(out["cl_pred_std"], out["cl_abs_err"])[0, 1]
    print("\nCorrelation (uncertainty, abs error):")
    print(f"cd: {cd_corr:.3f}")
    print(f"cl: {cl_corr:.3f}")

    # Save top uncertain points (useful for narrative)
    top_cd = out.sort_values("cd_pred_std", ascending=False).head(10)
    top_cl = out.sort_values("cl_pred_std", ascending=False).head(10)

    top_cd.to_csv(REPORT_DIR / "top10_uncertain_cd.csv", index=False)
    top_cl.to_csv(REPORT_DIR / "top10_uncertain_cl.csv", index=False)
    print("\nSaved:")
    print("- reports/top10_uncertain_cd.csv")
    print("- reports/top10_uncertain_cl.csv")
    print("- reports/figures/uncertainty_vs_error_cd.png")
    print("- reports/figures/uncertainty_vs_error_cl.png")

if __name__ == "__main__":
    main()
