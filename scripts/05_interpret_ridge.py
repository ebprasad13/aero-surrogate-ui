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

from src.utils import load_and_merge

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl", "clf", "clr"]

def main():
    df = load_and_merge(str(GEO), str(FORCES))

    EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]  # drop cs completely
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols]
    y = df[TARGETS]

    # Use same holdout logic as before: interpret coefficients using train/val portion only
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])

    model.fit(X_trainval, y_trainval)

    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]

    # Ridge supports multi-target: coef_ shape = (n_targets, n_features)
    coef_std = ridge.coef_  # coefficients in standardized feature space
    intercept_std = ridge.intercept_

    # Convert to "original feature units" coefficients (optional, sometimes less interpretable)
    # If x_std = (x - mean)/scale, then:
    # y = intercept_std + sum_j coef_std_j * x_std_j
    #   = intercept_unscaled + sum_j coef_unscaled_j * x_j
    scale = scaler.scale_
    mean = scaler.mean_

    coef_unscaled = coef_std / scale
    intercept_unscaled = intercept_std - np.sum((mean / scale) * coef_std, axis=1)

    # Save coefficient tables
    std_df = pd.DataFrame(coef_std.T, index=feature_cols, columns=[f"coef_std_{t}" for t in TARGETS])
    unscaled_df = pd.DataFrame(coef_unscaled.T, index=feature_cols, columns=[f"coef_unscaled_{t}" for t in TARGETS])

    std_df.to_csv(REPORT_DIR / "ridge_coefficients_standardized.csv")
    unscaled_df.to_csv(REPORT_DIR / "ridge_coefficients_unscaled.csv")

    # Also save intercepts
    pd.DataFrame({
        "target": TARGETS,
        "intercept_std": intercept_std,
        "intercept_unscaled": intercept_unscaled
    }).to_csv(REPORT_DIR / "ridge_intercepts.csv", index=False)

    print("Saved:")
    print("- reports/ridge_coefficients_standardized.csv")
    print("- reports/ridge_coefficients_unscaled.csv")
    print("- reports/ridge_intercepts.csv")

    # Plot: top features by absolute standardized coefficient for each target
    for i, t in enumerate(TARGETS):
        coefs = std_df[f"coef_std_{t}"].copy()
        top = coefs.abs().sort_values(ascending=False).head(10).index
        plot_series = coefs.loc[top].sort_values()

        plt.figure()
        plt.barh(plot_series.index, plot_series.values)
        plt.title(f"Ridge standardized coefficients (top 10) for {t}")
        plt.xlabel("Effect per +1 std dev of feature")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"ridge_top10_coeffs_{t}.png", dpi=160)
        plt.close()

    print("Saved coefficient plots to reports/figures/ (ridge_top10_coeffs_*.png)")

if __name__ == "__main__":
    main()
