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

TARGETS = ["cd", "cl", "clf", "clr"]
EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]

# Choose a few key geometry knobs to analyse
FEATURES_TO_BIN = [
    "Vehicle_Ride_Height",
    "Vehicle_Pitch",
    "Rear_Diffusor_Angle",
    "Vehicle_Width",
]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    df = load_and_merge(str(GEO), str(FORCES))

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols]
    y = df[TARGETS]

    # Same holdout split as before
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    model.fit(X_trainval, y_trainval)
    pred = model.predict(X_test)

    # Put predictions into dataframe
    pred_df = pd.DataFrame(pred, columns=[f"pred_{t}" for t in TARGETS], index=X_test.index)
    out = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    # Focus on cd and cl for the core story
    for target in ["cd", "cl"]:
        out[f"abs_err_{target}"] = (out[f"pred_{target}"] - out[target]).abs()

    summary_rows = []

    for feat in FEATURES_TO_BIN:
        if feat not in out.columns:
            print(f"Skipping {feat} (not found).")
            continue

        # Bin into quantiles so bins have similar sample counts
        out[f"{feat}_bin"] = pd.qcut(out[feat], q=6, duplicates="drop")

        for target in ["cd", "cl"]:
            grp = out.groupby(f"{feat}_bin", observed=True)

            bin_rmse = grp.apply(lambda g: rmse(g[target].values, g[f"pred_{target}"].values))
            bin_mae = grp[f"abs_err_{target}"].mean()
            bin_n = grp.size()

            bin_table = pd.DataFrame({
                "feature": feat,
                "target": target,
                "bin": bin_rmse.index.astype(str),
                "rmse": bin_rmse.values,
                "mae": bin_mae.values,
                "n": bin_n.values
            })

            summary_rows.append(bin_table)

            # Plot RMSE by bin
            plt.figure()
            plt.bar(range(len(bin_table)), bin_table["rmse"].values)
            plt.xticks(range(len(bin_table)), bin_table["bin"].values, rotation=30, ha="right")
            plt.ylabel("RMSE")
            plt.title(f"Test error by {feat} bin ({target})")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"error_by_{feat}_{target}.png", dpi=160)
            plt.close()

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(REPORT_DIR / "error_by_regime_summary.csv", index=False)
    print("Saved:")
    print("- reports/error_by_regime_summary.csv")
    print("- plots in reports/figures/error_by_*.png")

if __name__ == "__main__":
    main()
