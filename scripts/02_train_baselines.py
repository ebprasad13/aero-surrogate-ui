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
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import load_and_merge

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl", "clf", "clr"]  # you can reduce this to ["cd","cl"] if you want

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rows = []
    for i, t in enumerate(y_test.columns):
        yt = y_test.iloc[:, i].values
        yp = pred[:, i] if pred.ndim == 2 else pred
        rows.append({
            "model": name,
            "target": t,
            "MAE": mean_absolute_error(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "R2": r2_score(yt, yp),
        })
    return pd.DataFrame(rows), pred

def scatter_plot(y_true, y_pred, target_name, outpath):
    plt.figure()
    plt.scatter(y_true, y_pred, s=18)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Predicted vs Actual: {target_name}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def main():
    df = load_and_merge(str(GEO), str(FORCES))

    # Features: all geometry params except Run
    EXCLUDE = ["Run", "cd", "cl", "clf", "clr", "cs"]  # drop cs completely
    feature_cols = [c for c in df.columns if c not in EXCLUDE]

    X = df[feature_cols]
    y = df[TARGETS]

    # Simple random split to start
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "ridge_scaled": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42))
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),
        "hist_gbdt": MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=42)
        )
    }

    all_metrics = []
    preds = {}

    for name, model in models.items():
        mdf, pred = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        all_metrics.append(mdf)
        preds[name] = pred

    metrics = pd.concat(all_metrics, ignore_index=True)
    REPORT_DIR.mkdir(exist_ok=True)
    metrics.to_csv(REPORT_DIR / "metrics_baselines.csv", index=False)
    print("Saved metrics to reports/metrics_baselines.csv")
    print(metrics.sort_values(["target","RMSE"]).head(15))

    # Save a couple of plots for the best model on cd/cl (based on avg RMSE across targets)
    avg_rmse = metrics.groupby("model")["RMSE"].mean().sort_values()
    best_model = avg_rmse.index[0]
    print(f"Best model by mean RMSE: {best_model}")

    pred = preds[best_model]
    for t in ["cd", "cl"]:
        idx = y_test.columns.tolist().index(t)
        scatter_plot(
            y_test.iloc[:, idx].values,
            pred[:, idx],
            t,
            FIG_DIR / f"pred_vs_actual_{t}_{best_model}.png"
        )

    # Residual histogram for cd
    cd_idx = y_test.columns.tolist().index("cd")
    residuals = pred[:, cd_idx] - y_test.iloc[:, cd_idx].values
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (pred - actual) for cd")
    plt.ylabel("Count")
    plt.title(f"Residuals for cd ({best_model})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"residuals_cd_{best_model}.png", dpi=160)
    plt.close()

    print("Saved plots to reports/figures/")

if __name__ == "__main__":
    main()
