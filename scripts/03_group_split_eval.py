"""
Group split demo for DrivAerML summary CSVs.

Important:
- With the *_all.csv summary files you currently downloaded, there is ONE row per run (geometry).
  That means a 'group split' by Run is effectively the same as a normal split, because groups don't repeat.

- Group splits become critical if you later use per-run folders (many samples per geometry / condition).
  Then you must keep all samples for a given geometry together to test on truly unseen geometries.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from src.utils import load_and_merge

DATA_DIR = Path("data")
GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

TARGETS = ["cd", "cl"]

def main():
    df = load_and_merge(str(GEO), str(FORCES))
    feature_cols = [c for c in df.columns if c not in (["Run"] + TARGETS)]
    X = df[feature_cols]
    y = df[TARGETS]
    groups = df["Run"].values

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    (train_idx, test_idx) = next(splitter.split(X, y, groups=groups))

    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])

    rmse_cd = np.sqrt(mean_squared_error(y.iloc[test_idx]["cd"].values, pred[:, 0]))
    rmse_cl = np.sqrt(mean_squared_error(y.iloc[test_idx]["cl"].values, pred[:, 1]))

    print("Group split (by Run) RMSE:")
    print("  cd:", rmse_cd)
    print("  cl:", rmse_cl)
    print("\nNote: since there is one row per Run in the summary CSVs, this is similar to a normal split.")

if __name__ == "__main__":
    main()
