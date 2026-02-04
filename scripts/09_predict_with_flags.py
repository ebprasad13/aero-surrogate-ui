import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path("models")

def main():
    # Input file you provide
    input_csv = Path("data/new_geometries.csv")  # change if you want
    output_csv = Path("reports/predictions_with_flags.csv")

    with open(MODELS_DIR / "ensemble_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    targets = meta["targets"]
    thr = meta["thresholds"]
    model_files = meta["model_files"]

    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in input CSV: {missing}")

    X = df[feature_cols].copy()

    preds = []
    for mf in model_files:
        model = joblib.load(mf)
        preds.append(model.predict(X))

    preds = np.stack(preds, axis=0)  # (n_models, n_samples, n_targets)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    out = df.copy()
    out[f"{targets[0]}_pred"] = pred_mean[:, 0]
    out[f"{targets[1]}_pred"] = pred_mean[:, 1]
    out[f"{targets[0]}_uncertainty"] = pred_std[:, 0]
    out[f"{targets[1]}_uncertainty"] = pred_std[:, 1]

    out["flag_cd"] = out["cd_uncertainty"] > thr["cd_std_p90"]
    out["flag_cl"] = out["cl_uncertainty"] > thr["cl_std_p90"]

    out.to_csv(output_csv, index=False)
    print("Saved:", output_csv)
    print("Thresholds used:", thr)

if __name__ == "__main__":
    main()
