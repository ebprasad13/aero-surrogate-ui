import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd

MODELS_DIR = Path("models")

def main():
    with open(MODELS_DIR / "ensemble_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    cols = meta["feature_cols"]
    df = pd.DataFrame([{c: 0.0 for c in cols}])  # one blank row
    out = Path("data/new_geometries_template.csv")
    df.to_csv(out, index=False)
    print("Wrote template:", out)

if __name__ == "__main__":
    main()
