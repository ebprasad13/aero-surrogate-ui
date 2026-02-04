import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
GEO = DATA_DIR / "geo_parameters_all.csv"
FORCES = DATA_DIR / "force_mom_constref_all.csv"

def main():
    geo = pd.read_csv(GEO)
    forces = pd.read_csv(FORCES)

    # strip whitespace in column names (geo has leading spaces)
    geo.columns = [c.strip() for c in geo.columns]
    forces.columns = [c.strip() for c in forces.columns]

    print("=== geo_parameters_all.csv ===")
    print("shape:", geo.shape)
    print("columns:", geo.columns.tolist())
    print(geo.head(3), "\n")

    print("=== force_mom_constref_all.csv ===")
    print("shape:", forces.shape)
    print("columns:", forces.columns.tolist())
    print(forces.head(3), "\n")

    # runs present/missing
    geo_runs = set(geo["Run"].astype(int))
    force_runs = set(forces["run"].astype(int))
    missing = sorted(list(geo_runs - force_runs))
    print(f"Runs in geo but missing in forces: {len(missing)}")
    if missing:
        print(missing)

if __name__ == "__main__":
    main()
