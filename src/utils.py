import pandas as pd

def load_and_merge(geo_csv: str, forces_csv: str) -> pd.DataFrame:
    """
    Loads DrivAerML summary CSVs and merges them by run id.

    Notes:
    - geo_parameters_all.csv columns sometimes include leading spaces -> we strip them.
    - geo uses column 'Run' (capital R). forces uses 'run' (lowercase).
    - forces file is missing some runs; merge is inner to keep only common runs.
    """
    geo = pd.read_csv(geo_csv)
    forces = pd.read_csv(forces_csv)

    geo.columns = [c.strip() for c in geo.columns]
    forces.columns = [c.strip() for c in forces.columns]

    # Ensure integer run ids
    geo["Run"] = geo["Run"].astype(int)
    forces["run"] = forces["run"].astype(int)

    df = geo.merge(forces, left_on="Run", right_on="run", how="inner")

    # Optional: drop duplicate run column
    df = df.drop(columns=["run"])

    return df
