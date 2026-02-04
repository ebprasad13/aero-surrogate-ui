import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

MODELS_DIR = Path("models")

st.set_page_config(page_title="Aero Surrogate Demo", layout="wide")

@st.cache_resource
def load_models_and_config():
    with open(MODELS_DIR / "ui_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    models = []
    # assumes 30 models named ridge_ensemble_00..29
    for m in range(30):
        models.append(joblib.load(MODELS_DIR / f"ridge_ensemble_{m:02d}.joblib"))
    return cfg, models

def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def fmt_delta(name, new, base, better_when_lower=False):
    if base == 0:
        return f"{name}: {new:.6f} (baseline {base:.6f})"
    pct = (new - base) / abs(base) * 100.0
    direction = "increased" if pct > 0 else "decreased"
    msg = f"{name} has {direction} by {abs(pct):.2f}%"
    if better_when_lower:
        msg += " (lower is better)"
    return msg

cfg, models = load_models_and_config()
feature_cols = cfg["feature_cols"]
baseline = cfg["baseline"]
smin = cfg["slider_min"]
smax = cfg["slider_max"]
thr = cfg["uncertainty_thresholds"]
baseline_outputs = cfg["baseline_outputs"]

st.title("Aerodynamic Surrogate Model (DrivAerML) — Interactive Demo")

st.write(
    "Move sliders to change geometry parameters relative to the baseline (dataset mean). "
    "Click **Compute** to predict aerodynamic coefficients."
)

with st.sidebar:
    st.header("Geometry sliders (baseline centered)")
    st.caption("Baseline = 0 offset from reference. Values shown are the absolute parameter values.")

    params = {}
    for col in feature_cols:
        base_val = baseline[col]
        # slider is offset around baseline, clamped to percentile range
        left = smin[col] - base_val
        right = smax[col] - base_val
        offset = st.slider(col, float(left), float(right), 0.0, 0.001)
        params[col] = base_val + offset

    compute = st.button("Compute", type="primary")

if compute:
    X = pd.DataFrame([params])[feature_cols]
    mean, std = ensemble_predict(models, X)

    # If ensemble outputs only cd/cl
    cd = float(mean[0, 0])
    cl = float(mean[0, 1])
    cd_u = float(std[0, 0])
    cl_u = float(std[0, 1])

    base_cd = float(baseline_outputs.get("cd", np.nan))
    base_cl = float(baseline_outputs.get("cl", np.nan))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predictions")
        st.metric("cd", f"{cd:.6f}", delta=f"{(cd-base_cd):+.6f}")
        st.metric("cl", f"{cl:.6f}", delta=f"{(cl-base_cl):+.6f}")

        st.subheader("Plain-English change vs baseline")
        st.write(fmt_delta("Drag (cd)", cd, base_cd, better_when_lower=True))
        st.write(fmt_delta("Lift (cl)", cl, base_cl, better_when_lower=False))

    with col2:
        st.subheader("Reliability / uncertainty")
        st.write(f"cd uncertainty (std): **{cd_u:.6f}**")
        st.write(f"cl uncertainty (std): **{cl_u:.6f}**")

        cd_flag = cd_u > thr["cd_std_p90"]
        cl_flag = cl_u > thr["cl_std_p90"]

        if cd_flag or cl_flag:
            st.warning(
                "Low confidence for at least one output (uncertainty in top 10% of calibration set). "
                "Recommendation: verify with CFD / higher fidelity."
            )
        else:
            st.success(
                "High confidence region (uncertainty below the 90th percentile thresholds)."
            )

        st.caption(
            "Note: 'uncertainty' here is ensemble disagreement (lower is better). "
            "Thresholds are the 90th percentile from a calibration split."
        )

else:
    st.info("Set sliders and click **Compute** to run the surrogate.")
