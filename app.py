import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

MODELS_DIR = Path("models")
N_MODELS = 30

st.set_page_config(page_title="Aero Surrogate Demo", layout="wide")

def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def pct_change(new, base):
    if base == 0:
        return np.nan
    return (new - base) / abs(base) * 100.0

def reset_sliders(slider_features):
    for c in slider_features:
        st.session_state[f"off_{c}"] = 0.0

@st.cache_resource
def load_models_and_config():
    cfg_path = MODELS_DIR / "ui_config_4targets.json"
    if not cfg_path.exists():
        st.error("Missing file: models/ui_config_4targets.json (commit it to the repo).")
        st.stop()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_paths = [MODELS_DIR / f"ridge4_ensemble_{m:02d}.joblib" for m in range(N_MODELS)]
    missing = [str(p) for p in model_paths if not p.exists()]
    if missing:
        st.error("Missing model files in /models. Upload these to the repo:")
        st.code("\n".join(missing))
        st.stop()

    models = [joblib.load(p) for p in model_paths]
    return cfg, models

cfg, models = load_models_and_config()

feature_cols = cfg["feature_cols"]
slider_features = cfg.get("slider_features", feature_cols)  # fallback
targets = cfg["targets"]
baseline = cfg["baseline"]
smin = cfg["slider_min"]
smax = cfg["slider_max"]
thr = cfg["uncertainty_thresholds"]
baseline_outputs = cfg["baseline_outputs"]

st.title("DrivAerML Aero Surrogate — Interactive Coefficient Predictor")

st.markdown(
    """
**What you’re seeing:** a lightweight *surrogate model* trained on the **DrivAerML** dataset  
(500 parametrically-morphed DrivAer notchback geometries with high-fidelity CFD force coefficients).

**Model:** an **ensemble of Ridge Regression models** (linear model with L2 regularisation) trained on
geometry parameters to predict **cd, cl, clf, clr**.

**How to use:** sliders apply an offset around a reference (“baseline”) geometry (here: **dataset mean**).
Click **Compute** to predict coefficients and compare against baseline.
"""
)

# Sidebar sliders (ONLY top K)
with st.sidebar:
    st.header("Key geometry sliders (0 = baseline)")
    st.caption("Only the most influential parameters are shown to keep this UI clean.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset sliders"):
            reset_sliders(slider_features)
            st.rerun()  # ✅ correct in Streamlit 1.5x+
    with c2:
        compute = st.button("Compute", type="primary")

    params = {}
    for col in slider_features:
        base_val = baseline[col]
        left = smin[col] - base_val
        right = smax[col] - base_val

        key = f"off_{col}"
        if key not in st.session_state:
            st.session_state[key] = 0.0

        offset = st.slider(col, float(left), float(right), float(st.session_state[key]), 0.001, key=key)
        params[col] = base_val + offset

# Fill non-slider features with baseline (so model always gets full feature vector)
full_params = dict(baseline)
full_params.update(params)

if not compute:
    st.info("Adjust sliders and click **Compute**.")
    st.stop()

X = pd.DataFrame([full_params])[feature_cols]
mean, std = ensemble_predict(models, X)

pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

delta = {t: pred[t] - baseline_outputs[t] for t in targets}
delta_pct = {t: pct_change(pred[t], baseline_outputs[t]) for t in targets}

# Layout
colA, colB = st.columns([1.15, 0.85])

with colA:
    st.subheader("Predictions vs baseline")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta['cd']:+.6f}")
    m2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta['cl']:+.6f}")
    m3.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta['clf']:+.6f}")
    m4.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta['clr']:+.6f}")

    st.subheader("Plain-English change vs baseline")
    drag_msg = f"Drag (cd) has {'decreased' if delta_pct['cd'] < 0 else 'increased'} by {abs(delta_pct['cd']):.2f}% (lower is typically better)."
    lift_msg = f"Lift (cl) has {'decreased' if delta_pct['cl'] < 0 else 'increased'} by {abs(delta_pct['cl']):.2f}% vs baseline."
    st.write(drag_msg)
    st.write(lift_msg)

    # Plot 1: % change bars (nice and simple)
    st.subheader("Change relative to baseline (%)")
    labels = ["cd", "cl", "clf", "clr"]
    vals = [delta_pct[t] for t in labels]

    fig = plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel("% change vs baseline")
    plt.title("Predicted change relative to baseline")
    plt.tight_layout()
    st.pyplot(fig)

    # Plot 2: front vs rear lift distribution (absolute)
    st.subheader("Lift distribution (front vs rear)")
    fig2 = plt.figure()
    plt.bar(["clf (front)", "clr (rear)"], [pred["clf"], pred["clr"]])
    plt.ylabel("Coefficient value")
    plt.title("Predicted lift split")
    plt.tight_layout()
    st.pyplot(fig2)

with colB:
    st.subheader("Uncertainty / reliability")
    st.write("Uncertainty = ensemble disagreement (std). Lower is better.")

    def row(t):
        t_thr = float(thr.get(f"{t}_std_p90"))
        flag = unc[t] > t_thr
        status = "⚠️ Low" if flag else "✅ High"
        st.write(f"**{t}**: std `{unc[t]:.2e}` vs p90 `{t_thr:.2e}` → **{status}**")
        return flag

    flags = [row(t) for t in ["cd", "cl", "clf", "clr"]]

    if any(flags):
        st.warning("Low confidence for at least one output (uncertainty above p90). Recommend CFD / higher-fidelity verification.")
    else:
        st.success("High confidence region (all uncertainties below p90 thresholds).")

    st.caption(
        "The p90 thresholds are computed from ensemble uncertainty on a held-out calibration split. "
        "This is a reliability gate, not a guaranteed probability."
    )
