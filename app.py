import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

MODELS_DIR = Path("models")

st.set_page_config(page_title="Aero Surrogate Demo", layout="wide")

@st.cache_resource
def load_models_and_config():
    with open(MODELS_DIR / "ui_config_4targets.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    models = []
    for m in range(30):
        models.append(joblib.load(MODELS_DIR / f"ridge4_ensemble_{m:02d}.joblib"))
    return cfg, models

def ensemble_predict(models, X):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def pct_change(new, base):
    if base == 0:
        return np.nan
    return (new - base) / abs(base) * 100.0

def reset_sliders(feature_cols):
    for c in feature_cols:
        st.session_state[f"off_{c}"] = 0.0

cfg, models = load_models_and_config()
feature_cols = cfg["feature_cols"]
targets = cfg["targets"]
baseline = cfg["baseline"]
smin = cfg["slider_min"]
smax = cfg["slider_max"]
thr = cfg.get("uncertainty_thresholds", {})
baseline_outputs = cfg["baseline_outputs"]

st.title("Aerodynamic Surrogate Model (DrivAerML) — Interactive Demo")
st.write(
    "Sliders are centered on **0 offset** (baseline = dataset mean geometry). "
    "Click **Compute** to predict coefficients and compare vs baseline."
)

# Sidebar controls
with st.sidebar:
    st.header("Geometry sliders (0 = baseline)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset to baseline"):
            reset_sliders(feature_cols)
            st.rerun()
    with c2:
        compute = st.button("Compute", type="primary")

    st.caption("Values shown below are absolute parameter values (baseline + offset).")

    params = {}
    for col in feature_cols:
        base_val = baseline[col]
        left = smin[col] - base_val
        right = smax[col] - base_val

        key = f"off_{col}"
        if key not in st.session_state:
            st.session_state[key] = 0.0

        offset = st.slider(col, float(left), float(right), float(st.session_state[key]), 0.001, key=key)
        params[col] = base_val + offset

if compute:
    X = pd.DataFrame([params])[feature_cols]
    mean, std = ensemble_predict(models, X)

    pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
    unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

    # deltas vs baseline
    delta = {t: pred[t] - baseline_outputs[t] for t in targets}
    delta_pct = {t: pct_change(pred[t], baseline_outputs[t]) for t in targets}

    colA, colB = st.columns([1.1, 0.9])

    with colA:
        st.subheader("Predictions vs baseline")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta['cd']:+.6f}")
        m2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta['cl']:+.6f}")
        m3.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta['clf']:+.6f}")
        m4.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta['clr']:+.6f}")

        st.subheader("Plain-English summary")
        # Drag: lower is better (typically)
        cd_msg = f"Drag (cd) has {'decreased' if delta_pct['cd'] < 0 else 'increased'} by {abs(delta_pct['cd']):.2f}% (lower is typically better)."
        cl_msg = f"Lift (cl) has {'decreased' if delta_pct['cl'] < 0 else 'increased'} by {abs(delta_pct['cl']):.2f}% vs baseline."
        st.write(cd_msg)
        st.write(cl_msg)

        # Simple plot: baseline vs predicted bars
        st.subheader("Baseline vs predicted (bar chart)")
        labels = ["cd", "cl", "clf", "clr"]
        base_vals = [baseline_outputs[t] for t in labels]
        pred_vals = [pred[t] for t in labels]

        fig = plt.figure()
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, base_vals, width, label="baseline")
        plt.bar(x + width/2, pred_vals, width, label="predicted")
        plt.xticks(x, labels)
        plt.ylabel("Coefficient value")
        plt.title("Coefficients: baseline vs predicted")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # Lift distribution plot (clf vs clr)
        st.subheader("Front vs rear lift comparison")
        fig2 = plt.figure()
        plt.bar(["clf (front)", "clr (rear)"], [pred["clf"], pred["clr"]])
        plt.ylabel("Coefficient value")
        plt.title("Predicted lift distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    with colB:
        st.subheader("Uncertainty / reliability")
        st.write("Uncertainty here is **ensemble disagreement (std)** — lower is better.")

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("cd std", f"{unc['cd']:.6f}")
        u2.metric("cl std", f"{unc['cl']:.6f}")
        u3.metric("clf std", f"{unc['clf']:.6f}")
        u4.metric("clr std", f"{unc['clr']:.6f}")

        # Apply your p90 thresholds only for cd/cl (we haven't calibrated clf/clr thresholds yet)
        cd_flag = ("cd_std_p90" in thr) and (unc["cd"] > float(thr["cd_std_p90"]))
        cl_flag = ("cl_std_p90" in thr) and (unc["cl"] > float(thr["cl_std_p90"]))

        if cd_flag or cl_flag:
            st.warning(
                "Low confidence for at least one output (cd/cl uncertainty in top 10% of calibration set). "
                "Recommendation: verify with CFD / higher fidelity."
            )
        else:
            st.success("High confidence region (cd/cl uncertainty below p90 thresholds).")

        st.caption(
            "Note: the p90 thresholds shown are currently applied to cd/cl only. "
            "If you want, we can calibrate thresholds for clf/clr as well."
        )
else:
    st.info("Move sliders (0 = baseline) and click **Compute**.")

if not (MODELS_DIR / "ui_config_4targets.json").exists():
    st.error("Missing models/ui_config_4targets.json. Make sure it's committed to the repo.")
    st.stop()

