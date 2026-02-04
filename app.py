import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Settings
# ----------------------------
MODELS_DIR = Path("models")
N_MODELS = 30

st.set_page_config(page_title="DrivAerML Aero Surrogate", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def ensemble_predict(models, X: pd.DataFrame):
    preds = np.stack([m.predict(X) for m in models], axis=0)  # (n_models, n_samples, n_targets)
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
        st.error("Missing file: `models/ui_config_4targets.json`. Upload/commit it to the repo and redeploy.")
        st.stop()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_paths = [MODELS_DIR / f"ridge4_ensemble_{m:02d}.joblib" for m in range(N_MODELS)]
    missing = [str(p) for p in model_paths if not p.exists()]
    if missing:
        st.error("Some model files are missing from `models/`. Upload/commit these and redeploy:")
        st.code("\n".join(missing))
        st.stop()

    models = [joblib.load(p) for p in model_paths]
    return cfg, models

def pretty_param_name(s: str) -> str:
    return s.replace("_", " ")

# ----------------------------
# Load assets
# ----------------------------
cfg, models = load_models_and_config()

feature_cols = cfg["feature_cols"]
slider_features = cfg.get("slider_features", feature_cols)
targets = cfg["targets"]

baseline = cfg["baseline"]
smin = cfg["slider_min"]
smax = cfg["slider_max"]

thr = cfg["uncertainty_thresholds"]  # should include *_std_p90 for all 4 targets
baseline_outputs = cfg["baseline_outputs"]

# ----------------------------
# Header
# ----------------------------
st.title("DrivAerML Aero Surrogate — Interactive Coefficient Predictor")

st.markdown(
    """
**What this is:** A fast *surrogate model* trained on the **DrivAerML** dataset (500 parametrically-morphed DrivAer notchback geometries with high-fidelity CFD force coefficients).

**Model:** An **ensemble of Ridge Regression models** (linear model with L2 regularisation), predicting **cd, cl, clf, clr** from geometry parameters.

**How to use it:** Sliders adjust key geometry parameters around a reference (“baseline”) geometry (here: the **dataset mean**). Click **Compute** to see predicted coefficients and a comparison vs baseline.
"""
)

with st.expander("Baseline and slider limits (quick note)", expanded=False):
    st.markdown(
        """
- **Baseline geometry:** the mean of the dataset parameters (a neutral reference point).
- **Slider limits:** 5th–95th percentile of the dataset for each parameter (helps avoid extreme, out-of-distribution inputs).
- **Interpretation:** predictions are most reliable for *interpolation* within the dataset coverage.
"""
    )

# ----------------------------
# Sidebar controls (top K sliders only)
# ----------------------------
with st.sidebar:
    st.header("Controls")
    st.caption("To keep this demo tidy, you’re only seeing the most influential parameters (top 6–8).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset sliders"):
            reset_sliders(slider_features)
            st.rerun()
    with c2:
        compute = st.button("Compute", type="primary")

    st.divider()
    st.subheader("Key geometry sliders")
    st.caption("Sliders are offsets from baseline. The app uses baseline values for all non-shown parameters.")

    params = {}
    for col in slider_features:
        base_val = baseline[col]
        left = float(smin[col] - base_val)
        right = float(smax[col] - base_val)

        key = f"off_{col}"
        if key not in st.session_state:
            st.session_state[key] = 0.0

        offset = st.slider(pretty_param_name(col), left, right, float(st.session_state[key]), 0.001, key=key)
        params[col] = base_val + offset

# Fill non-slider parameters with baseline so we always pass full feature vector to the model
full_params = dict(baseline)
full_params.update(params)

if not compute:
    st.info("Adjust the sliders in the sidebar, then click **Compute**.")
    st.stop()

# ----------------------------
# Predict
# ----------------------------
X = pd.DataFrame([full_params])[feature_cols]
mean, std = ensemble_predict(models, X)

pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

delta = {t: pred[t] - baseline_outputs[t] for t in targets}
delta_pct = {t: pct_change(pred[t], baseline_outputs[t]) for t in targets}

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta['cd']:+.6f}")
    m2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta['cl']:+.6f}")
    m3.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta['clf']:+.6f}")
    m4.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta['clr']:+.6f}")

    st.subheader("Engineering summary (vs baseline)")
    drag_msg = (
        f"Drag (cd) has {'gone down' if delta_pct['cd'] < 0 else 'gone up'} by "
        f"{abs(delta_pct['cd']):.2f}% compared with baseline "
        f"({'usually a good thing' if delta_pct['cd'] < 0 else 'usually not ideal'})."
    )
    lift_msg = (
        f"Lift (cl) has {'gone down' if delta_pct['cl'] < 0 else 'gone up'} by "
        f"{abs(delta_pct['cl']):.2f}% compared with baseline."
    )
    st.write(drag_msg)
    st.write(lift_msg)

    st.subheader("Baseline vs predicted (table)")
    tbl = pd.DataFrame(
        {
            "baseline": [baseline_outputs[t] for t in targets],
            "predicted": [pred[t] for t in targets],
            "delta": [delta[t] for t in targets],
            "delta_%": [delta_pct[t] for t in targets],
        },
        index=targets,
    )
    st.dataframe(
        tbl.style.format(
            {
                "baseline": "{:.6f}",
                "predicted": "{:.6f}",
                "delta": "{:+.6f}",
                "delta_%": "{:+.2f}",
            }
        )
    )

    p1, p2 = st.columns(2)

    with p1:
        st.subheader("Change relative to baseline (%)")
        labels = ["cd", "cl", "clf", "clr"]
        vals = [delta_pct[t] for t in labels]

        fig = plt.figure()
        x = np.arange(len(labels))
        plt.bar(x, vals)
        plt.axhline(0, linewidth=1)
        plt.xticks(x, labels)
        plt.ylabel("% change vs baseline")
        plt.title("Predicted change vs baseline")

        for i, v in enumerate(vals):
            # avoid NaN label
            if not np.isnan(v):
                plt.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top")

        plt.tight_layout()
        st.pyplot(fig)

    with p2:
        st.subheader("Lift distribution (front vs rear)")
        fig2 = plt.figure()
        plt.bar(["clf (front)", "clr (rear)"], [pred["clf"], pred["clr"]])
        plt.ylabel("Coefficient value")
        plt.title("Predicted lift split")
        plt.tight_layout()
        st.pyplot(fig2)

with right:
    st.subheader("Uncertainty / reliability")

    with st.expander("What do “std” and “p90” mean?", expanded=False):
        st.markdown(
            """
- **std (standard deviation):** in this app it’s the *disagreement* between the models in the ensemble.  
  If the models broadly agree, the std is small — that’s usually a sign the prediction is more reliable in that region.

- **p90:** the 90th percentile of uncertainty on a held-out calibration split.  
  If the std is above p90, the prediction falls into the most uncertain ~10% of cases, so it’s flagged as **low confidence**.

This is a practical reliability check — not a guaranteed probability.
"""
        )

    def show_row(t):
        thr_val = float(thr.get(f"{t}_std_p90", np.nan))
        if np.isnan(thr_val):
            st.write(f"**{t}**: std `{unc[t]:.2e}` (no threshold available)")
            return False

        flagged = unc[t] > thr_val
        status = "⚠️ Low confidence" if flagged else "✅ High confidence"
        st.write(f"**{t}**: std `{unc[t]:.2e}` vs p90 `{thr_val:.2e}` → **{status}**")
        return flagged

    flags = [show_row(t) for t in ["cd", "cl", "clf", "clr"]]

    st.divider()

    if any(flags):
        st.warning(
            "One or more outputs are in a higher-uncertainty region. "
            "In a real workflow, you’d normally verify these cases with CFD / higher fidelity."
        )
    else:
        st.success("All outputs sit in a lower-uncertainty region (below the p90 thresholds).")

    st.caption(
        "Tip: The uncertainty tends to be lowest near the baseline and higher near the edge of the dataset coverage."
    )
