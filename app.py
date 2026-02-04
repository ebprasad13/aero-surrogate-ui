import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ----------------------------
# Settings
# ----------------------------
MODELS_DIR = Path("models")
N_MODELS = 30

st.set_page_config(page_title="DrivAerML aero surrogate — demo", layout="wide")

# ----------------------------
# Styling (keep header so sidebar can always be restored)
# ----------------------------
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Keep the “show sidebar” control visible when sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"] {
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  position: fixed !important;
  top: 0.9rem !important;
  left: 0.9rem !important;
  z-index: 10000 !important;
}

/* Page background */
.stApp {
  background: radial-gradient(1200px 800px at 20% 10%, #162033 0%, #0b0f17 55%, #070a10 100%);
  color: #e8eefc;
}

/* Container width/padding */
.block-container { max-width: 1200px; padding-top: 1.6rem; }

/* Card styling */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.04);
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: 0.55rem 0.9rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}
.stButton>button:hover {
  border: 1px solid rgba(156,194,255,0.7);
  transform: translateY(-1px);
  transition: 120ms ease;
}

/* Filled red primary button */
button[kind="primary"] {
  background: #d32f2f !important;
  border: 1px solid #d32f2f !important;
  color: white !important;
}
button[kind="primary"]:hover {
  background: #b71c1c !important;
  border: 1px solid #b71c1c !important;
}

/* Metric styling */
[data-testid="stMetricValue"] { font-size: 1.45rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Units (angles in degrees; lengths in mm)
# ----------------------------
PARAM_UNITS = {
    "Vehicle_Length": "mm",
    "Vehicle_Width": "mm",
    "Vehicle_Height": "mm",
    "Front_Overhang": "mm",
    "Rear_Overhang": "mm",
    "Vehicle_Ride_Height": "mm",
    "Hood_Angle": "deg",
    "Approach_Angle": "deg",
    "Windscreen_Angle": "deg",
    "Backlight_Angle": "deg",
    "Rear_Diffusor_Angle": "deg",
    "Vehicle_Pitch": "deg",
    "Front_Planview": "mm",
    "Greenhouse_Tapering": "mm",
    "Decklid_Height": "mm",
    "Rearend_tapering": "mm",
}

# ----------------------------
# Helpers
# ----------------------------
def pretty_param_name(s: str) -> str:
    return s.replace("_", " ")

def fmt_with_unit(value: float, unit: str) -> str:
    if unit == "deg":
        return f"{value:+.2f}°"
    if unit == "mm":
        return f"{value:+.1f} mm"
    return f"{value:+.4f}"

def pct_change(new, base):
    if base == 0:
        return np.nan
    return (new - base) / abs(base) * 100.0

def ensemble_predict(models, X: pd.DataFrame):
    preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

def reset_sliders(slider_features):
    for c in slider_features:
        st.session_state[f"off_{c}"] = 0.0

@st.cache_resource
def load_models_and_config():
    cfg_path = MODELS_DIR / "ui_config_4targets.json"
    if not cfg_path.exists():
        st.error("Missing file: `models/ui_config_4targets.json`. Commit it to the repo and redeploy.")
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

# ----------------------------
# Load assets
# ----------------------------
cfg, models = load_models_and_config()

feature_cols = cfg["feature_cols"]
slider_features = cfg.get("slider_features", feature_cols)  # top 8
targets = cfg["targets"]  # ["cd","cl","clf","clr"]

baseline = cfg["baseline"]
smin = cfg["slider_min"]
smax = cfg["slider_max"]
thr = cfg["uncertainty_thresholds"]
baseline_outputs = cfg["baseline_outputs"]

# Optional: show influential list if present in config; else derive from slider list
influential_hint = cfg.get("influential_hint", slider_features)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("DrivAerML aero surrogate — demo")
st.markdown(
    """
**What this demo does**
- Predicts **cd, cl, clf, clr** from a compact set of geometry parameters.

**How it works**
- Uses an **ensemble of Ridge regression pipelines** trained on **DrivAerML** (500 CFD-tested DrivAer variants).

**How to read the sliders**
- Slider values are **geometry deltas (Δ)** relative to the baseline DrivAer shape.
"""
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sidebar controls (only)
# ----------------------------
with st.sidebar:
    st.header("Controls")
    st.caption("To keep the demo tidy, you’re only seeing the top 8 most influential parameters.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset sliders"):
            reset_sliders(slider_features)
            st.rerun()
    with c2:
        compute = st.button("Compute", type="primary")

    st.divider()
    st.subheader("Key geometry sliders")
    st.caption("Values shown are deltas (Δ) relative to the baseline geometry.")

    params = {}
    for col in slider_features:
        unit = PARAM_UNITS.get(col, "")
        base_val = float(baseline[col])
        left = float(smin[col] - base_val)
        right = float(smax[col] - base_val)

        key = f"off_{col}"
        if key not in st.session_state:
            st.session_state[key] = 0.0

        offset = st.slider(
            f"{pretty_param_name(col)} ({unit})" if unit else pretty_param_name(col),
            left,
            right,
            float(st.session_state[key]),
            0.001,
            key=key,
        )
        current_val = base_val + offset
        params[col] = current_val

        # No “range” text; just current setting
        st.caption(f"Current setting: **{fmt_with_unit(current_val, unit)}**")

    st.divider()
    st.caption("Built by ebprasad")

# Fill non-slider parameters with baseline
full_params = dict(baseline)
full_params.update(params)

# ----------------------------
# If not computed yet, show instruction and stop
# ----------------------------
if not compute:
    left, right = st.columns([1.25, 0.75])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Predicted coefficients")
        st.info("Adjust sliders in the sidebar, then click **Compute**.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick notes")
        st.markdown(
            """
- This is a fast screening tool for exploring trends.
- Predictions are usually most reliable near the baseline and within the dataset’s coverage.
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Uncertainty / reliability")
        st.info("Click **Compute** to evaluate uncertainty for the current configuration.")
        st.caption("As a rule of thumb, the ensemble tends to agree more near the baseline, and less as you push towards the edges of dataset coverage.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ----------------------------
# Predict (with spinner for polish)
# ----------------------------
with st.spinner("Running surrogate prediction…"):
    X = pd.DataFrame([full_params])[feature_cols]
    mean, std = ensemble_predict(models, X)

pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

delta_abs = {t: pred[t] - baseline_outputs[t] for t in targets}
delta_pct = {t: pct_change(pred[t], baseline_outputs[t]) for t in targets}

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1.25, 0.75])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Predicted coefficients")

    # 2x2 layout with % deltas
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    r1c1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta_pct['cd']:+.2f}%")
    r1c2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta_pct['cl']:+.2f}%")
    r2c1.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta_pct['clf']:+.2f}%")
    r2c2.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta_pct['clr']:+.2f}%")

    st.caption("Baseline values are the model prediction at the baseline geometry (dataset mean).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Baseline vs predicted (table)")

    tbl = pd.DataFrame(
        {
            "baseline": [baseline_outputs[t] for t in targets],
            "predicted": [pred[t] for t in targets],
            "delta (absolute)": [delta_abs[t] for t in targets],
        },
        index=targets,
    )
    st.dataframe(tbl.style.format({"baseline": "{:.6f}", "predicted": "{:.6f}", "delta (absolute)": "{:+.6f}"}))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What to try next (quick suggestions)")
    st.markdown(
        """
- Change one slider at a time and watch which coefficients respond most strongly.
- If an output becomes **low confidence**, nudge parameters back towards baseline and re-check.
- Use this as a fast *trend explorer*, then confirm any “interesting” cases with CFD in a real workflow.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Uncertainty / reliability")

    with st.expander("What do “std” and “p90” mean?", expanded=False):
        st.markdown(
            """
- **std (standard deviation):** Here it represents disagreement between models in the ensemble.  
  If they broadly agree, the std stays small, which is usually reassuring.

- **p90:** The 90th percentile of std measured on a held-out calibration split.  
  If std is above p90, the prediction falls into the most uncertain ~10% of cases, so it is flagged.

This is a practical reliability check, not a guaranteed probability.
"""
        )

    def is_low_conf(t):
        thr_val = float(thr.get(f"{t}_std_p90", np.nan))
        if np.isnan(thr_val):
            return False
        return unc[t] > thr_val

    def show_row(t):
        thr_val = float(thr.get(f"{t}_std_p90", np.nan))
        flagged = is_low_conf(t)
        icon = "⚠️" if flagged else "✅"
        status = "Low confidence" if flagged else "High confidence"

        if np.isnan(thr_val):
            st.write(f"**{t}**: std `{unc[t]:.2e}` → {icon} **{status}**")
        else:
            st.write(f"**{t}**: std `{unc[t]:.2e}` vs p90 `{thr_val:.2e}` → {icon} **{status}**")
        return flagged

    flags = [show_row(t) for t in ["cd", "cl", "clf", "clr"]]

    st.divider()
    if any(flags):
        st.warning("At least one output is in a higher-uncertainty region. In practice, you’d normally verify that case with CFD.")
    else:
        st.success("All outputs are below their p90 thresholds. This is a lower-uncertainty region.")

    st.caption("As a rule of thumb, the ensemble tends to agree more near the baseline, and less as you push towards the edges of dataset coverage.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Technical notes (humble but not hiding the work)
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Technical notes (for the curious)")

st.markdown(
    """
This demo is intended as a lightweight *surrogate* — useful for fast iteration and intuition, not a substitute for CFD sign-off.

**Training**
- Model: an **ensemble of Ridge regression pipelines** (linear model with L2 regularisation).
- Inputs: DrivAerML geometry parameters (a compact set of physically meaningful shape variables).
- Outputs: **cd, cl, clf, clr** from time-averaged CFD forces (constant reference values).

**Validation**
- Evaluation is designed to reflect **unseen runs**, rather than random shuffles, to avoid overly optimistic results.
- In the project repo, results are supported by predicted-vs-actual plots and error distributions.

**Reliability**
- The uncertainty shown here is **ensemble disagreement (std)** — a practical indicator of when the model is being pushed beyond familiar patterns.
- The **p90 threshold** flags the most uncertain ~10% of cases based on a held-out calibration split.

**Limitations**
- Like most surrogates, it is strongest at **interpolation** (within dataset coverage) and weaker for aggressive extrapolation.
- For engineering decisions, treat it as a fast screening tool and verify important cases with higher fidelity.
"""
)

# Optional small sensitivity hint (kept humble)
st.markdown("**Most-influential parameters shown in this demo**")
st.write(", ".join([pretty_param_name(x) for x in influential_hint]))

st.caption("Built by ebprasad · DrivAerML dataset by N. Ashton et al. (see dataset citation in the repo)")
st.markdown("</div>", unsafe_allow_html=True)
