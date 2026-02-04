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

st.set_page_config(page_title="DrivAerML aero surrogate", layout="wide")

# ----------------------------
# Modern styling (simple, clean, Streamlit-safe)
# ----------------------------
st.markdown(
    """
<style>
/* Page background + typography */
.stApp {
  background: radial-gradient(1200px 800px at 20% 10%, #162033 0%, #0b0f17 55%, #070a10 100%);
  color: #e8eefc;
}
h1, h2, h3 { letter-spacing: -0.02em; }
a { color: #9cc2ff; }

/* Make content a bit narrower */
.block-container { max-width: 1200px; padding-top: 2rem; }

/* Card-like containers */
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

/* Metric styling */
[data-testid="stMetricValue"] {
  font-size: 1.45rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Units and parameter display hints
# Based on DrivAerML Table: parameters are defined as changes relative to baseline.
# Most are in mm; vehicle pitch is in degrees. :contentReference[oaicite:1]{index=1}
# ----------------------------
PARAM_UNITS = {
    "Vehicle_Length": "mm",
    "Vehicle_Width": "mm",
    "Vehicle_Height": "mm",
    "Front_Overhang": "mm",
    "Front_Planview": "mm",
    "Hood_Angle": "mm",
    "Approach_Angle": "mm",
    "Windscreen_Angle": "mm",
    "Greenhouse_Tapering": "mm",
    "Backlight_Angle": "mm",
    "Decklid_Height": "mm",
    "Rearend_tapering": "mm",
    "Rear_Overhang": "mm",
    "Rear_Diffusor_Angle": "mm",   # note: spelling in your CSV is Diffusor; paper uses Diffuser :contentReference[oaicite:2]{index=2}
    "Vehicle_Ride_Height": "mm",
    "Vehicle_Pitch": "deg",
}

def pretty_param_name(s: str) -> str:
    return s.replace("_", " ")

def fmt_with_unit(value: float, unit: str) -> str:
    if unit == "deg":
        return f"{value:+.3f}°"
    return f"{value:+.1f} {unit}"

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
slider_features = cfg.get("slider_features", feature_cols)  # should be top 8
targets = cfg["targets"]  # expected: ["cd","cl","clf","clr"]

baseline = cfg["baseline"]          # baseline deltas (dataset mean); in this dataset these are *parameter values*, i.e. deltas vs baseline geometry
smin = cfg["slider_min"]
smax = cfg["slider_max"]

thr = cfg["uncertainty_thresholds"]
baseline_outputs = cfg["baseline_outputs"]

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("DrivAerML aero surrogate")
st.markdown(
    """
This page demonstrates a lightweight **surrogate model** trained on **DrivAerML** (500 CFD-tested DrivAer geometry variants).  
The model is an **ensemble of Ridge regression pipelines** used to predict **cd, cl, clf, clr** from 16 geometry parameters.

The geometry sliders below represent **changes relative to the baseline DrivAer shape** (units from the dataset paper). :contentReference[oaicite:3]{index=3}
""",
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Sidebar controls
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
    slider_rows = []

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

        # show a little “context line” under each slider
        st.caption(f"Current setting: **{fmt_with_unit(current_val, unit)}**  ·  Range: [{fmt_with_unit(base_val+left, unit)}, {fmt_with_unit(base_val+right, unit)}]")

        slider_rows.append([pretty_param_name(col), fmt_with_unit(current_val, unit)])

# Fill non-slider parameters with baseline so the model always gets full feature vector
full_params = dict(baseline)
full_params.update(params)

if not compute:
    st.info("Adjust sliders in the sidebar, then click **Compute**.")
    st.stop()

# ----------------------------
# Predict
# ----------------------------
X = pd.DataFrame([full_params])[feature_cols]
mean, std = ensemble_predict(models, X)

pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

delta = {t: pred[t] - baseline_outputs[t] for t in targets}

def is_low_conf(t):
    thr_val = float(thr.get(f"{t}_std_p90", np.nan))
    if np.isnan(thr_val):
        return False
    return unc[t] > thr_val

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1.25, 0.75])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Predicted coefficients")

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)

    m1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta['cd']:+.6f}")
    m2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta['cl']:+.6f}")
    m3.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta['clf']:+.6f}")
    m4.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta['clr']:+.6f}")

    st.markdown("Baseline values are the model prediction at the baseline geometry (dataset mean).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Baseline vs predicted (table)")

    tbl = pd.DataFrame(
        {
            "baseline": [baseline_outputs[t] for t in targets],
            "predicted": [pred[t] for t in targets],
            "delta": [delta[t] for t in targets],
        },
        index=targets,
    )
    st.dataframe(
        tbl.style.format(
            {
                "baseline": "{:.6f}",
                "predicted": "{:.6f}",
                "delta": "{:+.6f}",
            }
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Parameter settings (shown sliders)")
    st.caption("These are the 8 displayed geometry deltas used for this prediction.")
    param_tbl = pd.DataFrame(slider_rows, columns=["Parameter", "Current Δ setting"])
    st.dataframe(param_tbl, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Uncertainty / reliability")

    with st.expander("What do “std” and “p90” mean?", expanded=False):
        st.markdown(
            """
- **std (standard deviation):** here it’s the *disagreement between the models* in the ensemble.  
  If they broadly agree, std is small, which is usually reassuring.

- **p90:** the 90th percentile of std measured on a held-out calibration split.  
  If a prediction’s std is above p90, it’s in the most uncertain ~10% of cases, so it gets flagged.

This is a practical reliability check, not a guaranteed probability.
"""
        )

    def show_row(t):
        thr_val = float(thr.get(f"{t}_std_p90", np.nan))
        flagged = is_low_conf(t)
        status = "Low confidence" if flagged else "High confidence"

        if np.isnan(thr_val):
            st.write(f"**{t}**: std `{unc[t]:.2e}` → **{status}**")
        else:
            st.write(f"**{t}**: std `{unc[t]:.2e}` vs p90 `{thr_val:.2e}` → **{status}**")

        return flagged

    flags = [show_row(t) for t in ["cd", "cl", "clf", "clr"]]

    st.divider()
    if any(flags):
        st.warning("At least one output is in a higher-uncertainty region. In practice, you’d normally verify that case with CFD.")
    else:
        st.success("All outputs are below their p90 thresholds. This is a lower-uncertainty region.")

    st.markdown("</div>", unsafe_allow_html=True)
