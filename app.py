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

def try_load_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None

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

# Optional metadata (if you’ve added it to ui_config_4targets.json)
split_note = cfg.get("split_note", "Group split by run (unseen geometries held out).")
n_total = cfg.get("n_total", None)
n_used = cfg.get("n_used", None)
n_train = cfg.get("n_train", None)
n_test = cfg.get("n_test", None)

# Metrics files (recommended to commit under models/)
df_final = try_load_csv(MODELS_DIR / "metrics_final_test_ridge.csv")
df_baselines = try_load_csv(MODELS_DIR / "metrics_baselines.csv")
df_thr = try_load_csv(MODELS_DIR / "uncertainty_thresholds_p90.csv")  # optional

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
# Technical evaluation (numbers + things F1 reviewers look for)
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Technical evaluation (model card)")

# Key facts that reviewers care about
st.markdown("**Model**")
st.markdown(
    f"""
- Ensemble size: **{N_MODELS}** (Ridge regression pipelines)
- Features: **{len(feature_cols)}** geometry parameters
- Targets: **{', '.join(targets)}**
- Split / leakage control: **{split_note}**
"""
)

# Dataset / usage counts (if you populated them in ui_config; otherwise we compute what we can)
meta_bits = []
if n_total is not None:
    meta_bits.append(f"Total geometries (nominal): **{n_total}**")
if n_used is not None:
    meta_bits.append(f"Used for modelling: **{n_used}**")
if n_train is not None:
    meta_bits.append(f"Train: **{n_train}**")
if n_test is not None:
    meta_bits.append(f"Test: **{n_test}**")

if meta_bits:
    st.markdown("**Data**")
    st.markdown("- " + "\n- ".join(meta_bits))

# Show your real final test metrics (R2/RMSE etc)
st.markdown("**Final test performance**")

if df_final is None:
    st.warning(
        "No `models/metrics_final_test_ridge.csv` found in the deployed repo. "
        "Commit that file to `models/` to display R² / RMSE here."
    )
else:
    # Expecting columns like: target, split, model, r2, rmse, mae ... (we’ll display what exists)
    # Make it robust by selecting common columns if present
    cols_prefer = [c for c in ["target", "r2", "rmse", "mae", "mape", "n"] if c in df_final.columns]
    if not cols_prefer:
        st.write(df_final)
    else:
        view = df_final[cols_prefer].copy()

        # Nice formatting
        fmt = {}
        if "r2" in view.columns:
            fmt["r2"] = "{:.4f}"
        if "rmse" in view.columns:
            fmt["rmse"] = "{:.6f}"
        if "mae" in view.columns:
            fmt["mae"] = "{:.6f}"
        if "mape" in view.columns:
            fmt["mape"] = "{:.2f}%"

        st.dataframe(view.style.format(fmt))

        # Quick textual summary (technical, not fluffy)
        if "r2" in view.columns and "rmse" in view.columns and "target" in view.columns:
            best = view.sort_values("r2", ascending=False).iloc[0]
            worst = view.sort_values("r2", ascending=True).iloc[0]
            st.caption(
                f"Best R²: {best['target']} = {best['r2']:.4f} (RMSE {best['rmse']:.6f}) · "
                f"Weakest R²: {worst['target']} = {worst['r2']:.4f} (RMSE {worst['rmse']:.6f})."
            )

# Optional: show baseline comparison table if present
with st.expander("Baselines and comparisons (optional)", expanded=False):
    if df_baselines is None:
        st.info("Commit `models/metrics_baselines.csv` to show baseline model comparisons here.")
    else:
        st.dataframe(df_baselines)

    if df_thr is None:
        st.info("Commit `models/uncertainty_thresholds_p90.csv` if you’d like the calibrated p90 thresholds shown as a table.")
    else:
        st.dataframe(df_thr)

st.markdown("**Engineering interpretation / limitations**")
st.markdown(
    """
- This surrogate is intended for **rapid screening** and **trend exploration**. Any design decision should be verified with higher fidelity when it matters.
- Expected failure mode: **extrapolation** (inputs near/outside training coverage) — which is why the app surfaces an ensemble-disagreement flag.
- The uncertainty here is **model disagreement**, not epistemic certainty in a strict probabilistic sense; it is used as a pragmatic “trust but verify” indicator.
"""
)

st.caption("Built by ebprasad · DrivAerML dataset by N. Ashton et al. (citation in the repo)")
st.markdown("</div>", unsafe_allow_html=True)
