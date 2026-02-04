import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from streamlit.components.v1 import html as st_html

# ----------------------------
# Settings
# ----------------------------
MODELS_DIR = Path("models")
N_MODELS = 30

st.set_page_config(page_title="DrivAerML aero surrogate — demo", layout="wide")

# ----------------------------
# Styling (modern dark, keep sidebar toggle visible, red primary button)
# ----------------------------
st.markdown(
    """
<style>
/* Hide Streamlit chrome (but keep sidebar collapsed control visible) */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
/* IMPORTANT: do NOT hide header entirely or the sidebar toggle can disappear on some builds */
/* [data-testid="stHeader"] {display: none;} */ 

/* Keep the “show sidebar” control visible when sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"] {
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
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

/* Buttons default */
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

/* Filled red primary button (Compute) */
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

    # Angles
    "Hood_Angle": "deg",
    "Approach_Angle": "deg",
    "Windscreen_Angle": "deg",
    "Backlight_Angle": "deg",
    "Rear_Diffusor_Angle": "deg",
    "Vehicle_Pitch": "deg",

    # Other geometry knobs (often treated like geometric deltas)
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

def norm01(x, lo, hi):
    if hi == lo:
        return 0.5
    return float((x - lo) / (hi - lo))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ----------------------------
# Simple 2D schematic (SVG)
# Shows baseline outline + current outline (top view + side view)
# This is a schematic, not exact geometry. It’s meant to communicate “direction” visually.
# ----------------------------
def car_schematic_svg(params, baseline, smin, smax):
    # Pick a few key knobs; if missing, treat as baseline
    def val(name):
        return float(params.get(name, baseline.get(name, 0.0)))

    # Normalise to 0..1 using slider min/max (keeps behaviour stable)
    L_n = norm01(val("Vehicle_Length"), float(smin.get("Vehicle_Length", -100)), float(smax.get("Vehicle_Length", 100)))
    W_n = norm01(val("Vehicle_Width"), float(smin.get("Vehicle_Width", -100)), float(smax.get("Vehicle_Width", 100)))
    H_n = norm01(val("Vehicle_Height"), float(smin.get("Vehicle_Height", -100)), float(smax.get("Vehicle_Height", 100)))
    FO_n = norm01(val("Front_Overhang"), float(smin.get("Front_Overhang", -100)), float(smax.get("Front_Overhang", 100)))
    RO_n = norm01(val("Rear_Overhang"), float(smin.get("Rear_Overhang", -100)), float(smax.get("Rear_Overhang", 100)))
    RH_n = norm01(val("Vehicle_Ride_Height"), float(smin.get("Vehicle_Ride_Height", -50)), float(smax.get("Vehicle_Ride_Height", 50)))
    P_n  = norm01(val("Vehicle_Pitch"), float(smin.get("Vehicle_Pitch", -3)), float(smax.get("Vehicle_Pitch", 3)))

    # Map normalised values to schematic dimensions
    # Baseline "car" box sizes
    base_len = 260
    base_wid = 110
    base_hgt = 65

    # Variation ranges (schematic, not physical)
    len_scale = 0.85 + 0.30 * clamp(L_n, 0, 1)
    wid_scale = 0.85 + 0.30 * clamp(W_n, 0, 1)
    hgt_scale = 0.85 + 0.30 * clamp(H_n, 0, 1)

    # Overhangs affect length distribution
    front_ext = (FO_n - 0.5) * 30
    rear_ext  = (RO_n - 0.5) * 30

    # Ride height shifts body up in side view
    ride_up = (RH_n - 0.5) * 18

    # Pitch rotates slightly
    pitch_deg = (P_n - 0.5) * 6.0  # schematic max ±3°
    pitch = pitch_deg

    # Top view dimensions
    cur_len = base_len * len_scale
    cur_wid = base_wid * wid_scale

    # Side view dimensions
    cur_hgt = base_hgt * hgt_scale

    # Canvas
    W = 520
    H = 240

    # Top view placement
    top_x = 30
    top_y = 40

    # Side view placement
    side_x = 300
    side_y = 160

    # Baseline box
    base_top_len = base_len
    base_top_wid = base_wid
    base_top_x = top_x + (base_len - base_top_len) / 2
    base_top_y = top_y + (base_wid - base_top_wid) / 2

    # Current box (apply overhang shifts)
    cur_top_x = top_x + (base_len - cur_len) / 2 - front_ext
    cur_top_y = top_y + (base_wid - cur_wid) / 2
    cur_top_len = cur_len + front_ext + rear_ext
    cur_top_wid = cur_wid

    # Side baseline box
    base_side_w = 170
    base_side_h = base_hgt
    base_side_x = side_x
    base_side_y = side_y - base_side_h

    # Side current box (height + ride)
    cur_side_w = 170 * len_scale
    cur_side_h = cur_hgt
    cur_side_x = side_x + (170 - cur_side_w) / 2
    cur_side_y = (side_y - cur_side_h) - ride_up

    # SVG
    svg = f"""
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g1" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="rgba(156,194,255,0.35)"/>
      <stop offset="1" stop-color="rgba(156,194,255,0.10)"/>
    </linearGradient>
  </defs>

  <!-- Labels -->
  <text x="{top_x}" y="22" fill="rgba(232,238,252,0.9)" font-size="14" font-family="system-ui, -apple-system, Segoe UI, Roboto">Geometry schematic (2D)</text>
  <text x="{top_x}" y="34" fill="rgba(232,238,252,0.65)" font-size="11" font-family="system-ui, -apple-system, Segoe UI, Roboto">Baseline outline vs current outline (schematic)</text>

  <!-- TOP VIEW -->
  <text x="{top_x}" y="{top_y-10}" fill="rgba(232,238,252,0.75)" font-size="11" font-family="system-ui, -apple-system, Segoe UI, Roboto">Top view</text>

  <!-- baseline top outline -->
  <rect x="{base_top_x}" y="{base_top_y}" width="{base_top_len}" height="{base_top_wid}"
        fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="2" rx="18"/>

  <!-- current top outline -->
  <rect x="{cur_top_x}" y="{cur_top_y}" width="{cur_top_len}" height="{cur_top_wid}"
        fill="url(#g1)" stroke="rgba(156,194,255,0.75)" stroke-width="2.5" rx="18"/>

  <!-- direction arrow -->
  <line x1="{cur_top_x+8}" y1="{cur_top_y+cur_top_wid/2}" x2="{cur_top_x+cur_top_len-8}" y2="{cur_top_y+cur_top_wid/2}"
        stroke="rgba(156,194,255,0.35)" stroke-width="2"/>
  <text x="{cur_top_x+cur_top_len/2-12}" y="{cur_top_y+cur_top_wid/2-8}"
        fill="rgba(156,194,255,0.65)" font-size="10" font-family="system-ui, -apple-system, Segoe UI, Roboto">Length</text>

  <!-- SIDE VIEW -->
  <text x="{side_x}" y="{side_y-90}" fill="rgba(232,238,252,0.75)" font-size="11" font-family="system-ui, -apple-system, Segoe UI, Roboto">Side view</text>

  <!-- baseline ground line -->
  <line x1="{side_x-10}" y1="{side_y}" x2="{side_x+190}" y2="{side_y}" stroke="rgba(255,255,255,0.15)" stroke-width="2"/>
  <text x="{side_x+140}" y="{side_y+16}" fill="rgba(255,255,255,0.35)" font-size="10" font-family="system-ui, -apple-system, Segoe UI, Roboto">Ground</text>

  <!-- baseline side outline -->
  <rect x="{base_side_x}" y="{base_side_y}" width="{base_side_w}" height="{base_side_h}"
        fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="2" rx="10"/>

  <!-- current side outline with pitch -->
  <g transform="rotate({pitch:.2f} {cur_side_x+cur_side_w/2:.2f} {cur_side_y+cur_side_h/2:.2f})">
    <rect x="{cur_side_x}" y="{cur_side_y}" width="{cur_side_w}" height="{cur_side_h}"
          fill="url(#g1)" stroke="rgba(156,194,255,0.75)" stroke-width="2.5" rx="10"/>
  </g>

  <!-- captions -->
  <text x="{side_x}" y="{side_y+38}" fill="rgba(232,238,252,0.55)" font-size="10" font-family="system-ui, -apple-system, Segoe UI, Roboto">
    Pitch: {pitch_deg:+.2f}° · Ride height shift: {ride_up:+.1f}px (schematic)
  </text>
</svg>
"""
    return svg

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

# ----------------------------
# Header (clean, structured, includes “demo”)
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("DrivAerML aero surrogate — demo")
st.markdown(
    """
**What this demo does**
- Predicts **cd, cl, clf, clr** from a small set of geometry parameters.

**How it works**
- Uses an **ensemble of Ridge regression pipelines** trained on **DrivAerML** (500 CFD-tested DrivAer variants).

**How to read the sliders**
- Slider values are **geometry deltas (Δ)** relative to the baseline DrivAer shape, with units shown per parameter.
"""
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

        # Only show current setting (no range line)
        st.caption(f"Current setting: **{fmt_with_unit(current_val, unit)}**")

    st.divider()
    st.caption("Built by ebprasad")

# Fill non-slider parameters with baseline so model sees full feature vector
full_params = dict(baseline)
full_params.update(params)

# Show schematic even before compute (so UI feels “live”)
schematic = car_schematic_svg(full_params, baseline, smin, smax)

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1.25, 0.75])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Predicted coefficients")

    if not compute:
        st.info("Adjust sliders in the sidebar, then click **Compute**.")
        st.caption("The geometry schematic below updates live as you move the sliders.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Predict
        X = pd.DataFrame([full_params])[feature_cols]
        mean, std = ensemble_predict(models, X)

        pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
        unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

        delta_abs = {t: pred[t] - baseline_outputs[t] for t in targets}
        delta_pct = {t: pct_change(pred[t], baseline_outputs[t]) for t in targets}

        # Keep square layout (2x2) and show % deltas in the chips
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        r1c1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta_pct['cd']:+.2f}%")
        r1c2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta_pct['cl']:+.2f}%")
        r2c1.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta_pct['clf']:+.2f}%")
        r2c2.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta_pct['clr']:+.2f}%")

        st.caption("Baseline values are the model prediction at the baseline geometry (dataset mean).")
        st.markdown("</div>", unsafe_allow_html=True)

        # Table (absolute deltas) stays, because you asked to keep absolute there
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
        st.dataframe(
            tbl.style.format(
                {
                    "baseline": "{:.6f}",
                    "predicted": "{:.6f}",
                    "delta (absolute)": "{:+.6f}",
                }
            )
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Geometry schematic card (always visible)
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Geometry schematic (2D)")
    st.caption("This is a simple schematic to visualise direction of change, not an exact CAD morph.")
    st_html(schematic, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Uncertainty / reliability")

    with st.expander("What do “std” and “p90” mean?", expanded=False):
        st.markdown(
            """
- **std (standard deviation):** Here it represents the disagreement between models in the ensemble.  
  If they broadly agree, the std stays small, which is usually reassuring.

- **p90:** The 90th percentile of std measured on a held-out calibration split.  
  If std is above p90, the prediction falls into the most uncertain ~10% of cases, so it is flagged.

This is a practical reliability check, not a guaranteed probability.
"""
        )

    if not compute:
        st.info("Click **Compute** to evaluate uncertainty for the current configuration.")
        st.caption("As a rule of thumb, the ensemble tends to agree more near the baseline, and less as you push towards the edges of dataset coverage.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        def is_low_conf(t):
            thr_val = float(thr.get(f"{t}_std_p90", np.nan))
            if np.isnan(thr_val):
                return False
            return unc[t] > thr_val

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

        st.caption("As a rule of thumb, the ensemble tends to agree more near the baseline, and less as you push towards the edges of dataset coverage.")
        st.markdown("</div>", unsafe_allow_html=True)
