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
# Styling (keep it clean; DO NOT hide header/sidebar controls now)
# ----------------------------
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Page background */
.stApp {
  background: radial-gradient(1200px 800px at 20% 10%, #162033 0%, #0b0f17 55%, #070a10 100%);
  color: #e8eefc;
}

/* Container */
.block-container { max-width: 1200px; padding-top: 1.4rem; }

/* Card */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.25);
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

/* Red primary button */
button[kind="primary"] {
  background: #d32f2f !important;
  border: 1px solid #d32f2f !important;
  color: white !important;
}
button[kind="primary"]:hover {
  background: #b71c1c !important;
  border: 1px solid #b71c1c !important;
}

/* Metric size */
[data-testid="stMetricValue"] { font-size: 1.45rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Units
# ----------------------------
PARAM_UNITS = {
    "Vehicle_Length": "mm",
    "Vehicle_Width": "mm",
    "Vehicle_Height": "mm",
    "Front_Overhang": "mm",
    "Rear_Overhang": "mm",
    "Vehicle_Ride_Height": "mm",
    "Vehicle_Pitch": "deg",
    "Approach_Angle": "deg",
    "Windscreen_Angle": "deg",
    "Backlight_Angle": "deg",
    "Rear_Diffusor_Angle": "deg",
    "Hood_Angle": "deg",
    "Front_Planview": "mm",
    "Greenhouse_Tapering": "mm",
    "Decklid_Height": "mm",
    "Rearend_tapering": "mm",
}

# Parameters we want for the schematic (if present in dataset)
SCHEM_KEYS = [
    "Vehicle_Length",
    "Vehicle_Width",
    "Vehicle_Height",
    "Front_Overhang",
    "Rear_Overhang",
    "Vehicle_Ride_Height",
    "Vehicle_Pitch",
]

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
# 2D schematic (SVG)
# ----------------------------
def car_schematic_svg(full_params, baseline, smin, smax):
    def get(name):
        return float(full_params.get(name, baseline.get(name, 0.0)))

    # If key missing from smin/smax, give safe defaults
    def lo(name, default=-1.0):
        return float(smin.get(name, default))

    def hi(name, default=1.0):
        return float(smax.get(name, default))

    L_n = norm01(get("Vehicle_Length"), lo("Vehicle_Length", -150), hi("Vehicle_Length", 150))
    W_n = norm01(get("Vehicle_Width"),  lo("Vehicle_Width", -100),  hi("Vehicle_Width", 100))
    H_n = norm01(get("Vehicle_Height"), lo("Vehicle_Height", -80),   hi("Vehicle_Height", 80))
    FO_n = norm01(get("Front_Overhang"), lo("Front_Overhang", -80),  hi("Front_Overhang", 80))
    RO_n = norm01(get("Rear_Overhang"),  lo("Rear_Overhang", -80),   hi("Rear_Overhang", 80))
    RH_n = norm01(get("Vehicle_Ride_Height"), lo("Vehicle_Ride_Height", -40), hi("Vehicle_Ride_Height", 40))
    P_n  = norm01(get("Vehicle_Pitch"), lo("Vehicle_Pitch", -3), hi("Vehicle_Pitch", 3))

    base_len, base_wid, base_hgt = 260, 110, 65

    len_scale = 0.85 + 0.30 * clamp(L_n, 0, 1)
    wid_scale = 0.85 + 0.30 * clamp(W_n, 0, 1)
    hgt_scale = 0.85 + 0.30 * clamp(H_n, 0, 1)

    front_ext = (FO_n - 0.5) * 30
    rear_ext  = (RO_n - 0.5) * 30

    ride_up = (RH_n - 0.5) * 18
    pitch_deg = (P_n - 0.5) * 6.0

    cur_len = base_len * len_scale
    cur_wid = base_wid * wid_scale
    cur_hgt = base_hgt * hgt_scale

    W, H = 520, 240
    top_x, top_y = 30, 55
    side_x, side_y = 300, 170

    base_top_x = top_x
    base_top_y = top_y
    base_top_len = base_len
    base_top_wid = base_wid

    cur_top_x = top_x + (base_len - cur_len) / 2 - front_ext
    cur_top_y = top_y + (base_wid - cur_wid) / 2
    cur_top_len = cur_len + front_ext + rear_ext
    cur_top_wid = cur_wid

    base_side_w = 170
    base_side_h = base_hgt
    base_side_x = side_x
    base_side_y = side_y - base_side_h

    cur_side_w = 170 * len_scale
    cur_side_h = cur_hgt
    cur_side_x = side_x + (170 - cur_side_w) / 2
    cur_side_y = (side_y - cur_side_h) - ride_up

    svg = f"""
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g1" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="rgba(156,194,255,0.35)"/>
      <stop offset="1" stop-color="rgba(156,194,255,0.10)"/>
    </linearGradient>
  </defs>

  <text x="{top_x}" y="22" fill="rgba(232,238,252,0.9)" font-size="14" font-family="system-ui, -apple-system, Segoe UI, Roboto">
    Geometry schematic (2D)
  </text>

  <text x="{top_x}" y="40" fill="rgba(232,238,252,0.75)" font-size="11" font-family="system-ui, -apple-system, Segoe UI, Roboto">
    Top view
  </text>

  <!-- Baseline top -->
  <rect x="{base_top_x}" y="{base_top_y}" width="{base_top_len}" height="{base_top_wid}"
        fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="2" rx="18"/>

  <!-- Current top -->
  <rect x="{cur_top_x}" y="{cur_top_y}" width="{cur_top_len}" height="{cur_top_wid}"
        fill="url(#g1)" stroke="rgba(156,194,255,0.75)" stroke-width="2.5" rx="18"/>

  <text x="{side_x}" y="{side_y-90}" fill="rgba(232,238,252,0.75)" font-size="11" font-family="system-ui, -apple-system, Segoe UI, Roboto">
    Side view
  </text>

  <!-- Ground -->
  <line x1="{side_x-10}" y1="{side_y}" x2="{side_x+190}" y2="{side_y}" stroke="rgba(255,255,255,0.15)" stroke-width="2"/>
  <text x="{side_x+140}" y="{side_y+16}" fill="rgba(255,255,255,0.35)" font-size="10" font-family="system-ui, -apple-system, Segoe UI, Roboto">
    Ground
  </text>

  <!-- Baseline side -->
  <rect x="{base_side_x}" y="{base_side_y}" width="{base_side_w}" height="{base_side_h}"
        fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="2" rx="10"/>

  <!-- Current side (rotated for pitch) -->
  <g transform="rotate({pitch_deg:.2f} {cur_side_x+cur_side_w/2:.2f} {cur_side_y+cur_side_h/2:.2f})">
    <rect x="{cur_side_x}" y="{cur_side_y}" width="{cur_side_w}" height="{cur_side_h}"
          fill="url(#g1)" stroke="rgba(156,194,255,0.75)" stroke-width="2.5" rx="10"/>
  </g>

  <text x="{side_x}" y="{side_y+38}" fill="rgba(232,238,252,0.55)" font-size="10" font-family="system-ui, -apple-system, Segoe UI, Roboto">
    Pitch: {pitch_deg:+.2f}° · Ride-height effect (schematic): {ride_up:+.1f}px
  </text>
</svg>
"""
    return svg

# ----------------------------
# Load config/models
# ----------------------------
cfg, models = load_models_and_config()
feature_cols = cfg["feature_cols"]
top8_features = cfg.get("slider_features", feature_cols)  # your chosen top 8
targets = cfg["targets"]

baseline = cfg["baseline"]
smin = cfg["slider_min"]
smax = cfg["slider_max"]
thr = cfg["uncertainty_thresholds"]
baseline_outputs = cfg["baseline_outputs"]

# ----------------------------
# Header
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
# In-page controls panel (so you never lose controls even if sidebar disappears)
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Controls")

cA, cB, cC = st.columns([1, 1, 1])
with cA:
    if st.button("Reset sliders"):
        for col in list(set(top8_features) | set(SCHEM_KEYS)):
            st.session_state[f"off_{col}"] = 0.0
        st.rerun()
with cB:
    compute = st.button("Compute", type="primary")
with cC:
    st.caption("Built by ebprasad")

st.caption("Top 8 influential parameters are shown below. The schematic is driven by width/length/height/overhang/ride height/pitch where available.")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Slider panel (always available)
# We show:
# - the top 8 features (for the “ML demo” story),
# - plus any missing schematic drivers (so the picture actually changes).
# ----------------------------
slider_set = []
for f in top8_features:
    if f in feature_cols and f not in slider_set:
        slider_set.append(f)
for f in SCHEM_KEYS:
    if f in feature_cols and f not in slider_set:
        slider_set.append(f)

# Build params from sliders as offsets from baseline
params = {}
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Geometry sliders")

g1, g2 = st.columns(2)
col_iter = 0
for col in slider_set:
    unit = PARAM_UNITS.get(col, "")
    base_val = float(baseline[col])
    left = float(smin[col] - base_val)
    right = float(smax[col] - base_val)

    key = f"off_{col}"
    if key not in st.session_state:
        st.session_state[key] = 0.0

    target_col = g1 if (col_iter % 2 == 0) else g2
    with target_col:
        offset = st.slider(
            f"{pretty_param_name(col)} ({unit})" if unit else pretty_param_name(col),
            left, right, float(st.session_state[key]), 0.001, key=key
        )
        current_val = base_val + offset
        params[col] = current_val
        st.caption(f"Current setting: **{fmt_with_unit(current_val, unit)}**")
    col_iter += 1

st.markdown("</div>", unsafe_allow_html=True)

# Fill non-slider params with baseline
full_params = dict(baseline)
full_params.update(params)

# ----------------------------
# Main layout (results + uncertainty + schematic)
# ----------------------------
left, right = st.columns([1.25, 0.75])

# Schematic always reflects current slider values
schematic = car_schematic_svg(full_params, baseline, smin, smax)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Predicted coefficients")

    if not compute:
        st.info("Adjust sliders, then click **Compute**.")
        st.caption("The geometry schematic below updates as you move the sliders.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        X = pd.DataFrame([full_params])[feature_cols]
        mean, std = ensemble_predict(models, X)

        pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
        unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

        delta_abs = {t: pred[t] - baseline_outputs[t] for t in targets}
        delta_pct = {t: pct_change(pred[t], baseline_outputs[t]) for t in targets}

        # 2x2 layout + % chips
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        r1c1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta_pct['cd']:+.2f}%")
        r1c2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta_pct['cl']:+.2f}%")
        r2c1.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta_pct['clf']:+.2f}%")
        r2c2.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta_pct['clr']:+.2f}%")

        st.caption("Baseline values are the model prediction at the baseline geometry (dataset mean).")
        st.markdown("</div>", unsafe_allow_html=True)

        # Table with absolute deltas
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

# Schematic card
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
- **std (standard deviation):** Here it represents disagreement between models in the ensemble.  
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
