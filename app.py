import json
import random
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
# Text variation (native UK tone, lightly “analyst-like”)
# ----------------------------
OPENERS = [
    "From what I can see,",
    "Based on the current inputs,",
    "Looking at the outputs,",
    "The results suggest",
    "It looks like",
    "On this run,",
    "This setup indicates",
    "From the model’s prediction,",
]

TRUST_HIGH = [
    "Uncertainty is below the p90 threshold, so I’m reasonably happy with this one.",
    "The ensemble agrees well here, which is a good sign.",
    "This sits in a lower-uncertainty region, so I’d treat it as fairly reliable.",
    "Model disagreement is low, so this looks like a confident estimate.",
]

TRUST_LOW = [
    "Uncertainty is above the p90 threshold, so I’d treat this as indicative and verify with CFD if it matters.",
    "The ensemble disagrees more than usual here — worth a cross-check before you act on it.",
    "This is in a higher-uncertainty region, so I wouldn’t rely on it without verification.",
    "It’s nudging the edge of what the model has seen — I’d take it as a direction of travel rather than a guarantee.",
]

TEMPLATES_GENERIC = [
    "{opener} the vehicle’s {metric} {direction} {pct} compared with baseline. {trust}",
    "{opener} {metric} {direction} {pct} versus baseline (Δ={abs}). {trust}",
    "{opener} I’m seeing a {direction_word} in {metric}: {pct}. {trust}",
    "{opener} relative to baseline, {metric} {direction} {pct}. {trust}",
    "{opener} {metric} {direction} {pct}; from the uncertainty check: {trust}",
    "{opener} the predicted {metric} {direction} {pct} (Δ={abs}). {trust}",
]

def make_statement(metric_name: str, delta_abs: float, delta_pct: float, is_low_conf: bool, plain: str | None = None):
    opener = random.choice(OPENERS)
    trust = random.choice(TRUST_LOW if is_low_conf else TRUST_HIGH)

    direction = "has decreased by" if delta_pct < 0 else "has increased by"
    direction_word = "reduction" if delta_pct < 0 else "increase"

    pct_str = f"{abs(delta_pct):.2f}%"
    abs_str = f"{delta_abs:+.6f}"

    template = random.choice(TEMPLATES_GENERIC)

    # Occasionally add the extra context line (keeps it fresh without being noisy)
    if plain and random.random() < 0.35:
        template = template + f" ({plain})"

    return template.format(
        opener=opener,
        metric=metric_name,
        direction=direction,
        direction_word=direction_word,
        pct=pct_str,
        abs=abs_str,
        trust=trust,
    )

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

thr = cfg["uncertainty_thresholds"]  # expects cd/cl/clf/clr std p90 entries
baseline_outputs = cfg["baseline_outputs"]

# ----------------------------
# Header
# ----------------------------
st.title("DrivAerML Aero Surrogate — Interactive Coefficient Predictor")

st.markdown(
    """
**What this is:** a fast *surrogate model* trained on **DrivAerML** (500 parametrically-morphed DrivAer geometries with CFD force coefficients).

**Model:** an **ensemble of Ridge Regression models** (linear, L2-regularised), predicting **cd, cl, clf, clr** from geometry parameters.

**How to use it:** the sliders tweak a handful of influential geometry parameters around a reference (“baseline”) geometry (here: the **dataset mean**).  
Click **Compute** to get predictions and a comparison vs baseline.
"""
)

with st.expander("Baseline and slider limits (quick note)", expanded=False):
    st.markdown(
        """
- **Baseline geometry:** mean of the dataset parameters (a neutral reference).
- **Slider limits:** 5th–95th percentile from the dataset (helps avoid extreme out-of-distribution inputs).
- **Interpretation:** it’s most trustworthy when interpolating within the dataset coverage.
"""
    )

# ----------------------------
# Sidebar controls (top K sliders only)
# ----------------------------
with st.sidebar:
    st.header("Controls")
    st.caption("To keep the demo tidy, you’re only seeing the most influential parameters (top 6–8).")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset sliders"):
            reset_sliders(slider_features)
            st.rerun()
    with c2:
        compute = st.button("Compute", type="primary")

    st.divider()
    st.subheader("Key geometry sliders")
    st.caption("Sliders are offsets from baseline. Parameters not shown stay at baseline.")

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

# Fill non-slider parameters with baseline (full feature vector always)
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

# flags using p90 thresholds
def is_low_conf(t):
    thr_val = float(thr.get(f"{t}_std_p90", np.nan))
    if np.isnan(thr_val):
        return False
    return unc[t] > thr_val

cd_low = is_low_conf("cd")
cl_low = is_low_conf("cl")
clf_low = is_low_conf("clf")
clr_low = is_low_conf("clr")

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

    st.markdown("**Drag (cd)**")
    st.write(make_statement(
        "drag coefficient (cd)",
        delta["cd"],
        delta_pct["cd"],
        cd_low,
        plain="Lower drag typically helps top speed and efficiency."
    ))

    st.markdown("**Lift (cl)**")
    st.write(make_statement(
        "lift coefficient (cl)",
        delta["cl"],
        delta_pct["cl"],
        cl_low,
        plain="Interpretation depends on sign convention (lift vs downforce)."
    ))

    st.markdown("**Front lift (clf)**")
    st.write(make_statement(
        "front lift coefficient (clf)",
        delta["clf"],
        delta_pct["clf"],
        clf_low,
        plain="Affects front-end aero balance and turn-in feel."
    ))

    st.markdown("**Rear lift (clr)**")
    st.write(make_statement(
        "rear lift coefficient (clr)",
        delta["clr"],
        delta_pct["clr"],
        clr_low,
        plain="Affects rear stability, especially in high-speed corners."
    ))

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
- **std (standard deviation):** in this app it’s the *disagreement* between models in the ensemble.  
  If they all broadly agree, std stays small — usually a sign the prediction is more trustworthy in that region.

- **p90:** the 90th percentile of uncertainty on a held-out calibration split.  
  If std is above p90, the prediction lands in the most uncertain ~10% of cases, so it’s flagged as **low confidence**.

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
            "One or more outputs sit in a higher-uncertainty region. "
            "In a real workflow, you’d usually verify those cases with CFD / higher fidelity."
        )
    else:
        st.success("All outputs sit in a lower-uncertainty region (below the p90 thresholds).")

    st.caption("Tip: uncertainty is often lowest near baseline, and rises as you push towards the edge of dataset coverage.")
