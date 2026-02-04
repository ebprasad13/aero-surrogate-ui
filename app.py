import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# Config
# ============================================================
MODELS_DIR = Path("models")
N_MODELS = 30  # ridge4_ensemble_00..29

st.set_page_config(page_title="DrivAerML aero surrogate — demo", layout="wide")

# ============================================================
# Styling (keep Streamlit top bar; blend app background to match)
# ============================================================
st.markdown(
    """
<style>
/* Keep the Streamlit top bar (do NOT hide stHeader). Hide only menu/footer. */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Typography: Apple-ish system stack */
html, body, [class*="css"]  {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
               "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  letter-spacing: 0.12px;
}

/* Match Streamlit Cloud dark chrome so the top area blends */
.stApp {
  background: #0f1115;   /* close to Streamlit top bar shade */
  color: #eef2ff;
}

/* Layout */
.block-container {
  max-width: 1180px;
  padding-top: 1.2rem;
  padding-bottom: 2.2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.04);
  border-right: 1px solid rgba(255,255,255,0.10);
}

/* Cards */
.card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: 0 14px 38px rgba(0,0,0,0.28);
  backdrop-filter: blur(10px);
}

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: 0.56rem 0.95rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}
.stButton>button:hover {
  border: 1px solid rgba(166,200,255,0.75);
  transform: translateY(-1px);
  transition: 140ms ease;
}

/* Primary (Compute) */
button[kind="primary"] {
  background: linear-gradient(90deg, #ff3b30 0%, #ff2d55 100%) !important;
  border: none !important;
  color: white !important;
  box-shadow: 0 10px 22px rgba(255,45,85,0.22);
}
button[kind="primary"]:hover { filter: brightness(0.97); }

/* Metric typography */
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stMetricDelta"] { font-size: 1.05rem; }

/* Highlight box for key parameters */
.highlight-box {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(170,200,255,0.34);
  border-radius: 14px;
  padding: 12px 12px 6px 12px;
  margin-bottom: 10px;
}
.small-muted { color: rgba(238,242,255,0.70); font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Units (reasonable defaults for DrivAer parameters)
# NOTE: DrivAerML geo_parameters_all.csv are often normalised / centred.
# We label units for human context; treat absolute magnitudes with caution.
# ============================================================
PARAM_UNITS = {
    "Vehicle_Length": "mm",
    "Vehicle_Width": "mm",
    "Vehicle_Height": "mm",
    "Front_Overhang": "mm",
    "Rear_Overhang": "mm",
    "Vehicle_Ride_Height": "mm",
    "Front_Planview": "mm",
    "Greenhouse_Tapering": "mm",
    "Decklid_Height": "mm",
    "Rearend_tapering": "mm",
    "Vehicle_Pitch": "deg",
    "Hood_Angle": "deg",
    "Approach_Angle": "deg",
    "Windscreen_Angle": "deg",
    "Backlight_Angle": "deg",
    "Rear_Diffusor_Angle": "deg",
}

# ============================================================
# Helpers
# ============================================================
def pretty_param_name(s: str) -> str:
    return s.replace("_", " ")

def fmt_value(value: float, unit: str) -> str:
    if unit == "deg":
        return f"{value:.2f}°"
    if unit == "mm":
        return f"{value:.1f} mm"
    return f"{value:.4f}"

def pct_change(new: float, base: float) -> float:
    if base == 0:
        return np.nan
    return (new - base) / abs(base) * 100.0

def ensemble_predict(models, X: pd.DataFrame):
    preds = np.stack([m.predict(X) for m in models], axis=0)  # (n_models, n_samples, n_targets)
    return preds.mean(axis=0), preds.std(axis=0)

def reset_offsets(cols):
    for c in cols:
        st.session_state[f"off_{c}"] = 0.0

@st.cache_resource
def load_models_and_config():
    cfg_path = MODELS_DIR / "ui_config_4targets.json"
    if not cfg_path.exists():
        st.error("Missing `models/ui_config_4targets.json`. Commit it to the repo and redeploy.")
        st.stop()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_paths = [MODELS_DIR / f"ridge4_ensemble_{m:02d}.joblib" for m in range(N_MODELS)]
    missing = [str(p) for p in model_paths if not p.exists()]
    if missing:
        st.error("Some model files are missing from `models/`. Commit these and redeploy:")
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

def normalise_final_metrics(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Accepts:
      target, test_MAE, test_RMSE, test_R2
    or:
      target, MAE, RMSE, R2
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "test_MAE" in df.columns:
        df = df.rename(columns={"test_MAE": "MAE", "test_RMSE": "RMSE", "test_R2": "R2"})
    keep = [c for c in ["target", "MAE", "RMSE", "R2"] if c in df.columns]
    if "target" not in keep:
        return df
    return df[keep]

def compact_best_vs_ridge(df_baselines: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    Shows best RMSE model per target plus Ridge result used in demo.
    Expected columns: model, target, MAE, RMSE, R2
    """
    df = df_baselines.copy()
    for c in ["MAE", "RMSE", "R2"]:
        df[c] = df[c].astype(float)

    best = (
        df.sort_values(["target", "RMSE"])
          .groupby("target", as_index=False)
          .head(1)
          .rename(columns={"model": "best_model", "RMSE": "best_RMSE", "R2": "best_R2"})
    )[["target", "best_model", "best_RMSE", "best_R2"]]

    ridge = df[df["model"].str.contains("ridge", case=False, na=False)].copy()
    ridge = (
        ridge.sort_values(["target", "RMSE"])
             .groupby("target", as_index=False)
             .head(1)
             .rename(columns={"RMSE": "ridge_RMSE", "R2": "ridge_R2"})
    )[["target", "ridge_RMSE", "ridge_R2"]]

    out = best.merge(ridge, on="target", how="left")
    out = out.set_index("target").reindex(targets).reset_index()
    return out

# ============================================================
# Load assets
# ============================================================
cfg, models = load_models_and_config()

feature_cols = cfg["feature_cols"]                    # full feature list
slider_features = cfg.get("slider_features", feature_cols)  # top 8 (you set in config)
targets = cfg["targets"]                               # ["cd","cl","clf","clr"]

baseline = cfg["baseline"]                             # dict of baseline feature values
smin = cfg["slider_min"]                               # dict of per-feature min
smax = cfg["slider_max"]                               # dict of per-feature max

thr = cfg.get("uncertainty_thresholds", {})            # expects keys like "cd_std_p90"
baseline_outputs = cfg["baseline_outputs"]             # baseline prediction per target

# Optional metadata (if present in config)
split_note = cfg.get("split_note", "Group split by run (unseen geometries held out).")
n_used = cfg.get("n_used", None)
n_train = cfg.get("n_train", None)
n_test = cfg.get("n_test", None)

df_final = normalise_final_metrics(try_load_csv(MODELS_DIR / "metrics_final_test_ridge.csv"))
df_base = try_load_csv(MODELS_DIR / "metrics_baselines.csv")
df_thr = try_load_csv(MODELS_DIR / "uncertainty_thresholds_p90.csv")

# ============================================================
# Header
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("DrivAerML aero surrogate — demo")
st.markdown(
    """
**What this demo does**
- Predicts **cd, cl, clf, clr** from a set of DrivAer geometry parameters.

**How it works**
- Uses an **ensemble of Ridge regression pipelines** trained on **DrivAerML** (500 CFD-tested DrivAer variants).

**How to read the sliders**
- Slider values are **geometry deltas (Δ)** relative to a baseline configuration (dataset mean).
"""
)
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Sidebar controls ONLY
# ============================================================
params = {}

def slider_for(col: str):
    unit = PARAM_UNITS.get(col, "")
    base_val = float(baseline[col])
    left = float(smin[col] - base_val)
    right = float(smax[col] - base_val)

    state_key = f"off_{col}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0.0

    label = f"{pretty_param_name(col)} ({unit})" if unit else pretty_param_name(col)

    offset = st.slider(
        label,
        left,
        right,
        float(st.session_state[state_key]),
        0.001,
        key=state_key,
    )
    current_val = base_val + offset
    params[col] = current_val

    # Current setting only (no “range” text)
    if unit:
        st.caption(f"Current setting: **{fmt_value(current_val, unit)}**")
    else:
        st.caption(f"Current setting: **{current_val:.4f}**")

with st.sidebar:
    st.header("Controls")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset sliders"):
            reset_offsets(feature_cols)
            st.rerun()
    with c2:
        compute = st.button("Compute", type="primary")

    st.divider()

    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.subheader("Key parameters")
    st.markdown('<div class="small-muted">Highlighted as the most influential set used in this demo.</div>', unsafe_allow_html=True)
    for col in slider_features:
        slider_for(col)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("All parameters")
    st.markdown('<div class="small-muted">Included for completeness.</div>', unsafe_allow_html=True)
    for col in [c for c in feature_cols if c not in slider_features]:
        slider_for(col)

    st.divider()
    st.caption("Built by ebprasad")

# Fill any missing (shouldn’t happen) with baseline
full_params = dict(baseline)
full_params.update(params)

# ============================================================
# Not computed yet
# ============================================================
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
        st.caption(
            "As a rule of thumb, the ensemble tends to agree more near the baseline, and less as you push towards the edges of dataset coverage."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ============================================================
# Predict
# ============================================================
with st.spinner("Running surrogate prediction…"):
    X = pd.DataFrame([full_params])[feature_cols]
    mean, std = ensemble_predict(models, X)

pred = {t: float(mean[0, i]) for i, t in enumerate(targets)}
unc = {t: float(std[0, i]) for i, t in enumerate(targets)}

delta_abs = {t: pred[t] - baseline_outputs[t] for t in targets}
delta_pct = {t: pct_change(pred[t], baseline_outputs[t]) for t in targets}

# ============================================================
# Main layout
# ============================================================
left, right = st.columns([1.25, 0.75])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Predicted coefficients")

    # Keep the 2x2 “square” format
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    r1c1.metric("cd", f"{pred['cd']:.6f}", delta=f"{delta_pct['cd']:+.2f}%")
    r1c2.metric("cl", f"{pred['cl']:.6f}", delta=f"{delta_pct['cl']:+.2f}%")
    r2c1.metric("clf", f"{pred['clf']:.6f}", delta=f"{delta_pct['clf']:+.2f}%")
    r2c2.metric("clr", f"{pred['clr']:.6f}", delta=f"{delta_pct['clr']:+.2f}%")

    st.caption("Baseline values are the model prediction at the baseline geometry (dataset mean).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Baseline vs predicted (absolute deltas)")
    tbl = pd.DataFrame(
        {
            "baseline": [baseline_outputs[t] for t in targets],
            "predicted": [pred[t] for t in targets],
            "delta": [delta_abs[t] for t in targets],
        },
        index=targets,
    )
    st.dataframe(
        tbl.style.format({"baseline": "{:.6f}", "predicted": "{:.6f}", "delta": "{:+.6f}"}),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Uncertainty / reliability")

    with st.expander("What do “std” and “p90” mean?", expanded=False):
        st.markdown(
            """
- **std (standard deviation):** Here it’s the disagreement between models in the ensemble.  
  If they broadly agree, std stays small, which is usually reassuring.

- **p90:** The 90th percentile of std measured on a held-out calibration split.  
  If std is above p90, it’s in the most uncertain ~10% of cases, so it gets flagged.

This is a practical reliability check, not a guaranteed probability.
"""
        )

    def thr_for(t: str):
        key = f"{t}_std_p90"
        try:
            return float(thr[key]) if key in thr else None
        except Exception:
            return None

    def show_row(t: str) -> bool:
        thr_val = thr_for(t)
        flagged = False
        if thr_val is not None:
            flagged = unc[t] > thr_val

        icon = "⚠️" if flagged else "✅"
        status = "Low confidence" if flagged else "High confidence"

        if thr_val is None:
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

    st.caption(
        "As a rule of thumb, the ensemble tends to agree more near the baseline, and less as you push towards the edges of dataset coverage."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Technical evaluation (model card) — sentences + compact tables
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Technical evaluation (model card)")

st.markdown("**Model**")
st.markdown(
    f"""
- Ensemble size: **{N_MODELS}** (Ridge regression pipelines with feature scaling)
- Features: **{len(feature_cols)}** geometry parameters
- Targets: **{', '.join(targets)}**
- Split / leakage control: **{split_note}**
"""
)

if any(v is not None for v in [n_used, n_train, n_test]):
    st.markdown("**Data**")
    lines = []
    if n_used is not None:
        lines.append(f"Geometries used: **{n_used}** (after removing runs with missing force coefficients)")
    if n_train is not None and n_test is not None:
        lines.append(f"Train/test: **{n_train}/{n_test}** (held-out runs)")
    st.markdown("- " + "\n- ".join(lines))

st.markdown("**Final test performance (held-out runs)**")

if df_final is None or df_final.empty or not set(["target", "RMSE", "R2"]).issubset(df_final.columns):
    st.info("Final test metrics aren’t available in this deployment.")
else:
    dfp = df_final.set_index("target").reindex(targets).reset_index()

    # Small, useful table only
    show_cols = [c for c in ["target", "RMSE", "R2", "MAE"] if c in dfp.columns]
    df_show = dfp[show_cols].copy()

    st.dataframe(
        df_show.style.format(
            {
                "MAE": "{:.6f}",
                "RMSE": "{:.6f}",
                "R2": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    r2_min, r2_max = float(dfp["R2"].min()), float(dfp["R2"].max())
    rmse_min, rmse_max = float(dfp["RMSE"].min()), float(dfp["RMSE"].max())

    best = dfp.sort_values("R2", ascending=False).iloc[0]
    worst = dfp.sort_values("R2", ascending=True).iloc[0]

    st.markdown(
        f"""
Across the four targets, performance on the held-out runs lands at **R² = {r2_min:.3f}–{r2_max:.3f}** and **RMSE = {rmse_min:.5f}–{rmse_max:.5f}**.
On this split, the strongest output is **{best['target']}** (R² **{float(best['R2']):.4f}**, RMSE **{float(best['RMSE']):.6f}**), and the toughest is **{worst['target']}** (R² **{float(worst['R2']):.4f}**, RMSE **{float(worst['RMSE']):.6f}**).
"""
    )

st.divider()
st.markdown("**Baseline model comparisons**")

if df_base is None or df_base.empty or not set(["model", "target", "RMSE", "R2", "MAE"]).issubset(df_base.columns):
    st.info("Baseline comparison metrics aren’t available in this deployment.")
else:
    compact = compact_best_vs_ridge(df_base, targets)

    st.markdown(
        "Below is a compact comparison: the **best RMSE model per target** from the baseline sweep, alongside the Ridge figures used in this demo."
    )

    compact_show = compact.rename(
        columns={
            "target": "Target",
            "best_model": "Best model (by RMSE)",
            "best_RMSE": "Best RMSE",
            "best_R2": "Best R²",
            "ridge_RMSE": "Ridge RMSE",
            "ridge_R2": "Ridge R²",
        }
    )

    st.dataframe(
        compact_show.style.format(
            {"Best RMSE": "{:.6f}", "Best R²": "{:.4f}", "Ridge RMSE": "{:.6f}", "Ridge R²": "{:.4f}"}
        ),
        use_container_width=True,
    )

st.divider()
st.markdown("**Calibration thresholds (p90)**")

# Prefer thresholds actually used by the app config (thr dict)
thr_used = {}
for t in ["cd", "cl", "clf", "clr"]:
    k = f"{t}_std_p90"
    if k in thr:
        try:
            thr_used[k] = float(thr[k])
        except Exception:
            pass

if thr_used:
    parts = [f"{t}: {thr_used[f'{t}_std_p90']:.2e}" for t in ["cd", "cl", "clf", "clr"] if f"{t}_std_p90" in thr_used]
    st.markdown(
        "Low-confidence flags trigger when the ensemble standard deviation exceeds the p90 threshold. "
        f"Thresholds in use: **{', '.join(parts)}**."
    )
else:
    st.info("No p90 thresholds were found in the current config.")

# Keep the full table out of the way (still available for technical readers)
if df_thr is not None and not df_thr.empty:
    with st.expander("Show calibration table (p90 thresholds)"):
        st.dataframe(df_thr.style.format("{:.6e}"), use_container_width=True)

st.divider()
st.markdown("**Engineering interpretation / limitations**")
st.markdown(
    """
- This surrogate is intended for **rapid screening** and **trend exploration**. If it’s a decision-making case, it still deserves a higher-fidelity check.
- Expected failure mode is **extrapolation** (inputs near/outside training coverage), which is why the app surfaces an **ensemble-disagreement flag**.
- The “uncertainty” here is **model disagreement**, not a guaranteed probability. It’s a pragmatic *trust-but-verify* indicator.
"""
)

st.caption("Built by ebprasad · DrivAerML dataset by N. Ashton et al. (citation in the repo)")
st.markdown("</div>", unsafe_allow_html=True)
