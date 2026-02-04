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

def normalise_final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts metrics_final_test_ridge.csv into a consistent schema:
    columns: target, MAE, RMSE, R2
    Accepts either (test_MAE/test_RMSE/test_R2) or (MAE/RMSE/R2).
    """
    if df is None or df.empty:
        return df

    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols).copy()

    mapping = {}
    if "test_MAE" in df.columns: mapping["test_MAE"] = "MAE"
    if "test_RMSE" in df.columns: mapping["test_RMSE"] = "RMSE"
    if "test_R2" in df.columns: mapping["test_R2"] = "R2"

    df = df.rename(columns=mapping)

    keep = [c for c in ["target", "MAE", "RMSE", "R2"] if c in df.columns]
    df = df[keep].copy()

    # enforce ordering by target list if present later
    return df

def summarise_r2_range(df: pd.DataFrame) -> str:
    r2 = df["R2"].astype(float)
    return f"{r2.min():.3f}–{r2.max():.3f}"

def summarise_rmse_range(df: pd.DataFrame) -> str:
    rmse = df["RMSE"].astype(float)
    return f"{rmse.min():.5f}–{rmse.max():.5f}"

def make_best_model_table(df_baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a compact table:
    per target -> best RMSE model, RMSE, R2, plus ridge_scaled row for comparison.
    """
    df = df_baselines.copy()
    df["RMSE"] = df["RMSE"].astype(float)
    df["R2"] = df["R2"].astype(float)
    df["MAE"] = df["MAE"].astype(float)

    # Best by RMSE per target
    best = (
        df.sort_values(["target", "RMSE"], ascending=[True, True])
          .groupby("target", as_index=False)
          .head(1)
          .rename(columns={"model": "best_model", "RMSE": "best_RMSE", "R2": "best_R2"})
    )[["target", "best_model", "best_RMSE", "best_R2"]]

    # Ridge rows for same targets (if present)
    ridge = df[df["model"].str.contains("ridge", case=False, na=False)].copy()
    ridge = (
        ridge.sort_values(["target", "RMSE"])
             .groupby("target", as_index=False)
             .head(1)
             .rename(columns={"RMSE": "ridge_RMSE", "R2": "ridge_R2"})
    )[["target", "ridge_RMSE", "ridge_R2"]]

    out = best.merge(ridge, on="target", how="left")
    return out

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
thr = cfg["uncertainty_thresholds"]  # used for flags
baseline_outputs = cfg["baseline_outputs"]

# Optional metadata you can add into ui_config_4targets.json
split_note = cfg.get("split_note", "Group split by run (unseen geometries held out).")
n_used = cfg.get("n_used", None)
n_train = cfg.get("n_train", None)
n_test = cfg.get("n_test", None)

# Metrics files (commit these under models/)
df_final_raw = try_load_csv(MODELS_DIR / "metrics_final_test_ridge.csv")
df_base = try_load_csv(MODELS_DIR / "metrics_baselines.csv")
df_thr_csv = try_load_csv(MODELS_DIR / "uncertainty_thresholds_p90.csv")

df_final = normalise_final_metrics(df_final_raw)

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
        st.caption(f"Current setting: **{fmt_with_unit(current_val, unit)}**")

    st.divider()
    st.caption("Built by ebprasad")

# Fill non-slider parameters with baseline
full_params = dict(baseline)
full_params.update(params)

# ----------------------------
# If not computed yet
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
# Predict
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
    st.dataframe(tbl.style.format({"baseline": "{:.6f}", "predicted": "{:.6f}", "delta": "{:+.6f}"}))
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
# Technical evaluation (curated, not CSV dumps)
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Technical evaluation (model card)")

st.markdown("**Model**")
st.markdown(
    f"""
- Ensemble size: **{N_MODELS}** (Ridge regression pipelines, each with feature scaling)
- Features: **{len(feature_cols)}** geometry parameters
- Targets: **{', '.join(targets)}**
- Split / leakage control: **{split_note}**
"""
)

if any(v is not None for v in [n_used, n_train, n_test]):
    st.markdown("**Data**")
    bits = []
    if n_used is not None:
        bits.append(f"Geometries used: **{n_used}** (after removing missing runs)")
    if n_train is not None and n_test is not None:
        bits.append(f"Train/test: **{n_train}/{n_test}**")
    st.markdown("- " + "\n- ".join(bits))

st.markdown("**Final test performance (held-out runs)**")

if df_final is None or df_final.empty:
    st.warning("Final test metrics aren’t available in this deployment.")
else:
    # Order rows by target list
    df_final = df_final.set_index("target").reindex(targets).reset_index()

    # Small, tidy table (only key numbers)
    perf_tbl = df_final.copy()
    st.dataframe(
        perf_tbl.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}", "R2": "{:.4f}"}),
        use_container_width=True,
    )

    # Natural-language technical summary (British tone, but properly technical)
    r2_range = summarise_r2_range(df_final)
    rmse_range = summarise_rmse_range(df_final)
    best = df_final.sort_values("R2", ascending=False).iloc[0]
    worst = df_final.sort_values("R2", ascending=True).iloc[0]

    st.markdown(
        f"""
On the held-out test split, the surrogate achieves **R² = {r2_range}** across the four targets, with **RMSE = {rmse_range}**.
Best-performing output here is **{best['target']}** (R² **{best['R2']:.4f}**, RMSE **{best['RMSE']:.6f}**), while **{worst['target']}** is the toughest of the four on this split (R² **{worst['R2']:.4f}**, RMSE **{worst['RMSE']:.6f}**).
"""
    )

st.divider()
st.markdown("**Baselines and comparisons**")

if df_base is None or df_base.empty:
    st.info("Baseline comparison metrics aren’t available in this deployment.")
else:
    # Curated comparison: best model per target vs ridge
    compact = make_best_model_table(df_base).set_index("target").reindex(targets).reset_index()

    st.markdown(
        "Below is a compact comparison: the **best RMSE model per target** (from the baseline sweep), alongside the Ridge ensemble figures."
    )

    show = compact.rename(
        columns={
            "best_model": "Best model (by RMSE)",
            "best_RMSE": "Best RMSE",
            "best_R2": "Best R²",
            "ridge_RMSE": "Ridge RMSE",
            "ridge_R2": "Ridge R²",
        }
    )

    st.dataframe(
        show.style.format(
            {"Best RMSE": "{:.6f}", "Best R²": "{:.4f}", "Ridge RMSE": "{:.6f}", "Ridge R²": "{:.4f}"}
        ),
        use_container_width=True,
    )

    # Short observation line, not braggy
    # Identify whether ridge is best/close
    try:
        close_count = 0
        for _, row in compact.iterrows():
            if pd.notna(row["ridge_RMSE"]) and pd.notna(row["best_RMSE"]):
                if row["ridge_RMSE"] <= row["best_RMSE"] * 1.05:
                    close_count += 1
        st.caption(
            f"As a quick sense-check: Ridge is within ~5% of the best RMSE on **{close_count}/{len(compact)}** targets in this sweep."
        )
    except Exception:
        pass

st.divider()
st.markdown("**Calibration thresholds (p90)**")

# Prefer showing the thresholds actually used for flagging (from ui_config), but also display CSV if present
thr_used = {}
for t in ["cd", "cl", "clf", "clr"]:
    k = f"{t}_std_p90"
    if k in thr:
        thr_used[k] = thr[k]

if thr_used:
    # sentence summary first
    parts = []
    for t in ["cd", "cl", "clf", "clr"]:
        k = f"{t}_std_p90"
        if k in thr_used:
            parts.append(f"{t}: {float(thr_used[k]):.2e}")
    st.markdown(
        "Low-confidence flags are triggered when the ensemble standard deviation exceeds the p90 threshold. "
        f"Thresholds in use: **{', '.join(parts)}**."
    )
else:
    st.info("No p90 thresholds were found in the current config.")

# small table (nice for engineers)
thr_table = pd.DataFrame([thr_used]) if thr_used else pd.DataFrame()
if not thr_table.empty:
    st.dataframe(thr_table.style.format("{:.6e}"), use_container_width=True)

# Also show the original CSV thresholds if present (but tiny)
if df_thr_csv is not None and not df_thr_csv.empty:
    # keep it compact: one row
    st.caption("Reference thresholds table (as generated during calibration):")
    st.dataframe(df_thr_csv.style.format("{:.6e}"), use_container_width=True)

st.divider()
st.markdown("**Engineering interpretation / limitations**")
st.markdown(
    """
- This surrogate is intended for **rapid screening** and **trend exploration**. If it’s a decision-making case, it still deserves a higher-fidelity check.
- Expected failure mode is **extrapolation** (inputs near/outside training coverage), which is why the app surfaces an **ensemble-disagreement flag**.
- The “uncertainty” here is **model disagreement**, not a guaranteed probability. It’s used as a pragmatic *trust-but-verify* indicator.
"""
)

st.caption("Built by ebprasad · DrivAerML dataset by N. Ashton et al. (citation in the repo)")
st.markdown("</div>", unsafe_allow_html=True)
