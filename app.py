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
N_MODELS = 30

st.set_page_config(page_title="DrivAerML aero surrogate — demo", layout="wide")

# ============================================================
# Styling (hide header bar + lock sidebar open + modern theme)
# ============================================================
st.markdown(
    """
<style>
/* Hide Streamlit header (removes the top bar) + menus */
header[data-testid="stHeader"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hide ALL sidebar collapse / expand controls across Streamlit versions */
button[data-testid="stSidebarCollapseButton"] { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stSidebarNav"] button { display: none !important; }

/* Extra: hide any floating chevron/controls Streamlit injects */
div[data-testid="stSidebar"] button[aria-label*="sidebar"] { display: none !important; }
div[data-testid="stSidebar"] button[title*="sidebar"] { display: none !important; }
div[data-testid="stSidebar"] button[aria-label*="Collapse"] { display: none !important; }
div[data-testid="stSidebar"] button[aria-label*="Expand"] { display: none !important; }

/* As a last resort, hide the entire collapsed-control container if it exists */
div[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Typography: Apple-ish system stack */
html, body, [class*="css"]  {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
               "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  letter-spacing: 0.1px;
}

/* Modern background */
.stApp {
  background: radial-gradient(1200px 700px at 15% 10%, rgba(125,90,255,0.28) 0%,
              rgba(20,32,60,0.92) 35%,
              rgba(8,10,16,1) 78%);
  color: #eef2ff;
}

/* Layout */
.block-container {
  max-width: 1180px;
  padding-top: 1.2rem;
  padding-bottom: 2.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.05);
  border-right: 1px solid rgba(255,255,255,0.10);
}

/* Cards */
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: 0 14px 40px rgba(0,0,0,0.28);
  backdrop-filter: blur(10px);
}

/* Subtle entrance animation */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.card { animation: fadeUp 220ms ease-out; }

/* Buttons */
.stButton>button {
  border-radius: 12px;
  padding: 0.56rem 0.95rem;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}
.stButton>button:hover {
  border: 1px solid rgba(166,200,255,0.8);
  transform: translateY(-1px);
  transition: 140ms ease;
}

/* Primary (Compute) */
button[kind="primary"] {
  background: linear-gradient(90deg, #ff3b30 0%, #ff2d55 100%) !important;
  border: none !important;
  color: white !important;
  box-shadow: 0 10px 24px rgba(255,45,85,0.25);
}
button[kind="primary"]:hover { filter: brightness(0.96); }

/* Metric typography */
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stMetricDelta"] { font-size: 1.05rem; }

/* Highlight box for key parameters */
.highlight-box {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(170,200,255,0.38);
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
# Units (angles in deg; lengths in mm — sensible defaults)
# ============================================================
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

def fmt_delta(value: float, unit: str) -> str:
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
    preds = np.stack([m.predict(X) for m in models], axis=0)  # (n_models, n_samples, n_targets)
    return preds.mean(axis=0), preds.std(axis=0)

def reset_offsets(cols):
    for c in cols:
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

def normalise_final_metrics(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Expect either:
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

    # Keep only key columns if present
    keep = [c for c in ["target", "MAE", "RMSE", "R2"] if c in df.columns]
    if not keep or "target" not in keep:
        return df  # fallback if schema is unexpected
    df = df[keep]
    return df

def summarise_range(series: pd.Series, fmt: str) -> str:
    vmin = float(series.min())
    vmax = float(series.max())
    return f"{fmt.format(vmin)}–{fmt.format(vmax)}"

def make_compact_baseline_table(df_baselines: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    Compact comparison: best RMSE model per target + Ridge (closest ridge entry).
    Assumes columns: model, target, MAE, RMSE, R2
    """
    df = df_baselines.copy()
    for c in ["MAE", "RMSE", "R2"]:
        df[c] = df[c].astype(float)

    best = (
        df.sort_values(["target", "RMSE"], ascending=[True, True])
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

feature_cols = cfg["feature_cols"]
slider_features = cfg.get("slider_features", feature_cols)  # top 8 (you set this in config)
targets = cfg["targets"]  # ["cd","cl","clf","clr"]

baseline = cfg["baseline"]
smin = cfg["slider_min"]
smax = cfg["slider_max"]

thr = cfg.get("uncertainty_thresholds", {})  # expects *_std_p90 for cd/cl/clf/clr
baseline_outputs = cfg["baseline_outputs"]

# Optional extra metadata in config
split_note = cfg.get("split_note", "Group split by run (unseen geometries held out).")
n_used = cfg.get("n_used", None)
n_train = cfg.get("n_train", None)
n_test = cfg.get("n_test", None)

# Metrics CSVs (commit under models/)
df_final = normalise_final_metrics(try_load_csv(MODELS_DIR / "metrics_final_test_ridge.csv"))
df_base = try_load_csv(MODELS_DIR / "metrics_baselines.csv")
df_thr_csv = try_load_csv(MODELS_DIR / "uncertainty_thresholds_p90.csv")

# ============================================================
# Header
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("DrivAerML aero surrogate — demo")
st.markdown(
    """
**What this demo does**
- Predicts **cd, cl, clf, clr** from a compact set of geometry parameters.

**How it works**
- Runs an **ensemble of Ridge regression pipelines** trained on **DrivAerML** (500 CFD-tested DrivAer geometry variants).

**How to read the sliders**
- Slider values are **geometry deltas (Δ)** relative to the baseline DrivAer shape.
"""
)
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Sidebar (ALL parameters; top 8 highlighted)
# ============================================================
def slider_for(col: str, params_out: dict):
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
    params_out[col] = current_val

    # show current value only (no "range" copy)
    st.caption(f"Current setting: **{fmt_value(current_val, unit)}**")

with st.sidebar:
    st.header("Controls")
    st.caption("Adjust geometry relative to the baseline (dataset mean), then click **Compute**.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset sliders"):
            reset_offsets(feature_cols)  # reset all
            st.rerun()
    with c2:
        compute = st.button("Compute", type="primary")

    st.divider()

    params = {}

    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.subheader("Key parameters")
    st.markdown(
        '<div class="small-muted">These are highlighted because they tend to drive most of the variation in this surrogate.</div>',
        unsafe_allow_html=True,
    )
    for col in slider_features:
        slider_for(col, params)
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("All parameters")
    st.markdown('<div class="small-muted">The rest are included for completeness.</div>', unsafe_allow_html=True)
    others = [c for c in feature_cols if c not in slider_features]
    for col in others:
        slider_for(col, params)

    st.divider()
    st.caption("Built by ebprasad")

# Build full parameter vector (non-slider values are baseline unless changed)
full_params = dict(baseline)
full_params.update(params)

# ============================================================
# If not computed yet
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

    # 2x2 layout
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
    st.dataframe(tbl.style.format({"baseline": "{:.6f}", "predicted": "{:.6f}", "delta": "{:+.6f}"}), use_container_width=True)
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

    def thr_for(t: str) -> float | None:
        k = f"{t}_std_p90"
        if k in thr:
            try:
                return float(thr[k])
            except Exception:
                return None
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
# Technical evaluation (model card) — no CSV dumps
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
    bits = []
    if n_used is not None:
        bits.append(f"Geometries used: **{n_used}** (after removing missing runs)")
    if n_train is not None and n_test is not None:
        bits.append(f"Train/test: **{n_train}/{n_test}**")
    st.markdown("- " + "\n- ".join(bits))

st.markdown("**Final test performance (held-out runs)**")

if df_final is None or df_final.empty or not set(["target", "RMSE", "R2"]).issubset(set(df_final.columns)):
    st.warning("Final test metrics aren’t available in this deployment.")
else:
    df_final = df_final.copy()
    # order by target list
    df_final = df_final.set_index("target").reindex(targets).reset_index()

    # short table
    show_cols = [c for c in ["target", "MAE", "RMSE", "R2"] if c in df_final.columns]
    perf_tbl = df_final[show_cols]

    st.dataframe(
        perf_tbl.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}", "R2": "{:.4f}"}),
        use_container_width=True,
    )

    r2_range = summarise_range(df_final["R2"].astype(float), "{:.3f}")
    rmse_range = summarise_range(df_final["RMSE"].astype(float), "{:.5f}")

    best = df_final.sort_values("R2", ascending=False).iloc[0]
    worst = df_final.sort_values("R2", ascending=True).iloc[0]

    st.markdown(
        f"""
On the held-out test split, the surrogate achieves **R² = {r2_range}** across the four targets, with **RMSE = {rmse_range}**.
Best-performing output is **{best['target']}** (R² **{float(best['R2']):.4f}**, RMSE **{float(best['RMSE']):.6f}**), while **{worst['target']}** is the toughest of the four on this split (R² **{float(worst['R2']):.4f}**, RMSE **{float(worst['RMSE']):.6f}**).
"""
    )

st.divider()
st.markdown("**Baselines and comparisons**")

if df_base is None or df_base.empty or not set(["model", "target", "RMSE", "R2", "MAE"]).issubset(set(df_base.columns)):
    st.info("Baseline comparison metrics aren’t available in this deployment.")
else:
    compact = make_compact_baseline_table(df_base, targets)

    st.markdown(
        "A compact comparison is shown below: the **best RMSE model per target** from the baseline sweep, alongside the Ridge figures used in this demo."
    )

    show = compact.rename(
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
        show.style.format({"Best RMSE": "{:.6f}", "Best R²": "{:.4f}", "Ridge RMSE": "{:.6f}", "Ridge R²": "{:.4f}"}),
        use_container_width=True,
    )

    # Mild, non-boasty note
    close = 0
    total = 0
    for _, row in compact.iterrows():
        if pd.notna(row.get("ridge_RMSE")) and pd.notna(row.get("best_RMSE")):
            total += 1
            if float(row["ridge_RMSE"]) <= float(row["best_RMSE"]) * 1.05:
                close += 1
    if total > 0:
        st.caption(f"As a sense-check, Ridge sits within ~5% of the best RMSE on **{close}/{total}** targets in this sweep.")

st.divider()
st.markdown("**Calibration thresholds (p90)**")

# Prefer thresholds used by the app (from config), then show CSV as reference if present.
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
        "Low-confidence flags are triggered when the ensemble standard deviation exceeds the p90 threshold. "
        f"Thresholds in use: **{', '.join(parts)}**."
    )
    st.dataframe(pd.DataFrame([thr_used]).style.format("{:.6e}"), use_container_width=True)
else:
    st.info("No p90 thresholds were found in the current config.")

if df_thr_csv is not None and not df_thr_csv.empty:
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
