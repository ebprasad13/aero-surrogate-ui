st.markdown(
    """
<style>
/* Hide Streamlit header + footer (removes the black bar area) */
header[data-testid="stHeader"] { display: none; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Disable sidebar collapse controls (expanded and collapsed) */
button[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Typography: Apple-like system stack */
html, body, [class*="css"]  {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
               "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  letter-spacing: 0.1px;
}

/* Modern background */
.stApp {
  background: radial-gradient(1200px 700px at 15% 10%, rgba(125,90,255,0.28) 0%,
              rgba(20,32,60,0.9) 35%,
              rgba(8,10,16,1) 78%);
  color: #eef2ff;
}

/* Layout padding */
.block-container {
  max-width: 1180px;
  padding-top: 1.2rem;
  padding-bottom: 2.5rem;
}

/* Card */
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

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.05);
  border-right: 1px solid rgba(255,255,255,0.10);
}

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

/* Primary */
button[kind="primary"] {
  background: linear-gradient(90deg, #ff3b30 0%, #ff2d55 100%) !important;
  border: none !important;
  color: white !important;
  box-shadow: 0 10px 24px rgba(255,45,85,0.25);
}
button[kind="primary"]:hover {
  filter: brightness(0.96);
}

/* Metric */
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stMetricDelta"] { font-size: 1.05rem; }

/* Highlight box for "important parameters" */
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
