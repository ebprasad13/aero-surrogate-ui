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
