"""
CollideX Dashboard — Main Entry Point
========================================
Uses st.navigation() for URL-based routing so browser
Back / Forward buttons work natively.

Run:
    cd cosmicguard/dashboard
    streamlit run app.py
"""

import sys
import os
import importlib.util

# ---------------------------------------------------------------------------
# Path bootstrap — make dashboard root importable
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

import streamlit as st
from assets.styles import GLOBAL_CSS
from components.ui import sidebar_logo
from config import APP_TITLE, APP_SUBTITLE, APP_ICON

# ---------------------------------------------------------------------------
# Page Config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} — Space Debris Collision Prediction",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": f"### {APP_TITLE}\n{APP_SUBTITLE}",
    }
)

# ---------------------------------------------------------------------------
# Inject Global CSS + Custom Nav Styling
# ---------------------------------------------------------------------------
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Make sidebar nav buttons look exactly like the image (link-style, centered)
st.markdown("""
<style>
/* Remove button borders and backgrounds from sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #8BA3C7 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 0px !important;
    width: 100% !important;
    text-align: center !important;
    box-shadow: none !important;
    transition: color 0.2s ease !important;
    margin: 2px 0 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    color: #00D4FF !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* Active page — highlighted cyan text, no background */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    color: #00D4FF !important;
    font-weight: 700 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    color: #00D4FF !important;
    background: transparent !important;
}
/* Remove focus outline */
section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page Runners — each loads its module and calls render()
# ---------------------------------------------------------------------------
def _make_runner(rel_path: str):
    """Return a callable that loads the page module and calls render()."""
    def _runner():
        abs_path = os.path.join(DASHBOARD_DIR, rel_path)
        spec = importlib.util.spec_from_file_location("_page_module", abs_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.render()
    return _runner

# ---------------------------------------------------------------------------
# Define Pages for st.navigation()
# Each page gets its own URL — browser Back/Forward work natively
# ---------------------------------------------------------------------------
NAV_CONFIG = [
    ("Home",               "pages/1_home.py",             "◎",  "home"),
    ("Prediction",         "pages/2_prediction.py",       "⚡", "prediction"),
    ("Evaluation",         "pages/3_evaluation.py",       "◈",  "evaluation"),
    ("Visual Analytics",   "pages/4_visual_analytics.py", "◉",  "visual_analytics"),
    ("Prediction History", "pages/5_history.py",          "◷",  "history"),
    ("Downloads",          "pages/6_downloads.py",        "↓",  "downloads"),
    ("About",              "pages/7_about.py",            "◌",  "about"),
]

nav_pages = [
    st.Page(
        _make_runner(path),
        title=name,
        icon=icon,
        url_path=url,
    )
    for name, path, icon, url in NAV_CONFIG
]

# Use st.navigation with position="hidden" so we draw our own sidebar
pg = st.navigation(nav_pages, position="hidden")

# ---------------------------------------------------------------------------
# Sidebar — styled exactly like the reference image
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo / branding
    st.markdown(sidebar_logo(), unsafe_allow_html=True)

    # "NAVIGATION" label
    st.markdown("""
    <div style="padding:12px 20px 8px;">
        <div style="font-size:9px;color:#1A3A5C;text-transform:uppercase;
                    letter-spacing:2px;font-weight:700;">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    # Nav buttons — look like the image (centered link-style)
    for page in nav_pages:
        is_active = pg.title == page.title
        btn = st.button(
            f"{page.icon}  {page.title}",
            key=f"nav_{page.title}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        )
        if btn:
            st.switch_page(page)

    # System Status
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid #1A3A5C;padding:16px 20px 0;">
        <div style="font-size:9px;color:#1A3A5C;text-transform:uppercase;
                    letter-spacing:2px;font-weight:700;margin-bottom:10px;">
            System Status
        </div>
    """, unsafe_allow_html=True)

    from config import COLLISION_REPORT_CSV, METRICS_CSV, DEFAULT_TLE_FILE
    _checks = [
        ("Predict Engine", os.path.exists(os.path.join(DASHBOARD_DIR, "..", "scripts", "predict.py"))),
        ("TLE Catalog",    os.path.exists(DEFAULT_TLE_FILE)),
        ("Results",        os.path.exists(COLLISION_REPORT_CSV)),
        ("Metrics",        os.path.exists(METRICS_CSV)),
    ]
    for lbl, ok in _checks:
        dot_c = "#00FF9F" if ok else "#FF3860"
        state = "OK" if ok else "MISSING"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:3px 0;">
            <span style="width:6px;height:6px;border-radius:50%;
                         background:{dot_c};display:inline-block;"></span>
            <span style="font-size:11px;color:#4A6A8A;flex:1;">{lbl}</span>
            <span style="font-size:10px;color:{dot_c};
                         font-family:'JetBrains Mono',monospace;">{state}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Version footer
    st.markdown("""
    <div style="text-align:center;padding:16px 0 4px;">
        <div style="font-size:10px;color:#1A3A5C;font-family:'JetBrains Mono',monospace;">
            CollideX v1.0 · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run the selected page
# ---------------------------------------------------------------------------
pg.run()
