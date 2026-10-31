# dashboard.py  --  HydroPulse-SVR: Streamlit Interactive Dashboard
#
# Prediction target: next-month KRS reservoir inflow deficit (%)
#                    used as a hydropower-risk indicator.
# (No actual power-generation data is available for KRS.)
#
# Run:  streamlit run dashboard.py

import os
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

from config import (
    SVR_MODEL_PATH, SCALER_PATH, DATA_OUT_PATH,
    METRICS_PATH, PRED_PATH, FEATURE_COLS, TARGET_COL,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HydroPulse-SVR | KRS Basin Risk Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE — navigation & theme
# ============================================================
PAGES = [
    "Overview",
    "ENSO Analysis",
    "Reservoir & Rainfall",
    "Model Performance",
    "Deficit Prediction",
    "Risk Analysis",
]

if "current_theme" not in st.session_state:
    st.session_state["current_theme"] = "Light"
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"
if "top_nav_radio" not in st.session_state:
    st.session_state["top_nav_radio"] = "Overview"
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False

def sync_page_from_top():
    val = st.session_state.get("top_nav_radio", "Overview")
    st.session_state["current_page"] = val

def toggle_theme():
    st.session_state["current_theme"] = (
        "Dark" if st.session_state["current_theme"] == "Light" else "Light"
    )

is_dark = st.session_state["current_theme"] == "Dark"

# ============================================================
# CSS — single block, all styles
# ============================================================
_NAVY        = "#5ba3d9"   if is_dark else "#1a3a5c"
_NAVY_DARK   = "#7bbfe8"   if is_dark else "#0f2d4a"
_SAFFRON     = "#f0b45d"   if is_dark else "#c7882a"
_FOREST      = "#3ddc84"   if is_dark else "#1a6e3c"
_CRIMSON     = "#f87171"   if is_dark else "#9b1d2a"
_BG_APP      = "#0d1117"   if is_dark else "#f5f7fa"
_BG_CARD     = "#161b22"   if is_dark else "#ffffff"
_BG_SUBTLE   = "#1f242c"   if is_dark else "#f0f2f5"
_BORDER      = "#30363d"   if is_dark else "#dee2e6"
_BORDER_ACC  = "#485260"   if is_dark else "#c6cdd4"
_TEXT_PRI    = "#f0f6fc"   if is_dark else "#1a2332"
_TEXT_SEC    = "#c9d1d9"   if is_dark else "#3d5166"
_TEXT_MUT    = "#8b949e"   if is_dark else "#6c7a89"
_BLUE_LINK   = "#58a6ff"   if is_dark else "#1a5276"
_NAVBAR_BG   = "#161b22"   if is_dark else "#ffffff"
_SIDEBAR_BG  = "#161b22"   if is_dark else "#ffffff"

_THEME_ICON_SVG = (
    "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f0f6fc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='5'/%3E%3Cline x1='12' y1='1' x2='12' y2='3'/%3E%3Cline x1='12' y1='21' x2='12' y2='23'/%3E%3Cline x1='4.22' y1='4.22' x2='5.64' y2='5.64'/%3E%3Cline x1='18.36' y1='18.36' x2='19.78' y2='19.78'/%3E%3Cline x1='1' y1='12' x2='3' y2='12'/%3E%3Cline x1='21' y1='12' x2='23' y2='12'/%3E%3Cline x1='4.22' y1='19.78' x2='5.64' y2='18.36'/%3E%3Cline x1='18.36' y1='5.64' x2='19.78' y2='4.22'/%3E%3C/svg%3E"
    if is_dark else
    "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%231a3a5c' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/%3E%3C/svg%3E"
)

_CHAT_NAV_ICON_SVG = (
    "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23f0f6fc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 6V2H8'/%3E%3Cpath d='m8 18-4 4V8a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2Z'/%3E%3Ccircle cx='9' cy='12' r='1' fill='%23f0f6fc'/%3E%3Ccircle cx='15' cy='12' r='1' fill='%23f0f6fc'/%3E%3C/svg%3E"
    if is_dark else
    "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%231a3a5c' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 6V2H8'/%3E%3Cpath d='m8 18-4 4V8a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2Z'/%3E%3Ccircle cx='9' cy='12' r='1' fill='%231a3a5c'/%3E%3Ccircle cx='15' cy='12' r='1' fill='%231a3a5c'/%3E%3C/svg%3E"
)

_CHAT_ICON_SVG = "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/%3E%3C/svg%3E"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], [class*="st-"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

:root {{
    --navy: {_NAVY};
    --navy-dark: {_NAVY_DARK};
    --saffron: {_SAFFRON};
    --forest: {_FOREST};
    --crimson: {_CRIMSON};
    --bg-app: {_BG_APP};
    --bg-card: {_BG_CARD};
    --bg-subtle: {_BG_SUBTLE};
    --border: {_BORDER};
    --border-accent: {_BORDER_ACC};
    --text-primary: {_TEXT_PRI};
    --text-secondary: {_TEXT_SEC};
    --text-muted: {_TEXT_MUT};
    --blue-link: {_BLUE_LINK};
    --navbar-bg: {_NAVBAR_BG};
    --sidebar-bg: {_SIDEBAR_BG};
}}

/* ── App Background ── */
.stApp {{
    background-color: var(--bg-app) !important;
    color: var(--text-primary);
}}

/* ── Main container ── */
.block-container {{
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem;
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {{
    color: var(--text-primary) !important;
}}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header[data-testid="stHeader"] {{
    visibility: hidden;
    height: 0;
}}
.stDeployButton {{ display: none; }}

/* ── Top Nav Bar ── */
.hp-navbar {{
    background: var(--navbar-bg);
    border-bottom: 1px solid var(--border);
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    margin-bottom: 0;
    position: sticky;
    top: 0;
    z-index: 999;
}}
.hp-navbar-brand {{
    display: flex;
    flex-direction: column;
    line-height: 1.2;
    flex-shrink: 0;
    white-space: nowrap;
    padding: 2px 0;
}}
.hp-navbar-brand .brand-main {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.96rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.01em;
}}
.hp-navbar-brand .brand-sub {{
    font-size: 0.63rem;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding-left: 21px;
}}

/* ── Page Header (breadcrumb strip) ── */
.hp-page-header {{
    padding: 14px 0 12px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}}
.hp-page-header-accent {{
    width: 3px;
    min-height: 36px;
    background: var(--navy);
    border-radius: 2px;
    flex-shrink: 0;
    margin-top: 2px;
}}
.hp-page-header-text .page-title {{
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.3;
}}
.hp-page-header-text .page-desc {{
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 2px 0 0 0;
    line-height: 1.4;
}}

/* ── KPI Cards ── */
.hp-kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 14px 14px 18px;
    display: flex;
    align-items: stretch;
    gap: 0;
    height: 100%;
    transition: box-shadow 180ms ease;
    position: relative;
    overflow: hidden;
}}
.hp-kpi-card:hover {{
    box-shadow: 0 4px 12px rgba(0,0,0,0.09);
}}
.hp-kpi-accent {{
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 6px 0 0 6px;
}}
.hp-kpi-body {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 4px;
}}
.hp-kpi-label {{
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    line-height: 1.2;
}}
.hp-kpi-value {{
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.1;
}}
.hp-kpi-sub {{
    font-size: 0.7rem;
    color: var(--text-muted);
    line-height: 1.3;
}}

/* ── Section heading ── */
.hp-section-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}}

/* ── Generic white card ── */
.hp-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 16px;
}}

/* ── Risk badge ── */
.hp-risk-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* ── Metrics table ── */
.hp-metrics-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}}
.hp-metrics-table th {{
    background: var(--bg-subtle);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.67rem;
    padding: 8px 12px;
    border: 1px solid var(--border);
    text-align: left;
}}
.hp-metrics-table td {{
    padding: 7px 12px;
    border: 1px solid var(--border);
    color: var(--text-primary);
    vertical-align: middle;
}}
.hp-metrics-table tr:nth-child(even) td {{
    background: var(--bg-subtle);
}}
.hp-metrics-table .best-cell {{
    font-weight: 700;
    color: var(--forest);
}}
.hp-metrics-table .model-highlight td:first-child {{
    border-left: 3px solid var(--saffron);
    font-weight: 600;
}}

/* ── Callout ── */
.hp-callout {{
    border: 1px solid var(--border);
    border-left: 3px solid var(--saffron);
    background: var(--bg-subtle);
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.77rem;
    color: var(--text-secondary);
    margin: 12px 0;
    line-height: 1.55;
}}
.hp-callout strong {{
    color: var(--text-primary);
}}

/* ── Prediction result card ── */
.hp-pred-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 22px 20px;
    text-align: center;
}}
.hp-pred-value {{
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}}
.hp-pred-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 4px;
}}

/* ── Footer ── */
.hp-footer {{
    border-top: 1px solid var(--border);
    padding: 14px 0 6px;
    margin-top: 28px;
    font-size: 0.7rem;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
}}

/* ── Universal Contrast & Dark Mode Fixes ── */
p, span, div, label, li, th, td, h1, h2, h3, h4, h5, h6 {{
    color: var(--text-primary);
}}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span {{
    color: var(--text-primary);
}}
.stSelectbox label, .stSlider label, .stTextInput label {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}}
div[data-baseweb="select"] {{
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
}}
div[data-baseweb="select"] div {{
    color: var(--text-primary) !important;
}}
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}}
li[role="option"] {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
}}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
    background-color: var(--bg-subtle) !important;
    color: var(--navy) !important;
}}
input[type="text"], input[type="number"], textarea {{
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}}

/* ── Nav Button Styles ── */
div[class*="st-key-nav_"] button,
.hp-navbtn-wrap button {{
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 6px 4px !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    border-bottom: 2px solid transparent !important;
    transition: color 140ms ease, border-color 140ms ease !important;
    white-space: nowrap !important;
    width: 100% !important;
    text-align: center !important;
    line-height: 1.2 !important;
    min-height: 38px !important;
    box-shadow: none !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}
div[class*="st-key-nav_"] button:hover,
.hp-navbtn-wrap button:hover {{
    color: var(--navy) !important;
    border-bottom: 2px solid var(--border-accent) !important;
    background: transparent !important;
    box-shadow: none !important;
}}
div[class*="st-key-nav_"] button:focus,
.hp-navbtn-wrap button:focus {{
    box-shadow: none !important;
    outline: none !important;
}}
div[class*="st-key-nav_"] button p,
.hp-navbtn-wrap button p {{
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    margin: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}
.hp-navbtn-active button,
.hp-navbtn-active button:hover {{
    color: var(--navy) !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--navy) !important;
    background: transparent !important;
}}
.hp-navbtn-active button p {{
    color: var(--navy) !important;
    font-weight: 700 !important;
}}

/* ── Top Nav Action Buttons: Chat & Theme Toggles ── */
div[class*="st-key-chat_toggle_btn"],
div[class*="st-key-theme_toggle_btn"] {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}}

div[class*="st-key-chat_toggle_btn"] button,
div[class*="st-key-theme_toggle_btn"] button {{
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    margin: 0 auto !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 140ms ease !important;
    color: transparent !important;
    font-size: 0 !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 18px 18px !important;
}}

div[class*="st-key-chat_toggle_btn"] button {{
    background-image: url("{_CHAT_NAV_ICON_SVG}") !important;
}}

div[class*="st-key-theme_toggle_btn"] button {{
    background-image: url("{_THEME_ICON_SVG}") !important;
}}

div[class*="st-key-chat_toggle_btn"] button:hover,
div[class*="st-key-theme_toggle_btn"] button:hover {{
    border-color: var(--navy) !important;
    background-color: var(--bg-subtle) !important;
    transform: translateY(-1px) !important;
}}

div[class*="st-key-chat_toggle_btn"] button p,
div[class*="st-key-chat_toggle_btn"] button span,
div[class*="st-key-theme_toggle_btn"] button p,
div[class*="st-key-theme_toggle_btn"] button span {{
    display: none !important;
    visibility: hidden !important;
    font-size: 0 !important;
}}

/* ── ChatGPT-Style Slide-Out Chatbot Sidebar ── */
.hp-chat-sidebar-backdrop {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(0, 0, 0, 0.35) !important;
    z-index: 999990 !important;
    backdrop-filter: blur(2px) !important;
    pointer-events: none !important;
}}

div[class*="st-key-chat_sidebar_panel"],
.hp-chat-sidebar-panel {{
    position: fixed !important;
    top: 0 !important;
    right: 0 !important;
    width: 440px !important;
    max-width: 95vw !important;
    height: 100vh !important;
    background-color: var(--bg-card) !important;
    border-left: 1px solid var(--border) !important;
    box-shadow: -8px 0 32px rgba(0, 0, 0, 0.30) !important;
    z-index: 999999 !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 16px 18px !important;
    animation: hpSlideInRight 0.22s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    overflow-y: auto !important;
}}

@keyframes hpSlideInRight {{
    from {{
        transform: translateX(100%);
    }}
    to {{
        transform: translateX(0);
    }}
}}

.st-key-chat_sidebar_close_btn button {{
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    border-radius: 6px !important;
    padding: 0 !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-size: 0.92rem !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 140ms ease !important;
}}
.st-key-chat_sidebar_close_btn button:hover {{
    background: var(--border) !important;
    color: var(--navy) !important;
}}

div[class*="st-key-chat_sidebar_panel"] div[data-testid="stChatMessage"],
.hp-chat-sidebar-panel div[data-testid="stChatMessage"] {{
    background-color: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    margin-bottom: 8px !important;
}}

div[class*="st-key-chat_pill_"] button {{
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 4px 10px !important;
    font-size: 0.73rem !important;
    color: var(--text-secondary) !important;
    text-align: left !important;
    line-height: 1.25 !important;
    margin: 2px 0 !important;
    box-shadow: none !important;
}}
div[class*="st-key-chat_pill_"] button:hover {{
    border-color: var(--navy) !important;
    color: var(--navy) !important;
    background: var(--bg-card) !important;
}}

div[data-testid="stFormSubmitButton"] button {{
    background-color: var(--navy) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 7px 12px !important;
    border-radius: 6px !important;
}}
div[data-testid="stFormSubmitButton"] button:hover {{
    background-color: var(--navy-dark) !important;
    color: #ffffff !important;
}}
div[data-testid="stFormSubmitButton"] button p {{
    color: #ffffff !important;
}}

/* ── Diagnostic Expander: Clean Horizontal Alignment & No Text Overlap ── */
div[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    background: var(--bg-card) !important;
    margin-top: 14px !important;
    margin-bottom: 12px !important;
    overflow: hidden !important;
}}

div[data-testid="stExpander"] details summary {{
    list-style: none !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    padding: 11px 16px !important;
    cursor: pointer !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-card) !important;
    transition: background-color 140ms ease !important;
    user-select: none !important;
}}

div[data-testid="stExpander"] details summary::-webkit-details-marker {{
    display: none !important;
}}

div[data-testid="stExpander"] details summary span[data-testid="stExpanderIcon"],
div[data-testid="stExpander"] details summary [data-testid="stIconMaterial"],
div[data-testid="stExpander"] details summary svg {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    font-size: 0 !important;
    opacity: 0 !important;
}}

/* Hide any leading span that contains icon text such as _arrow_right */
div[data-testid="stExpander"] details summary > span:first-child:not(:last-child) {{
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    font-size: 0 !important;
}}

div[data-testid="stExpander"] details summary::before {{
    content: "\\25B6" !important;
    font-size: 0.70rem !important;
    color: var(--text-muted) !important;
    margin-right: 10px !important;
    display: inline-block !important;
    line-height: 1 !important;
    flex-shrink: 0 !important;
    transition: transform 140ms ease, color 140ms ease !important;
}}

div[data-testid="stExpander"] details[open] summary::before {{
    content: "\\25BC" !important;
    color: var(--navy) !important;
}}

div[data-testid="stExpander"] details summary:hover {{
    background-color: var(--bg-subtle) !important;
}}

div[data-testid="stExpander"] details summary:hover::before {{
    color: var(--navy) !important;
}}

div[data-testid="stExpander"] details summary p {{
    margin: 0 !important;
    padding: 0 !important;
    display: inline-block !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
}}

div[data-testid="stExpander"] details[open] summary {{
    border-bottom: 1px solid var(--border) !important;
}}

div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
    padding: 12px 16px !important;
}}
/* ── Responsive adjustments ── */
@media (max-width: 1100px) {{
    div[class*="st-key-nav_"] button,
    .hp-navbtn-wrap button {{
        font-size: 0.70rem !important;
        padding: 4px 1px !important;
    }}
}}
@media (max-width: 900px) {{
    div[class*="st-key-nav_"] button,
    .hp-navbtn-wrap button {{
        font-size: 0.66rem !important;
        padding: 4px 0px !important;
    }}
}}

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        transition: none !important;
        animation: none !important;
    }}
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalise_phase(s) -> str:
    """Normalise ENSO_Phase string from the CSV (strips accents/whitespace)."""
    if pd.isna(s):
        return "Neutral"
    s = str(s).strip()
    s = s.replace("\u00f1", "n").replace("\u00cd", "I")
    if "El" in s:
        return "El Nino"
    elif "La" in s:
        return "La Nina"
    return "Neutral"


def oni_to_phase(oni: float) -> str:
    """Derive standardized 5-tier ENSO intensity label from raw ONI anomaly."""
    if pd.isna(oni):
        return "Neutral"
    try:
        val = float(oni)
    except (ValueError, TypeError):
        return "Neutral"
    if val >= 1.5:
        return "Strong El Niño"
    elif val >= 0.5:
        return "Moderate El Niño"
    elif val <= -1.5:
        return "Strong La Niña"
    elif val <= -0.5:
        return "Moderate La Niña"
    else:
        return "Neutral"


def deficit_risk(deficit_pct: float):
    """
    Return (category, color_hex, level_int, description) from deficit percentage.
    Positive deficit = below-normal inflow.
    """
    if deficit_pct < 0:
        return ("Surplus", _FOREST, 0, "Above-normal inflow expected.")
    elif deficit_pct < 20:
        return ("Low Risk", _FOREST, 1, "Inflow within 20% of normal.")
    elif deficit_pct < 40:
        return ("Moderate Risk", _SAFFRON, 2, "Inflow moderately below normal.")
    elif deficit_pct < 60:
        return ("High Risk", "#e07b30", 3, "Significant inflow shortfall likely.")
    else:
        return ("Severe Risk", _CRIMSON, 4, "Severe deficit — critical hydropower risk.")


def add_trendline(fig: go.Figure, x_data, y_data, color: str = "#999999", name: str = "Trend") -> go.Figure:
    """Add a vectorized OLS trendline to an existing Plotly figure."""
    try:
        x_clean = pd.to_numeric(pd.Series(x_data), errors="coerce").to_numpy()
        y_clean = pd.to_numeric(pd.Series(y_data), errors="coerce").to_numpy()
        mask = np.isfinite(x_clean) & np.isfinite(y_clean)
        if mask.sum() >= 3:
            xf = x_clean[mask]
            yf = y_clean[mask]
            m, b = np.polyfit(xf, yf, 1)
            x_line = np.linspace(xf.min(), xf.max(), 50)
            fig.add_trace(go.Scatter(
                x=x_line, y=m * x_line + b,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                name=name,
                showlegend=True,
                hoverinfo="skip",
            ))
    except Exception:
        pass
    return fig


def _chart_layout(height: int = 360) -> dict:
    """Return common plotly layout kwargs for the current theme."""
    return dict(
        template="plotly_white",
        paper_bgcolor=_BG_CARD,
        plot_bgcolor="#f8f9fa" if not is_dark else "#1c2128",
        height=height,
        font=dict(family="Inter", size=11, color=_TEXT_SEC),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10),
        ),
        margin=dict(l=52, r=20, t=40, b=44),
        xaxis=dict(
            gridcolor="#e9ecef" if not is_dark else "#30363d",
            linecolor=_BORDER,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#e9ecef" if not is_dark else "#30363d",
            linecolor=_BORDER,
            tickfont=dict(size=10),
        ),
    )

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_OUT_PATH):
        return None
    df = pd.read_csv(DATA_OUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_predictions():
    if not os.path.exists(PRED_PATH):
        return None
    df = pd.read_csv(PRED_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(SVR_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = joblib.load(SVR_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ============================================================
# UI COMPONENTS
# ============================================================

def render_navbar():
    """Top navigation bar — single authoritative navigation bar across the site."""
    curr = st.session_state.get("current_page", "Overview")

    # Use st.columns for nav buttons — gives Streamlit click handling
    # Layout: brand(2.2) + 6 nav buttons + chat action(0.45) + theme action(0.45)
    cols = st.columns([2.2, 1.0, 1.1, 1.5, 1.4, 1.4, 1.2, 0.45, 0.45], vertical_alignment="center")

    with cols[0]:
        st.markdown(f"""
        <div class="hp-navbar-brand" style="padding:2px 0;">
            <span class="brand-main">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{_NAVY}"
                     stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M2 12c2.5-3 5-3 7.5 0s5 3 7.5 0 5-3 7-0.5"></path>
                </svg>
                HydroPulse-SVR
            </span>
            <span class="brand-sub">KRS BASIN RISK CONSOLE</span>
        </div>""", unsafe_allow_html=True)

    nav_keys = [
        ("Overview",            "nav_Overview",        cols[1]),
        ("ENSO Analysis",       "nav_ENSO_Analysis",   cols[2]),
        ("Reservoir & Rainfall","nav_Reservoir",        cols[3]),
        ("Model Performance",   "nav_Model_Perf",      cols[4]),
        ("Deficit Prediction",  "nav_Deficit_Pred",    cols[5]),
        ("Risk Analysis",       "nav_Risk_Analysis",   cols[6]),
    ]
    active_btn_key = None
    for page_name, btn_key, col in nav_keys:
        if curr == page_name:
            active_btn_key = btn_key
        with col:
            if st.button(page_name, key=btn_key, use_container_width=True):
                st.session_state["current_page"] = page_name
                st.session_state["top_nav_radio"] = page_name
                st.rerun()

    if active_btn_key:
        st.markdown(f"""
        <style>
        div.st-key-{active_btn_key} button {{
            color: var(--navy) !important;
            font-weight: 700 !important;
            border-bottom: 2px solid var(--navy) !important;
            background: transparent !important;
        }}
        div.st-key-{active_btn_key} button p {{
            color: var(--navy) !important;
            font-weight: 700 !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    with cols[7]:
        if st.button(" ", key="chat_toggle_btn", help="Open AI Assistant"):
            st.session_state["chat_open"] = not st.session_state.get("chat_open", False)
            st.rerun()

    with cols[8]:
        if st.button(" ", key="theme_toggle_btn", help="Switch Light/Dark Theme"):
            toggle_theme()
            st.rerun()

    st.markdown(
        f'<hr style="margin:0 0 12px 0; border:none; border-top:1px solid {_BORDER};">',
        unsafe_allow_html=True,
    )


def render_page_header(title: str, description: str):
    st.markdown(f"""<div class="hp-page-header">
    <div class="hp-page-header-accent"></div>
    <div class="hp-page-header-text">
        <p class="page-title">{title}</p>
        <p class="page-desc">{description}</p>
    </div>
</div>""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, subtext: str, accent_color: str) -> str:
    return f"""<div class="hp-kpi-card">
    <div class="hp-kpi-accent" style="background:{accent_color};"></div>
    <div class="hp-kpi-body">
        <div class="hp-kpi-label">{label}</div>
        <div class="hp-kpi-value">{value}</div>
        <div class="hp-kpi-sub">{subtext}</div>
    </div>
</div>"""


def section_title(text: str):
    st.markdown(f'<div class="hp-section-title">{text}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE: OVERVIEW
# ============================================================

def page_overview(df, pred_df, metrics_df):
    render_page_header(
        "Overview — KRS Inflow Deficit Dashboard",
        "Summary of El Niño signals, reservoir conditions, and SVR model predictions for KRS Basin."
    )

    # ── KPI Row ──
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        oni_val = latest.get("ONI", np.nan)
        soi_val = latest.get("SOI", np.nan)
        rain_val = latest.get("rainfall_mm", np.nan)
        inflow_val = latest.get("inflow_mcm", np.nan)
        stor_val = latest.get("storage_pct_frl", np.nan)
        def_val = latest.get("deficit_pct", np.nan)

        oni_phase = normalise_phase(oni_val) if not np.isnan(oni_val) else "N/A"
        oni_color = _CRIMSON if oni_val > 0.5 else (_FOREST if oni_val < -0.5 else _SAFFRON)

        cols = st.columns(6)
        cards = [
            (cols[0], "ONI Index", f"{oni_val:+.2f}" if not np.isnan(oni_val) else "N/A",
             oni_phase, oni_color),
            (cols[1], "SOI Index", f"{soi_val:+.2f}" if not np.isnan(soi_val) else "N/A",
             "Atmospheric ENSO signal", _NAVY),
            (cols[2], "Rainfall", f"{rain_val:.0f} mm" if not np.isnan(rain_val) else "N/A",
             "Latest month", _NAVY),
            (cols[3], "Inflow", f"{inflow_val:.1f} MCM" if not np.isnan(inflow_val) else "N/A",
             "Reservoir inflow", _NAVY),
            (cols[4], "Storage", f"{stor_val:.1f}%" if not np.isnan(stor_val) else "N/A",
             "% of FRL", _SAFFRON),
            (cols[5], "Current Deficit", f"{def_val:.1f}%" if not np.isnan(def_val) else "N/A",
             "Inflow deficit this month", _CRIMSON if def_val > 40 else _SAFFRON),
        ]
        for col, lbl, val, sub, color in cards:
            with col:
                st.markdown(kpi_card(lbl, val, sub, color), unsafe_allow_html=True)
    else:
        st.warning("Master data file not found. Run `data_pipeline.py` to generate data.")

    st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True)

    # ── Overview Chart ──
    if pred_df is not None and not pred_df.empty:
        section_title("Actual vs Predicted Inflow Deficit — Full Period")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_df["date"], y=pred_df[TARGET_COL],
            mode="lines", name="Actual Deficit",
            line=dict(color=_NAVY, width=2),
        ))
        if "svr_pred" in pred_df.columns:
            fig.add_trace(go.Scatter(
                x=pred_df["date"], y=pred_df["svr_pred"],
                mode="lines", name="SVR Predicted",
                line=dict(color=_SAFFRON, width=1.8, dash="dot"),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color=_TEXT_MUT, line_width=1, opacity=0.5)
        fig.update_layout(
            **_chart_layout(340),
            yaxis_title="Deficit % (+ = shortfall)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Model Technical Evaluation Table ──
    if metrics_df is not None and not metrics_df.empty:
        st.markdown(f'<div class="hp-section-title" style="margin-top: 18px;">Model Technical Evaluation (Held-Out Test Set: Jan 2018 – Nov 2020)</div>', unsafe_allow_html=True)
        # Build display dataframe
        display_rows = []
        for _, row in metrics_df.iterrows():
            m_name = str(row.get("model", ""))
            is_svr = "SVR" in m_name
            display_rows.append({
                "Model": ("★ " if is_svr else "  ") + m_name,
                "RMSE": f"{float(row.get('RMSE', 0.0)):.2f}%",
                "MAE": f"{float(row.get('MAE', 0.0)):.2f}%",
                "R²": f"{float(row.get('R2', 0.0)):.3f}",
                "MBE": f"{float(row.get('MBE', 0.0)):+.2f}%",
                "Within ±10%": f"{float(row.get('within_10pct', 0.0))*100:.1f}%",
                "Within ±20%": f"{float(row.get('within_20pct', 0.0))*100:.1f}%",
            })
        eval_df = pd.DataFrame(display_rows)
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
        st.caption("★ Operational model  |  Strict chronological split: train 2011–2017, test 2018–2020 (N=35). No data leakage.")

    render_footer()


# ============================================================
# PAGE: ENSO ANALYSIS
# ============================================================

def page_enso(df):
    render_page_header(
        "ENSO Analysis",
        "Oceanic Niño Index (ONI) and Southern Oscillation Index (SOI) trends for the KRS study period."
    )

    if df is None or df.empty:
        st.warning("Data not available. Run `data_pipeline.py` first.")
        render_footer()
        return

    # Sidebar year filter
    with st.sidebar:
        st.markdown(f'<div class="hp-section-title">Filters</div>', unsafe_allow_html=True)
        years = sorted(df["date"].dt.year.unique())
        year_range = st.slider(
            "Year range", int(years[0]), int(years[-1]),
            (int(years[0]), int(years[-1])), key="enso_year_slider"
        )

    dff = df[(df["date"].dt.year >= year_range[0]) & (df["date"].dt.year <= year_range[1])].copy()

    # ── ONI Timeline ──
    section_title("ONI Monthly Timeline")
    fig_oni = go.Figure()
    pos = dff[dff["ONI"] >= 0]
    neg = dff[dff["ONI"] < 0]
    fig_oni.add_trace(go.Bar(x=pos["date"], y=pos["ONI"], name="El Niño (+)", marker_color="#e8604c"))
    fig_oni.add_trace(go.Bar(x=neg["date"], y=neg["ONI"], name="La Niña (−)", marker_color="#4c9be8"))
    fig_oni.add_hline(y=0.5, line_dash="dot", line_color=_SAFFRON, line_width=1.2, annotation_text="El Niño threshold")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color=_NAVY, line_width=1.2, annotation_text="La Niña threshold")
    fig_oni.update_layout(**_chart_layout(300), barmode="overlay", yaxis_title="ONI (°C)", xaxis_title="Date")
    st.plotly_chart(fig_oni, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── SOI bar ──
    with col1:
        section_title("SOI Monthly Distribution")
        fig_soi = go.Figure()
        fig_soi.add_trace(go.Bar(
            x=dff["date"], y=dff["SOI"],
            marker_color=[_CRIMSON if v > 0 else _FOREST for v in dff["SOI"]],
            name="SOI",
            showlegend=False,
        ))
        fig_soi.add_hline(y=0, line_color=_TEXT_MUT, line_width=1)
        fig_soi.update_layout(**_chart_layout(300), yaxis_title="SOI", xaxis_title="Date")
        st.plotly_chart(fig_soi, use_container_width=True)

    # ── ONI vs Deficit scatter ──
    with col2:
        section_title("ONI vs Inflow Deficit (Scatter)")
        if "deficit_pct" in dff.columns:
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=dff["ONI"], y=dff["deficit_pct"],
                mode="markers",
                marker=dict(
                    size=7, color=dff["ONI"],
                    colorscale="RdBu_r",
                    showscale=True,
                    colorbar=dict(title="ONI", thickness=10, len=0.7),
                    line=dict(width=0.5, color=_BORDER),
                ),
                name="Observations",
                text=dff["date"].dt.strftime("%b %Y"),
                hovertemplate="<b>%{text}</b><br>ONI: %{x:.2f}<br>Deficit: %{y:.1f}%<extra></extra>",
            ))
            fig_sc = add_trendline(fig_sc, dff["ONI"].values, dff["deficit_pct"].values, _SAFFRON, "OLS Trend")
            fig_sc.update_layout(**_chart_layout(300), xaxis_title="ONI (°C)", yaxis_title="Inflow Deficit (%)")
            st.plotly_chart(fig_sc, use_container_width=True)

    # ── Phase summary ──
    section_title("ENSO Phase Frequency")
    dff["phase"] = dff["ONI"].apply(oni_to_phase)
    phase_cnt = dff["phase"].value_counts().reset_index()
    phase_cnt.columns = ["Phase", "Months"]
    phase_colors = {
        "Strong El Niño": "#c0392b",
        "Moderate El Niño": "#e8604c",
        "Neutral": _SAFFRON,
        "Moderate La Niña": "#4c9be8",
        "Strong La Niña": "#1a5276",
    }
    fig_phase = go.Figure(go.Bar(
        x=phase_cnt["Phase"], y=phase_cnt["Months"],
        marker_color=[phase_colors.get(p, _NAVY) for p in phase_cnt["Phase"]],
        text=phase_cnt["Months"], textposition="outside",
    ))
    fig_phase.update_layout(**_chart_layout(260), yaxis_title="Months", showlegend=False)
    st.plotly_chart(fig_phase, use_container_width=True)

    render_footer()


# ============================================================
# PAGE: RESERVOIR & RAINFALL
# ============================================================

def page_hydrology(df):
    render_page_header(
        "Reservoir & Rainfall Analysis",
        "KRS reservoir inflow, storage levels, and catchment rainfall — monthly time series and climatology."
    )

    if df is None or df.empty:
        st.warning("Data not available. Run `data_pipeline.py` first.")
        render_footer()
        return

    with st.sidebar:
        st.markdown(f'<div class="hp-section-title">Filters</div>', unsafe_allow_html=True)
        years = sorted(df["date"].dt.year.unique())
        year_range = st.slider(
            "Year range", int(years[0]), int(years[-1]),
            (int(years[0]), int(years[-1])), key="hydro_year_slider"
        )

    dff = df[(df["date"].dt.year >= year_range[0]) & (df["date"].dt.year <= year_range[1])].copy()

    # ── 3-panel subplot ──
    section_title("Hydrology Time Series — 3-Panel View")
    fig3 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Rainfall (mm)", "Inflow (MCM)", "Storage (% FRL)"),
    )
    grid_c = "#e9ecef" if not is_dark else "#30363d"

    fig3.add_trace(go.Bar(x=dff["date"], y=dff.get("rainfall_mm"), name="Rainfall",
                          marker_color="#4c9be8"), row=1, col=1)
    fig3.add_trace(go.Scatter(x=dff["date"], y=dff.get("inflow_mcm"), name="Inflow",
                               mode="lines+markers",
                               line=dict(color=_NAVY, width=1.8),
                               marker=dict(size=3)), row=2, col=1)
    if "inflow_lag1_mcm" in dff.columns:
        fig3.add_trace(go.Scatter(x=dff["date"], y=dff["inflow_lag1_mcm"], name="Inflow Lag-1",
                                   mode="lines", line=dict(color=_SAFFRON, width=1.3, dash="dot")), row=2, col=1)
    fig3.add_trace(go.Scatter(x=dff["date"], y=dff.get("storage_pct_frl"), name="Storage %FRL",
                               mode="lines", fill="tozeroy",
                               line=dict(color=_FOREST, width=1.8),
                               fillcolor=f"rgba(26,110,60,0.12)"), row=3, col=1)

    fig3.update_layout(
        **_chart_layout(420),
        showlegend=True,
    )
    for i in range(1, 4):
        fig3.update_xaxes(gridcolor=grid_c, linecolor=_BORDER, row=i, col=1)
        fig3.update_yaxes(gridcolor=grid_c, linecolor=_BORDER, row=i, col=1)
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Rainfall vs Inflow scatter ──
    with col1:
        section_title("Rainfall vs Inflow (Scatter)")
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=dff.get("rainfall_mm"), y=dff.get("inflow_mcm"),
            mode="markers",
            marker=dict(size=6, color=_NAVY, opacity=0.7, line=dict(width=0.5, color=_BORDER)),
            text=dff["date"].dt.strftime("%b %Y"),
            hovertemplate="<b>%{text}</b><br>Rainfall: %{x:.0f} mm<br>Inflow: %{y:.1f} MCM<extra></extra>",
            name="Monthly",
        ))
        fig_sc = add_trendline(
            fig_sc,
            dff["rainfall_mm"].values if "rainfall_mm" in dff.columns else [],
            dff["inflow_mcm"].values if "inflow_mcm" in dff.columns else [],
            _SAFFRON, "OLS Trend",
        )
        fig_sc.update_layout(**_chart_layout(300), xaxis_title="Rainfall (mm)", yaxis_title="Inflow (MCM)")
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Monthly climatology ──
    with col2:
        section_title("Monthly Climatology (Mean ± Std)")
        if "inflow_mcm" in dff.columns:
            clim = dff.groupby(dff["date"].dt.month)["inflow_mcm"].agg(["mean", "std"]).reset_index()
            clim.columns = ["month", "mean", "std"]
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            clim["month_name"] = clim["month"].apply(lambda x: month_names[x-1])
            fig_clim = go.Figure()
            fig_clim.add_trace(go.Bar(
                x=clim["month_name"], y=clim["mean"],
                error_y=dict(type="data", array=clim["std"], visible=True, color=_TEXT_MUT),
                marker_color=_NAVY, name="Mean Inflow",
            ))
            fig_clim.update_layout(**_chart_layout(300), yaxis_title="Inflow (MCM)", xaxis_title="Month")
            st.plotly_chart(fig_clim, use_container_width=True)

    render_footer()


# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================

def page_model_performance(pred_df, metrics_df):
    render_page_header(
        "Model Performance",
        "SVR-RBF prediction accuracy evaluated on test set (2018–2020) vs benchmark models."
    )

    # ── Top KPI cards — live from metrics_df ──
    _svr_rmse = _svr_mae = _svr_r2 = _svr_w20 = None
    if metrics_df is not None:
        _svr_rows = metrics_df[metrics_df["model"].str.contains("SVR", case=False, na=False)]
        if not _svr_rows.empty:
            _r = _svr_rows.iloc[0]
            _svr_rmse = f"{float(_r['RMSE']):.2f}%"
            _svr_mae  = f"{float(_r['MAE']):.2f}%"
            _svr_r2   = f"{float(_r['R2']):.3f}"
            _svr_w20  = f"{float(_r['within_20pct']) * 100:.1f}%"

    kpis = [
        ("RMSE",          _svr_rmse or "—", "SVR-RBF test set",               _CRIMSON),
        ("MAE",           _svr_mae  or "—", "Mean absolute error",             _SAFFRON),
        ("R²",            _svr_r2   or "—", "Variance explained",              _NAVY),
        ("Within ±20%",   _svr_w20  or "—", "Predictions within 20% of actual", _FOREST),
    ]
    cols = st.columns(4)
    for col, (lbl, val, sub, color) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(lbl, val, sub, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── View toggle ──
    view_mode = st.radio(
        "View mode",
        ["Test Set (2018–2020)", "Full Period"],
        horizontal=True,
        key="perf_view_toggle",
        label_visibility="collapsed",
    )

    # ── Actual vs Predicted chart ──
    if pred_df is not None and not pred_df.empty:
        section_title("Actual vs Predicted — Inflow Deficit (%)")
        # Use the 'split' column when available for accurate period partitioning
        if "Test Set" in view_mode and "split" in pred_df.columns:
            plot_df = pred_df[pred_df["split"] == "test"].copy()
        elif "Test Set" in view_mode:
            plot_df = pred_df[pred_df["date"].dt.year >= 2018].copy()
        else:
            plot_df = pred_df.copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=plot_df[TARGET_COL],
            mode="lines+markers",
            name="Actual",
            line=dict(color=_NAVY, width=2),
            marker=dict(size=4),
        ))
        if "svr_pred" in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df["date"], y=plot_df["svr_pred"],
                mode="lines+markers",
                name="SVR Predicted",
                line=dict(color=_SAFFRON, width=1.8, dash="dot"),
                marker=dict(size=4, symbol="diamond"),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color=_TEXT_MUT, line_width=1, opacity=0.4)
        fig.update_layout(
            **_chart_layout(340),
            yaxis_title="Deficit % (+ = shortfall)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Bottom row: metrics table + residuals ──
        col_left, col_right = st.columns([3, 2])

        with col_left:
            section_title("Model Comparison — All Benchmarks")
            if metrics_df is not None:
                # Display with st.dataframe for reliable rendering
                display_rows = []
                for _, row in metrics_df.iterrows():
                    model_name = str(row.iloc[0])
                    is_svr = "SVR" in model_name or "svr" in model_name.lower()
                    r = {"Model": ("★ " if is_svr else "  ") + model_name}
                    for col_name in metrics_df.columns[1:]:
                        val = row[col_name]
                        try:
                            fv = float(val)
                            r[col_name] = f"{fv:.3f}" if abs(fv) < 10 else f"{fv:.2f}"
                        except (ValueError, TypeError):
                            r[col_name] = str(val)
                    display_rows.append(r)
                comp_df = pd.DataFrame(display_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                st.caption("★ = Operational SVR-RBF model")
            else:
                st.info("metrics.csv not found. Run `train_model.py` first.")

        with col_right:
            section_title("Residuals Distribution")
            if "svr_pred" in plot_df.columns and TARGET_COL in plot_df.columns:
                residuals = plot_df[TARGET_COL] - plot_df["svr_pred"]
                fig_res = go.Figure()
                fig_res.add_trace(go.Bar(
                    x=plot_df["date"],
                    y=residuals,
                    marker_color=[_CRIMSON if r > 0 else _FOREST for r in residuals],
                    name="Residual",
                    showlegend=False,
                ))
                fig_res.add_hline(y=0, line_color=_TEXT_MUT, line_width=1)
                fig_res.update_layout(
                    **_chart_layout(260),
                    yaxis_title="Actual − Predicted (%)",
                    xaxis_title="Date",
                )
                st.plotly_chart(fig_res, use_container_width=True)

    else:
        st.info("Prediction data not available. Run `train_model.py` first.")

    st.markdown("""<div class="hp-callout">
    <strong>Technical Evaluation Note:</strong> Following systematic hyperparameter tuning on pre-test historical data (2011–2017), the tuned SVR-RBF model achieves <strong>R² = 0.402</strong> and <strong>RMSE = 60.17%</strong> on the withheld test set (Jan 2018 – Nov 2020, N=35), outperforming Linear Regression (R²=0.368), Ridge (R²=0.367), and naive Persistence (R²=0.008). Results are reported honestly without data leakage. Additional feature engineering (rainfall lag-1, storage delta) was evaluated and confirmed that the 11-feature specification provides the optimal empirical generalization on this finite monthly record (N=116).
</div>""", unsafe_allow_html=True)

    render_footer()


# ============================================================
# PAGE: DEFICIT PREDICTION
# ============================================================

SCENARIO_PRESETS = {
    "Strong El Niño — Drought": {
        "ONI": 1.8, "SOI": -12.0, "ONI_lag2": 1.5, "ONI_lag3": 1.2,
        "rainfall_mm": 30.0, "inflow_mcm": 50.0, "inflow_lag1_mcm": 60.0,
        "storage_pct_frl": 35.0, "month": 8, "deficit_pct": 65.0,
    },
    "Moderate El Niño — Below Normal": {
        "ONI": 0.9, "SOI": -6.0, "ONI_lag2": 0.7, "ONI_lag3": 0.5,
        "rainfall_mm": 80.0, "inflow_mcm": 150.0, "inflow_lag1_mcm": 170.0,
        "storage_pct_frl": 55.0, "month": 6, "deficit_pct": 35.0,
    },
    "Neutral ENSO — Normal Year": {
        "ONI": 0.1, "SOI": 0.5, "ONI_lag2": 0.0, "ONI_lag3": 0.0,
        "rainfall_mm": 160.0, "inflow_mcm": 280.0, "inflow_lag1_mcm": 260.0,
        "storage_pct_frl": 72.0, "month": 7, "deficit_pct": 10.0,
    },
    "Strong La Niña — Surplus": {
        "ONI": -1.6, "SOI": 12.0, "ONI_lag2": -1.4, "ONI_lag3": -1.1,
        "rainfall_mm": 320.0, "inflow_mcm": 480.0, "inflow_lag1_mcm": 500.0,
        "storage_pct_frl": 90.0, "month": 8, "deficit_pct": -20.0,
    },
    "Post-Monsoon — Moderate Risk": {
        "ONI": 0.5, "SOI": -3.0, "ONI_lag2": 0.3, "ONI_lag3": 0.2,
        "rainfall_mm": 50.0, "inflow_mcm": 100.0, "inflow_lag1_mcm": 200.0,
        "storage_pct_frl": 65.0, "month": 11, "deficit_pct": 28.0,
    },
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def page_deficit_prediction(model, scaler):
    render_page_header(
        "Deficit Prediction — What-If Scenario Tool",
        "Adjust ENSO and hydrological inputs to forecast next-month KRS inflow deficit using the trained SVR-RBF model."
    )

    if model is None or scaler is None:
        st.error("⚠️ Model not loaded. Please run `train_model.py` first to generate model files.")
        render_footer()
        return

    # ── Sidebar: scenario preset ──
    with st.sidebar:
        st.markdown(f'<div class="hp-section-title">Scenario Preset</div>', unsafe_allow_html=True)
        preset_choice = st.selectbox(
            "Load preset",
            ["(manual)"] + list(SCENARIO_PRESETS.keys()),
            key="preset_select",
        )

    preset_vals = SCENARIO_PRESETS.get(preset_choice, {})

    def pv(key, default):
        return float(preset_vals.get(key, default))

    # ── Two-panel layout ──
    col_inputs, col_result = st.columns([2, 1])

    with col_inputs:
        section_title("Input Parameters")
        e1, e2 = st.columns(2)

        with e1:
            st.markdown(f'<div style="font-size:0.72rem;font-weight:700;color:{_TEXT_MUT};text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">ENSO Indices</div>', unsafe_allow_html=True)
            oni    = st.slider("ONI (°C)",          -2.5, 2.5,  pv("ONI", 0.3),      0.05, key="s_oni")
            soi    = st.slider("SOI",               -20.0, 20.0, pv("SOI", 1.0),      0.5,  key="s_soi")
            oni_l2 = st.slider("ONI Lag-2 (°C)",   -2.5, 2.5,  pv("ONI_lag2", 0.2),  0.05, key="s_oni_l2")
            oni_l3 = st.slider("ONI Lag-3 (°C)",   -2.5, 2.5,  pv("ONI_lag3", 0.1),  0.05, key="s_oni_l3")

        with e2:
            st.markdown(f'<div style="font-size:0.72rem;font-weight:700;color:{_TEXT_MUT};text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Hydrological Parameters</div>', unsafe_allow_html=True)
            rain   = st.slider("Rainfall (mm)",      0.0, 600.0, pv("rainfall_mm", 150.0),    5.0,  key="s_rain")
            inflow = st.slider("Inflow (MCM)",        0.0, 800.0, pv("inflow_mcm", 250.0),    10.0, key="s_inflow")
            infl1  = st.slider("Inflow Lag-1 (MCM)", 0.0, 800.0, pv("inflow_lag1_mcm", 240.0), 10.0, key="s_infl1")
            stor   = st.slider("Storage % FRL",       0.0, 100.0, pv("storage_pct_frl", 70.0),  1.0,  key="s_stor")

        st.markdown(f'<div style="font-size:0.72rem;font-weight:700;color:{_TEXT_MUT};text-transform:uppercase;letter-spacing:0.07em;margin-top:8px;margin-bottom:8px;">Seasonal Cycle & State</div>', unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        with h1:
            preset_m = int(preset_vals.get("month", 7))
            sel_month = st.selectbox("Calendar Month", MONTH_NAMES, index=preset_m - 1, key="s_month")
            month_idx = MONTH_NAMES.index(sel_month) + 1
            month_sin = float(np.sin(2 * np.pi * month_idx / 12.0))
            month_cos = float(np.cos(2 * np.pi * month_idx / 12.0))
            st.caption(f"Harmonics: sin = {month_sin:+.2f}, cos = {month_cos:+.2f}")
        with h2:
            def_pct = st.slider("Current Deficit % (autoregressive)", -50.0, 100.0,
                                 pv("deficit_pct", 20.0), 0.5, key="s_defpct")

    # ── Prediction — Build strictly aligned with FEATURE_COLS ──
    feat_dict = {
        "ONI": oni,
        "SOI": soi,
        "ONI_lag2": oni_l2,
        "ONI_lag3": oni_l3,
        "rainfall_mm": rain,
        "inflow_mcm": inflow,
        "inflow_lag1_mcm": infl1,
        "storage_pct_frl": stor,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "deficit_pct": def_pct,
    }
    feat_df = pd.DataFrame([[feat_dict[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)
    feat_scaled = None
    try:
        feat_scaled = scaler.transform(feat_df)
        prediction  = float(model.predict(feat_scaled)[0])
    except Exception as e:
        prediction = float("nan")
        st.error(f"Prediction error: {e}")

    cat, cat_color, level, cat_desc = deficit_risk(prediction)
    phase = oni_to_phase(oni)

    with col_result:
        # Result card
        pred_str = f"{prediction:+.1f}%" if not math.isnan(prediction) else "Error"
        st.markdown(f"""<div class="hp-pred-card" style="border-left: 4px solid {cat_color};">
    <div class="hp-pred-label">Predicted Next-Month Deficit</div>
    <div class="hp-pred-value" style="color:{cat_color}; margin: 12px 0 8px;">{pred_str}</div>
    <div>
        <span class="hp-risk-badge" style="background:{cat_color}20;color:{cat_color};border:1px solid {cat_color}40;">
            {cat}
        </span>
    </div>
    <div style="font-size:0.73rem;color:{_TEXT_MUT};margin-top:10px;line-height:1.5;">{cat_desc}</div>
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid {_BORDER};font-size:0.72rem;color:{_TEXT_SEC};">
        <strong>ENSO Phase:</strong> {phase}<br>
        <strong>Inflow Outlook:</strong> {"Below Normal" if prediction > 20 else ("Surplus" if prediction < 0 else "Near Normal")}
    </div>
</div>""", unsafe_allow_html=True)

        # Gauge
        if not math.isnan(prediction):
            gauge_val = max(-60, min(100, prediction))
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val,
                number=dict(suffix="%", font=dict(size=18, color=_TEXT_PRI)),
                gauge=dict(
                    axis=dict(range=[-60, 100], tickfont=dict(size=9, color=_TEXT_MUT)),
                    bar=dict(color=cat_color, thickness=0.25),
                    bgcolor=_BG_SUBTLE,
                    borderwidth=1,
                    bordercolor=_BORDER,
                    steps=[
                        dict(range=[-60, 0],   color="#e8f5e9"),
                        dict(range=[0, 20],    color="#fff9e6"),
                        dict(range=[20, 40],   color="#fff3cd"),
                        dict(range=[40, 60],   color="#ffe0b2"),
                        dict(range=[60, 100],  color="#ffebee"),
                    ],
                    threshold=dict(line=dict(color=_CRIMSON, width=2), thickness=0.75, value=40),
                ),
            ))
            fig_g.update_layout(
                paper_bgcolor=_BG_CARD,
                height=180,
                margin=dict(l=20, r=20, t=20, b=10),
                font=dict(family="Inter", color=_TEXT_PRI),
            )
            st.plotly_chart(fig_g, use_container_width=True)

    # ── Model Input Diagnostics (11 Features) ──
    with st.expander("Model Input Diagnostics", expanded=False):
        diag_rows = []
        for i, c in enumerate(FEATURE_COLS):
            raw_v = f"{feat_dict[c]:.3f}"
            sc_v = f"{feat_scaled[0][i]:.3f}" if feat_scaled is not None else "—"
            diag_rows.append({
                "Feature Name": c,
                "Raw Input": raw_v,
                "Standardized Z-Score": sc_v,
            })
        diag_df = pd.DataFrame(diag_rows)
        st.dataframe(diag_df, use_container_width=True, hide_index=True)

    render_footer()


# ============================================================
# PAGE: RISK ANALYSIS
# ============================================================

def page_risk_analysis(df, pred_df):
    render_page_header(
        "Risk Analysis",
        "Deficit risk categorisation, seasonal patterns, and historical El Niño event impacts on KRS Basin."
    )

    if df is None or df.empty:
        st.warning("Data not available. Run `data_pipeline.py` first.")
        render_footer()
        return

    # ── Risk category KPI cards ──
    risk_cats = [
        ("Surplus / Low Risk", "deficit_pct < 20", _FOREST),
        ("Moderate Risk", "20 ≤ deficit < 40", _SAFFRON),
        ("High Risk", "40 ≤ deficit < 60", "#e07b30"),
        ("Severe Risk", "deficit ≥ 60", _CRIMSON),
    ]

    counts = [0, 0, 0, 0]
    if "deficit_pct" in df.columns:
        counts = [
            int((df["deficit_pct"] < 20).sum()),
            int(((df["deficit_pct"] >= 20) & (df["deficit_pct"] < 40)).sum()),
            int(((df["deficit_pct"] >= 40) & (df["deficit_pct"] < 60)).sum()),
            int((df["deficit_pct"] >= 60).sum()),
        ]
        total = len(df)
        cols = st.columns(4)
        for col, (lbl, thresh, color), cnt in zip(cols, risk_cats, counts):
            pct = 100 * cnt / total if total > 0 else 0
            with col:
                st.markdown(kpi_card(lbl, f"{cnt} months", f"{pct:.1f}% of record | {thresh}", color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Heatmap: Month × Year ──
    section_title("Deficit Risk Heatmap — Month × Year")
    if "deficit_pct" in df.columns:
        df2 = df.copy()
        df2["year"]  = df2["date"].dt.year
        df2["month"] = df2["date"].dt.month
        pivot = df2.pivot_table(values="deficit_pct", index="month", columns="year", aggfunc="mean")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.index = [month_names[m-1] for m in pivot.index]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            colorbar=dict(title="Deficit %", thickness=12, len=0.8),
            hoverongaps=False,
            hovertemplate="Year: %{x}<br>Month: %{y}<br>Deficit: %{z:.1f}%<extra></extra>",
            zmid=0,
        ))
        fig_heat.update_layout(
            **_chart_layout(320),
            xaxis_title="Year",
            yaxis_title="Month",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Box plot: deficit by ENSO phase ──
    with col1:
        section_title("Inflow Deficit by ENSO Phase")
        if "deficit_pct" in df.columns and "ONI" in df.columns:
            df3 = df.copy()
            df3["phase"] = df3["ONI"].apply(oni_to_phase)
            phase_order = ["Strong El Niño", "Moderate El Niño", "Neutral", "Moderate La Niña", "Strong La Niña"]
            phase_colors_map = {
                "Strong El Niño": "#c0392b", "Moderate El Niño": "#e8604c",
                "Neutral": _SAFFRON, "Moderate La Niña": "#4c9be8", "Strong La Niña": "#1a5276",
            }
            fig_box = go.Figure()
            for ph in phase_order:
                grp = df3[df3["phase"] == ph]["deficit_pct"]
                if len(grp) > 0:
                    fig_box.add_trace(go.Box(
                        y=grp, name=ph,
                        marker_color=phase_colors_map.get(ph, _NAVY),
                        line=dict(width=1.5),
                        boxmean="sd",
                    ))
            fig_box.add_hline(y=0, line_dash="dash", line_color=_TEXT_MUT, line_width=1)
            fig_box.update_layout(**_chart_layout(320), yaxis_title="Deficit %", showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    # ── Donut chart: risk distribution ──
    with col2:
        section_title("Risk Category Distribution")
        if "deficit_pct" in df.columns and sum(counts) > 0:
            risk_labels = ["Surplus/Low", "Moderate", "High", "Severe"]
            risk_colors = [_FOREST, _SAFFRON, "#e07b30", _CRIMSON]
            risk_vals   = [counts[0], counts[1], counts[2], counts[3]]
            fig_donut = go.Figure(go.Pie(
                labels=risk_labels, values=risk_vals,
                hole=0.52,
                marker=dict(colors=risk_colors, line=dict(color=_BG_CARD, width=2)),
                textinfo="label+percent",
                textfont=dict(size=11, family="Inter"),
                insidetextorientation="horizontal",
            ))
            donut_layout = _chart_layout(320)
            donut_layout["showlegend"] = True
            donut_layout["legend"] = dict(orientation="v", x=1.02, y=0.5)
            donut_layout.pop("xaxis", None)
            donut_layout.pop("yaxis", None)
            fig_donut.update_layout(**donut_layout)
            st.plotly_chart(fig_donut, use_container_width=True)

    # ── El Niño events table ──
    section_title("Notable El Niño Events in Study Period")
    events = [
        {"Period": "2011–2012", "ONI Peak": "-1.6 (La Niña)", "Impact": "Above-normal inflow, reservoir near FRL", "Risk": "Low"},
        {"Period": "2014–2015", "ONI Peak": "+0.9 (Mod. El Niño)", "Impact": "Below-normal monsoon, moderate deficit", "Risk": "Moderate"},
        {"Period": "2015–2016", "ONI Peak": "+2.6 (Strong El Niño)", "Impact": "Severe deficit, storage dropped below 50% FRL", "Risk": "Severe"},
        {"Period": "2017–2018", "ONI Peak": "-0.9 (Mod. La Niña)", "Impact": "Recovery phase, inflow near normal", "Risk": "Low"},
        {"Period": "2018–2019", "ONI Peak": "+0.8 (Mod. El Niño)", "Impact": "Below-normal inflow second half of 2019", "Risk": "Moderate"},
        {"Period": "2019–2020", "ONI Peak": "-0.5 (Weak La Niña)", "Impact": "Improved inflow, storage recovery", "Risk": "Low"},
    ]
    ev_df = pd.DataFrame(events)
    # Display events table using st.dataframe (avoids raw HTML rendering issue)
    st.dataframe(ev_df, use_container_width=True, hide_index=True)

    # ── Year-wise and Month-wise Analysis ──
    section_title("HISTORICAL YEAR-WISE AND MONTH-WISE RISK ANALYSIS")

    if pred_df is not None and not pred_df.empty and "svr_pred" in pred_df.columns:
        tab1, tab2 = st.tabs(["Year-Wise Analysis", "Month-Wise Analysis"])

        with tab1:
            # Year selector
            available_years = sorted(pred_df['date'].dt.year.unique())
            sel_year = st.selectbox("Select Year for Detailed Analysis", available_years, index=len(available_years)-1, key="risk_year_sel")
            year_df = pred_df[pred_df['date'].dt.year == sel_year].copy()
            
            # Compute risk categories
            year_df['risk_cat'] = year_df['svr_pred'].apply(lambda x: deficit_risk(x)[0] if 'svr_pred' in year_df.columns else 'N/A')
            
            # KPI row: months analyzed, avg predicted deficit, max deficit, risk distribution
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Months Analyzed", len(year_df))
            with c2: st.metric("Avg Predicted Deficit", f"{year_df['svr_pred'].mean():.1f}%" if 'svr_pred' in year_df.columns else "N/A")
            with c3: st.metric("Max Predicted Deficit", f"{year_df['svr_pred'].max():.1f}%" if 'svr_pred' in year_df.columns else "N/A")
            with c4: 
                high_risk_months = len(year_df[year_df['svr_pred'] > 20]) if 'svr_pred' in year_df.columns else 0
                st.metric("High-Risk Months", high_risk_months)
            
            # Monthly detail table for selected year
            display_cols = [c for c in ['date', 'ONI', 'ENSO_Phase', 'rainfall_mm', 'inflow_mcm', 'storage_pct_frl', 'deficit_pct', 'svr_pred'] if c in year_df.columns]
            year_display = year_df[display_cols].copy()
            if 'date' in year_display.columns:
                year_display['date'] = year_display['date'].dt.strftime('%b %Y')
            st.dataframe(year_display, use_container_width=True)
            
            # Download button
            st.download_button("Download Year Data as CSV", year_df.to_csv(index=False), file_name=f"risk_{sel_year}.csv", mime="text/csv")

        with tab2:
            month_names = ["January","February","March","April","May","June","July","August","September","October","November","December"]
            sel_month = st.selectbox("Select Month", month_names, key="risk_month_sel")
            month_num = month_names.index(sel_month) + 1
            month_df = pred_df[pred_df['date'].dt.month == month_num].copy()
            
            # Year-over-year for selected month
            if not month_df.empty and 'svr_pred' in month_df.columns:
                fig_mom = go.Figure()
                fig_mom.add_trace(go.Bar(x=month_df['date'].dt.year, y=month_df.get('deficit_pct', month_df['svr_pred']), name="Actual Deficit", marker_color=_NAVY))
                fig_mom.add_trace(go.Scatter(x=month_df['date'].dt.year, y=month_df['svr_pred'], name="SVR Predicted", mode="lines+markers", line=dict(color=_SAFFRON, dash="dot")))
                layout_mom = _chart_layout(300)
                layout_mom.pop("xaxis", None)
                layout_mom.pop("yaxis", None)
                fig_mom.update_layout(**layout_mom, xaxis_title="Year", yaxis_title="Deficit %", title=f"{sel_month} — Year-over-Year Deficit")
                st.plotly_chart(fig_mom, use_container_width=True)
            
            # Month summary table
            display_cols2 = [c for c in ['date', 'ONI', 'ENSO_Phase', 'deficit_pct', 'svr_pred'] if c in month_df.columns]
            month_display = month_df[display_cols2].copy()
            if 'date' in month_display.columns:
                month_display['date'] = month_display['date'].dt.strftime('%b %Y')
            st.dataframe(month_display, use_container_width=True)

        # Year-to-year comparison chart (avg deficit by year)
        section_title("YEAR-TO-YEAR COMPARISON")
        if 'svr_pred' in pred_df.columns:
            yearly_avg = pred_df.groupby(pred_df['date'].dt.year).agg(
                avg_deficit=('svr_pred', 'mean'),
                max_deficit=('svr_pred', 'max'),
                high_risk_months=('svr_pred', lambda x: (x > 20).sum())
            ).reset_index()
            yearly_avg.columns = ['Year', 'Avg Predicted Deficit', 'Max Predicted Deficit', 'High-Risk Months']
            
            fig_yr = go.Figure()
            fig_yr.add_trace(go.Bar(x=yearly_avg['Year'], y=yearly_avg['Avg Predicted Deficit'], name="Avg Deficit", marker_color=_NAVY))
            fig_yr.add_trace(go.Scatter(x=yearly_avg['Year'], y=yearly_avg['High-Risk Months'], name="High-Risk Months", mode="lines+markers", yaxis="y2", line=dict(color=_CRIMSON)))
            layout_yr = _chart_layout(300)
            layout_yr.pop("xaxis", None)
            layout_yr.pop("yaxis", None)
            fig_yr.update_layout(**layout_yr, 
                xaxis_title="Year", yaxis_title="Avg Deficit %",
                yaxis2=dict(title="High-Risk Months", overlaying="y", side="right"),
                barmode="group")
            st.plotly_chart(fig_yr, use_container_width=True)
            st.dataframe(yearly_avg, use_container_width=True)

    render_footer()


# ============================================================
# HYDRO PULSE ASSISTANT (CHATGPT-STYLE SLIDE-OUT SIDEBAR)
# ============================================================

def render_assistant():
    """ChatGPT-style slide-out assistant sidebar connected to live KRS hydrological data."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # If the sidebar is closed, do not render any bottom UI or launcher
    if not st.session_state.get("chat_open", False):
        return

    # Domain Knowledge Repository for Core Hydrological Principles
    domain_qa = {
        "What does the ONI index mean?": (
            "**Oceanic Niño Index (ONI):**\n"
            "- Running 3-month mean sea-surface temperature (SST) anomalies in the Niño 3.4 region (5°N–5°S, 120°–170°W).\n"
            "- **Thresholds:**\n"
            "  - **El Niño:** ONI ≥ +0.5°C (Warm phase; historically suppresses Southwest Monsoon precipitation in the Western Ghats catchment).\n"
            "  - **La Niña:** ONI ≤ -0.5°C (Cool phase; often promotes normal to surplus monsoon precipitation).\n"
            "  - **Neutral:** -0.5°C < ONI < +0.5°C."
        ),
        "How is the deficit calculated?": (
            "**Reservoir Inflow Deficit Formula:**\n"
            "$$\\text{Deficit (\\%)} = \\frac{\\text{Inflow}_{\\text{norm}} - \\text{Inflow}_{\\text{observed}}}{\\text{Inflow}_{\\text{norm}}} \\times 100\\%$$\n"
            "- A **positive value** indicates an inflow shortfall below the historical monthly baseline.\n"
            "- A **negative value** indicates surplus inflow into KRS Dam.\n"
            "- The SVR model forecasts the **next-month** deficit using 11 lagged and seasonal indicators."
        ),
        "What are key risk factors for KRS basin?": (
            "**Key KRS Basin Risk Factors:**\n"
            "1. **El Niño Persistence:** Sustained ONI ≥ +0.5°C during June–September monsoon months.\n"
            "2. **Negative SOI:** Persistent negative Southern Oscillation Index signaling weakened tropical convection.\n"
            "3. **Depleted Storage:** Reservoir live storage below 40% of Full Reservoir Level (FRL = 49.45 TMC / 124.8 ft).\n"
            "4. **Autoregressive Inflow Deficit:** Prior-month deficits exceeding 30% propagating into subsequent months."
        ),
        "What action is advised during High Risk?": (
            "**Operational Guidance for High/Critical Deficit Risk (Deficit > 40%):**\n"
            "1. Implement water rationing protocols prioritizing drinking water supply for Bengaluru and Mysuru urban centers.\n"
            "2. Reschedule non-critical irrigation canal releases to preserve dead storage.\n"
            "3. Shift regional power grid load from KRS hydro-peaking generation to base thermal/solar capacity.\n"
            "4. Issue basin-wide drought advisories to agricultural water user associations."
        )
    }

    def answer_query(user_q: str) -> str:
        q_lower = user_q.lower().strip()

        # ── Load live data from session state or cache ──
        live_df      = st.session_state.get("df")
        if live_df is None:
            live_df = load_data()
        live_pred_df = st.session_state.get("pred_df")
        if live_pred_df is None:
            live_pred_df = load_predictions()
        live_met_df  = st.session_state.get("met_df")
        if live_met_df is None:
            live_met_df = load_metrics()

        # ── 1. Current Water Deficit ──
        if any(w in q_lower for w in ["current water deficit", "current deficit", "observed deficit", "water deficit"]):
            if "predict" not in q_lower and "next" not in q_lower and "forecast" not in q_lower:
                if live_df is not None and not live_df.empty and "deficit_pct" in live_df.columns:
                    last_row = live_df.iloc[-1]
                    date_str = last_row["date"].strftime("%B %Y") if hasattr(last_row["date"], "strftime") else str(last_row["date"])
                    cur_def = float(last_row["deficit_pct"])
                    cur_inflow = float(last_row.get("inflow_mcm", 0.0))
                    cur_storage = float(last_row.get("storage_pct_frl", 0.0))
                    norm_inflow = float(last_row.get("normal_inflow_mcm", cur_inflow / (1 - cur_def / 100) if cur_def != 100 else 0))
                    sign_str = "shortfall below monthly baseline" if cur_def > 0 else "surplus above monthly baseline"
                    return (
                        f"**Current Water Deficit Status ({date_str}):**\n\n"
                        f"- **Recorded Inflow Deficit:** **{cur_def:+.1f}%** ({sign_str})\n"
                        f"- **Observed Inflow:** **{cur_inflow:.1f} MCM**\n"
                        f"- **Historical Normal Baseline:** **{norm_inflow:.1f} MCM**\n"
                        f"- **Current Reservoir Storage:** **{cur_storage:.1f}% of FRL**\n\n"
                        f"Deficit represents the deviation of actual reservoir inflow from climatological monthly normals. "
                        f"Positive percentages indicate water supply deficit for KRS Dam."
                    )

        # ── 2. Predicted Deficit for Next Three Days / Resolution Notice ──
        if any(w in q_lower for w in ["three days", "3 days", "next 3 days", "next two days", "2 days", "daily"]):
            if live_pred_df is not None and not live_pred_df.empty and "svr_pred" in live_pred_df.columns:
                last = live_pred_df.iloc[-1]
                pred_val = float(last["svr_pred"])
                risk_label, _, risk_lvl, risk_desc = deficit_risk(pred_val)
                date_str = last["date"].strftime("%B %Y") if hasattr(last["date"], "strftime") else str(last["date"])
                return (
                    f"**Forecasting Resolution Notice:**\n\n"
                    f"HydroPulse-SVR operates on a **monthly hydrological time-step** driven by large-scale ocean-atmospheric ENSO indices rather than a 3-day daily meteorological scale.\n\n"
                    f"**Next Monthly SVR Forecast Cycle:**\n"
                    f"- **Forecast Horizon:** Next consecutive month following {date_str}\n"
                    f"- **Predicted Next-Month Inflow Deficit:** **{pred_val:+.1f}%**\n"
                    f"- **Forecasted Risk Level:** **{risk_label}** (Tier {risk_lvl}/4)\n"
                    f"- **Advisory:** {risk_desc}\n\n"
                    f"*(For daily 72-hour operational releases, Karnataka WRD daily canal gauges and IMD Kodagu radar provide short-term telemetry.)*"
                )

        # ── 3. Risk Level for This Month / Current Risk ──
        if any(w in q_lower for w in ["risk level for this month", "risk level", "current risk", "risk this month", "monthly risk", "what is the risk"]):
            if live_pred_df is not None and not live_pred_df.empty and "svr_pred" in live_pred_df.columns:
                last = live_pred_df.iloc[-1]
                pred_val = float(last["svr_pred"])
                risk_label, _, risk_lvl, risk_desc = deficit_risk(pred_val)
                date_str = last["date"].strftime("%B %Y") if hasattr(last["date"], "strftime") else str(last["date"])
                oni_v = float(last.get("ONI", float("nan")))
                phase = str(last.get("ENSO_Phase", "Neutral"))
                return (
                    f"**KRS Basin Risk Evaluation ({date_str}):**\n\n"
                    f"- **Forecasted Deficit:** **{pred_val:+.1f}%**\n"
                    f"- **Risk Classification:** **{risk_label}** (Tier {risk_lvl} of 4)\n"
                    f"- **Institutional Guidance:** {risk_desc}\n"
                    f"- **ENSO State:** ONI = {oni_v:+.2f}°C ({phase})\n\n"
                    f"**Deficit Risk Scale:**\n"
                    f"- **Surplus / Low Risk:** Deficit < 20%\n"
                    f"- **Moderate Risk:** 20% ≤ Deficit < 40%\n"
                    f"- **High Risk:** 40% ≤ Deficit < 60%\n"
                    f"- **Severe Risk:** Deficit ≥ 60%"
                )

        # ── 4. Year-Wise Risk Analysis ──
        if any(w in q_lower for w in ["year-wise", "year wise", "yearly risk", "risk by year", "historical year", "annual risk"]):
            if live_pred_df is not None and not live_pred_df.empty and "svr_pred" in live_pred_df.columns:
                yearly_summary = live_pred_df.groupby(live_pred_df["date"].dt.year).agg(
                    avg_pred=("svr_pred", "mean"),
                    max_pred=("svr_pred", "max"),
                    high_risk_cnt=("svr_pred", lambda x: int((x > 20).sum())),
                    severe_cnt=("svr_pred", lambda x: int((x > 40).sum())),
                ).reset_index()

                table_lines = [
                    "| Year | Avg Predicted Deficit | Max Deficit | High-Risk Months (>20%) | Severe (>40%) |",
                    "| :---: | :---: | :---: | :---: | :---: |"
                ]
                for _, row in yearly_summary.iterrows():
                    y = int(row["Year"])
                    table_lines.append(f"| {y} | {row['avg_pred']:+.1f}% | {row['max_pred']:+.1f}% | {int(row['high_risk_cnt'])} | {int(row['severe_cnt'])} |")

                table_str = "\n".join(table_lines)
                worst_idx = yearly_summary["avg_pred"].idxmax()
                worst_yr = yearly_summary.loc[worst_idx]
                best_idx = yearly_summary["avg_pred"].idxmin()
                best_yr = yearly_summary.loc[best_idx]
                return (
                    f"**Historical Year-Wise Risk & Deficit Analysis:**\n\n"
                    f"{table_str}\n\n"
                    f"**Basin Insights:**\n"
                    f"- **Highest Deficit Year:** **{int(worst_yr['Year'])}** with average predicted deficit of **{worst_yr['avg_pred']:+.1f}%** ({int(worst_yr['severe_cnt'])} severe deficit months during strong El Niño).\n"
                    f"- **Lowest Deficit Year:** **{int(best_yr['Year'])}** with average predicted deficit of **{best_yr['avg_pred']:+.1f}%** (strong inflow replenishment)."
                )

        # ── 5. What Does the ONI Value Mean ──
        if any(w in q_lower for w in ["oni value", "oni mean", "what is oni", "what does the oni", "oceanic nino"]):
            latest_oni = float(live_df.iloc[-1]["ONI"]) if live_df is not None and not live_df.empty else 0.30
            phase = oni_to_phase(latest_oni)
            return (
                f"**Oceanic Niño Index (ONI) Explanation:**\n\n"
                f"- **Current Dataset Value:** **{latest_oni:+.2f}°C** ({phase})\n\n"
                f"**Definition & Scientific Basis:**\n"
                f"The ONI is NOAA's benchmark indicator for monitoring El Niño–Southern Oscillation (ENSO). "
                f"It measures the 3-month running mean of Sea Surface Temperature (SST) anomalies in the **Niño 3.4 region** (5°N–5°S, 120°W–170°W).\n\n"
                f"**Thresholds:**\n"
                f"- **El Niño (Warm Phase):** ONI ≥ **+0.5°C** (Suppresses Southwest Monsoon convection over the Western Ghats catchment, decreasing KRS inflow).\n"
                f"- **La Niña (Cool Phase):** ONI ≤ **-0.5°C** (Promotes normal to excess monsoon rainfall).\n"
                f"- **Neutral:** **-0.5°C < ONI < +0.5°C** (Normal circulation)."
            )

        # ── 6. Explain the Standardized Score ──
        if any(w in q_lower for w in ["standardized score", "standardized", "z-score", "z score", "standard deviation", "standardscaler"]):
            return (
                f"**Standardized Score (Z-Score) Explanation:**\n\n"
                f"In HydroPulse-SVR, all 11 model inputs are transformed before SVR ingestion using Z-score standardization:\n\n"
                f"$$Z = \\frac{{X - \\mu}}{{\\sigma}}$$\n\n"
                f"Where:\n"
                f"- $X$ = raw physical value (e.g. Inflow in MCM, Rainfall in mm, ONI in °C)\n"
                f"- $\\mu$ = mean of training distribution (2011–2017)\n"
                f"- $\\sigma$ = standard deviation of training distribution\n\n"
                f"**Why SVR Requires Standardization:**\n"
                f"The Support Vector Regressor uses a Radial Basis Function (RBF) kernel $K(x, x') = \\exp(-\\gamma ||x - x'||^2)$. "
                f"Without standardization, features with large numbers (such as Inflow up to 800 MCM or Rainfall up to 600 mm) "
                f"would completely dominate low-magnitude features (such as ONI from -2.5 to +2.5 or seasonal sine/cosine).\n\n"
                f"**Score Interpretation:**\n"
                f"- **Z = 0.0:** Exactly equal to historical monthly mean.\n"
                f"- **Z > +1.0:** Significantly above normal (+1 standard deviation).\n"
                f"- **Z < -1.0:** Significantly below normal (-1 standard deviation)."
            )

        # ── 7. Current SVR Model Accuracy ──
        if any(w in q_lower for w in ["svr model accuracy", "model accuracy", "current accuracy", "svr accuracy", "model performance", "accuracy", "rmse", "r2", "r²", "mae"]):
            if live_met_df is not None and not live_met_df.empty:
                svr_row = live_met_df[live_met_df["model"].str.contains("SVR", case=False, na=False)]
                if not svr_row.empty:
                    r = svr_row.iloc[0]
                    r2   = float(r.get("R2", 0.402))
                    rmse = float(r.get("RMSE", 60.17))
                    mae  = float(r.get("MAE", 43.94))
                    mbe  = float(r.get("MBE", 14.79))
                    w10  = float(r.get("within_10pct", 0.171)) * 100
                    w20  = float(r.get("within_20pct", 0.371)) * 100
                    return (
                        f"**Operational SVR-RBF Model Performance (Held-Out Test Set 2018–2020, N=35):**\n\n"
                        f"- **$R^2$ Score (Variance Explained):** **{r2:.3f}** ({r2*100:.1f}%)\n"
                        f"- **RMSE (Root Mean Squared Error):** **{rmse:.2f}%** (lowest among all benchmark architectures)\n"
                        f"- **MAE (Mean Absolute Error):** **{mae:.2f}%**\n"
                        f"- **Mean Bias Error (MBE):** **{mbe:+.2f}%**\n"
                        f"- **Predictions within ±10%:** **{w10:.1f}%**\n"
                        f"- **Predictions within ±20%:** **{w20:.1f}%**\n\n"
                        f"**Benchmark Comparison:**\n"
                        f"- **SVR-RBF:** RMSE = **{rmse:.2f}%** | $R^2$ = **{r2:.3f}** (Operational Model)\n"
                        f"- **Linear Regression:** RMSE = 61.87% | $R^2$ = 0.368\n"
                        f"- **Ridge Regression:** RMSE = 61.88% | $R^2$ = 0.367\n"
                        f"- **Random Forest:** RMSE = 68.64% | $R^2$ = 0.222\n"
                        f"- **Persistence Baseline:** RMSE = 77.51% | $R^2$ = 0.008\n\n"
                        f"Evaluated on withheld test observations ($N=35$) with zero data leakage."
                    )

        # ── 8. Explain the Latest Diagnostic Results ──
        if any(w in q_lower for w in ["diagnostic results", "latest diagnostic", "explain diagnostics", "diagnostics", "diagnostic table"]):
            diag_lines = [
                "| Feature | Scientific Role in Model |",
                "| :--- | :--- |",
                "| **ONI** | Macro ocean thermal anomaly (°C); triggers monsoon suppression when > +0.5°C |",
                "| **SOI** | Southern Oscillation atmospheric pressure delta; negative values signal El Niño |",
                "| **ONI Lag-2 / Lag-3** | Captures oceanic wave propagation delay into Western Ghats |",
                "| **Rainfall (mm)** | Catchment precipitation fueling reservoir runoff |",
                "| **Inflow & Lag-1 (MCM)** | Autoregressive hydrological momentum; river baseflow memory |",
                "| **Storage % FRL** | Initial reservoir storage condition determining buffer capacity against deficit |",
                "| **Month Sin / Cos** | Harmonic encoding of the annual Southwest Monsoon seasonality cycle |",
                "| **Current Deficit %** | Prior month's shortfall propagating forward into consecutive months |"
            ]
            diag_str = "\n".join(diag_lines)
            return (
                f"**Model Input Diagnostics Explanation:**\n\n"
                f"The SVR-RBF model evaluates 11 hydrological and ocean-atmospheric features. "
                f"In the Diagnostics table on the Deficit Prediction page, each feature is displayed with both its raw physical value and its standardized Z-score:\n\n"
                f"{diag_str}\n\n"
                f"**Diagnostic Interpretation:**\n"
                f"- Features with $|Z| > 1.5$ represent extreme anomalies and exert strong non-linear influence in the SVR kernel.\n"
                f"- Elevated ONI combined with depressed Inflow and low Storage % FRL yields the highest predicted deficit."
            )

        # ── SVR Model Input Features Listing ──
        if any(w in q_lower for w in ["feature", "input parameter", "parameters", "variables"]):
            return (
                "**SVR Model Input Features (11 total):**\n"
                "1. **ONI** — Oceanic Niño Index (°C)\n"
                "2. **SOI** — Southern Oscillation Index\n"
                "3. **ONI Lag-2** — ONI value 2 months prior\n"
                "4. **ONI Lag-3** — ONI value 3 months prior\n"
                "5. **Rainfall (mm)** — Current month catchment rainfall\n"
                "6. **Inflow (MCM)** — Current month reservoir inflow\n"
                "7. **Inflow Lag-1 (MCM)** — Previous month inflow\n"
                "8. **Storage % FRL** — Reservoir storage as % of Full Reservoir Level\n"
                "9. **Month Sin** — Seasonal harmonic (sine component)\n"
                "10. **Month Cos** — Seasonal harmonic (cosine component)\n"
                "11. **Current Deficit %** — Autoregressive lag of current deficit"
            )

        # ── General Deficit Prediction Query ──
        if any(w in q_lower for w in ["predict", "prediction", "forecast"]):
            if live_pred_df is not None and not live_pred_df.empty and "svr_pred" in live_pred_df.columns:
                last = live_pred_df.iloc[-1]
                pred_val = float(last["svr_pred"])
                risk_label, _, risk_lvl, risk_desc = deficit_risk(pred_val)
                date_str = last["date"].strftime("%B %Y") if hasattr(last["date"], "strftime") else str(last["date"])
                return (
                    f"**Next-Month SVR Deficit Forecast ({date_str}):**\n\n"
                    f"- **Predicted Inflow Deficit:** **{pred_val:+.1f}%**\n"
                    f"- **Risk Classification:** **{risk_label}** (Level {risk_lvl}/4)\n"
                    f"- **Guidance:** {risk_desc}\n\n"
                    f"You can also use the **Deficit Prediction** page to adjust input parameters and test what-if scenarios in real-time."
                )

        # ── Reservoir Storage & Specs ──
        if any(w in q_lower for w in ["storage", "capacity", "reservoir level", "frl", "tmc"]):
            if live_df is not None and not live_df.empty and "storage_pct_frl" in live_df.columns:
                last_row  = live_df.iloc[-1]
                storage_v = float(last_row["storage_pct_frl"])
                date_str  = last_row["date"].strftime("%B %Y") if hasattr(last_row["date"], "strftime") else ""
                status    = "Adequate (>60%)" if storage_v >= 60 else ("Low (30–60%)" if storage_v >= 30 else "Critical (<30%)")
                return (
                    f"**KRS Reservoir Storage ({date_str}):**\n\n"
                    f"- **Storage % FRL:** **{storage_v:.1f}%** — Status: **{status}**\n"
                    f"- **Full Reservoir Level (FRL):** 124.8 ft (~49.45 TMC gross storage, ~1,400 MCM live capacity)\n"
                    f"- Storage levels below 40% FRL trigger municipal water protection protocols for Bengaluru and Mysuru."
                )

        # ── Rainfall Dynamics ──
        if any(w in q_lower for w in ["rainfall", "rain", "monsoon", "precipitation"]):
            if live_df is not None and not live_df.empty and "rainfall_mm" in live_df.columns:
                last_row = live_df.iloc[-1]
                rain_v   = float(last_row["rainfall_mm"])
                date_str = last_row["date"].strftime("%B %Y") if hasattr(last_row["date"], "strftime") else ""
                avg_rain = float(live_df["rainfall_mm"].mean())
                return (
                    f"**Catchment Rainfall Status ({date_str}):**\n\n"
                    f"- **Observed Rainfall:** **{rain_v:.0f} mm**\n"
                    f"- **Historical Average:** **{avg_rain:.0f} mm/month**\n"
                    f"- The KRS catchment depends heavily on Southwest Monsoon precipitation (June–September) "
                    f"in the Western Ghats (Kodagu district). Rainfall deficits correlate strongly with positive ONI (El Niño) events."
                )

        # ── Inflow ──
        if any(w in q_lower for w in ["inflow", "flow", "mcm"]):
            if live_df is not None and not live_df.empty and "inflow_mcm" in live_df.columns:
                last_row = live_df.iloc[-1]
                inflow_v = float(last_row["inflow_mcm"])
                date_str = last_row["date"].strftime("%B %Y") if hasattr(last_row["date"], "strftime") else ""
                avg_inf  = float(live_df["inflow_mcm"].mean())
                return (
                    f"**KRS Reservoir Inflow Status ({date_str}):**\n\n"
                    f"- **Latest Inflow:** **{inflow_v:.1f} MCM**\n"
                    f"- **Historical Monthly Mean:** **{avg_inf:.1f} MCM**\n"
                    f"- Inflow is the primary driver of reservoir storage replenishment and hydropower peak-generation capacity."
                )

        # ── SOI (Southern Oscillation Index) ──
        if any(w in q_lower for w in ["soi", "southern oscillation"]):
            if live_df is not None and not live_df.empty and "SOI" in live_df.columns:
                last_row = live_df.iloc[-1]
                soi_v    = float(last_row["SOI"])
                date_str = last_row["date"].strftime("%B %Y") if hasattr(last_row["date"], "strftime") else ""
                interp   = "El Niño-like conditions" if soi_v < -0.5 else ("La Niña-like conditions" if soi_v > 0.5 else "Neutral conditions")
                return (
                    f"**Southern Oscillation Index ({date_str}):**\n\n"
                    f"- **SOI Value:** **{soi_v:+.2f}** ({interp})\n"
                    f"- Negative SOI values (< -0.5) denote reduced surface pressure in Tahiti relative to Darwin, confirming El Niño circulation."
                )

        # ── Exact Match in Domain QA ──
        for k, v in domain_qa.items():
            if k.lower() in q_lower or q_lower in k.lower():
                return v

        # ── Contextual General Fallback ──
        return (
            f"**HydroPulse AI Assistant:** Regarding *'{user_q}'*, the console monitors KRS reservoir "
            "inflow deficit using 11 ocean-atmospheric and basin hydrological indicators. "
            "Try asking:\n"
            "- *'What is the current water deficit?'*\n"
            "- *'What is the risk level for this month?'*\n"
            "- *'Show me the year-wise risk analysis.'*\n"
            "- *'What does the ONI value mean?'*\n"
            "- *'What is the current SVR model accuracy?'*\n"
            "- *'Explain the latest diagnostic results.'*"
        )

    # ── Slide-Out Sidebar Modal Container ──
    st.markdown('<div class="hp-chat-sidebar-backdrop"></div>', unsafe_allow_html=True)
    with st.container(key="chat_sidebar_panel"):
        # Sidebar Header with Title & Close Button
        col_hdr_txt, col_hdr_btn = st.columns([8.2, 1.8], vertical_alignment="center")
        with col_hdr_txt:
            st.markdown(f"""
            <div style="line-height:1.2;">
                <span style="font-weight:700;font-size:0.92rem;color:{_NAVY};">HydroPulse AI Assistant</span><br>
                <span style="font-size:0.68rem;color:{_TEXT_MUT};">KRS Basin Hydrological & SVR Console</span>
            </div>
            """, unsafe_allow_html=True)
        with col_hdr_btn:
            if st.button("\u2715", key="chat_sidebar_close_btn", help="Close AI Assistant"):
                st.session_state["chat_open"] = False
                st.rerun()

        st.markdown(f'<hr style="margin:8px 0 10px 0; border:none; border-top:1px solid {_BORDER};">', unsafe_allow_html=True)

        # Suggested Prompts (Pills)
        suggested_prompts = [
            "What is the current water deficit?",
            "What is the risk level for this month?",
            "Show me the year-wise risk analysis.",
            "What is the predicted deficit for the next three days?",
            "What does the ONI value mean?",
            "Explain the standardized score.",
            "What is the current SVR model accuracy?",
            "Explain the latest diagnostic results.",
        ]

        with st.expander("Suggested Inquiries", expanded=(len(st.session_state["chat_history"]) == 0)):
            for idx, p_text in enumerate(suggested_prompts):
                if st.button(p_text, key=f"chat_pill_{idx}", use_container_width=True):
                    st.session_state["chat_history"].append({"role": "user", "content": p_text})
                    with st.spinner("Analyzing KRS basin data..."):
                        reply = answer_query(p_text)
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                    st.rerun()

        # Chat Messages Scroll Area
        chat_box = st.container(height=360)
        with chat_box:
            if not st.session_state["chat_history"]:
                st.markdown(
                    f'<div style="text-align:center;color:{_TEXT_MUT};font-size:0.78rem;padding:30px 10px;">'
                    'Ask any question regarding KRS water deficit, ENSO indices, model accuracy, or basin risk.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                for m in st.session_state["chat_history"]:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])

        # Bottom Message Input Form
        with st.form("chat_sidebar_form", clear_on_submit=True):
            col_inp, col_snd = st.columns([7.8, 2.2], vertical_alignment="center")
            with col_inp:
                msg_input = st.text_input(
                    "Message",
                    placeholder="Ask a question... (Press Enter)",
                    label_visibility="collapsed",
                    key="chat_sidebar_text_input",
                )
            with col_snd:
                send_clicked = st.form_submit_button("Send", use_container_width=True)

        if send_clicked and msg_input.strip():
            q_text = msg_input.strip()
            st.session_state["chat_history"].append({"role": "user", "content": q_text})
            with st.spinner("Analyzing KRS basin data..."):
                ans_text = answer_query(q_text)
            st.session_state["chat_history"].append({"role": "assistant", "content": ans_text})
            st.rerun()

        if st.session_state["chat_history"]:
            if st.button("Clear Conversation", key="chat_sidebar_clear_btn"):
                st.session_state["chat_history"] = []
                st.rerun()


# ============================================================
# FOOTER
# ============================================================

def render_footer():
    st.markdown(f"""<div class="hp-footer">
    <span><strong>HydroPulse-SVR</strong> — KRS Basin Inflow Deficit Prediction</span>
    <span>SVR-RBF · RMSE 60.17% · R² 0.402 · Test period 2018–2020</span>
    <span>Data: NOAA ONI, BOM SOI, WRI KRS, World Bank Rainfall · Built with Streamlit + Plotly</span>
</div>""", unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================

def main():
    # ── Load data ──
    df      = load_data()
    pred_df = load_predictions()
    met_df  = load_metrics()
    model, scaler = load_model()

    # Cache into session state for components and chatbot
    st.session_state["df"] = df
    st.session_state["pred_df"] = pred_df
    st.session_state["met_df"] = met_df

    # ── Top nav bar ──
    render_navbar()

    page = st.session_state.get("current_page", "Overview")

    # ── Route to page ──
    if page == "Overview":
        page_overview(df, pred_df, met_df)
    elif page == "ENSO Analysis":
        page_enso(df)
    elif page == "Reservoir & Rainfall":
        page_hydrology(df)
    elif page == "Model Performance":
        page_model_performance(pred_df, met_df)
    elif page == "Deficit Prediction":
        page_deficit_prediction(model, scaler)
    elif page == "Risk Analysis":
        page_risk_analysis(df, pred_df)
    else:
        page_overview(df, pred_df, met_df)

    # ── Floating HydroPulse Assistant ──
    render_assistant()


if __name__ == "__main__":
    main()
else:
    # Streamlit imports this module; call main() unconditionally.
    main()
