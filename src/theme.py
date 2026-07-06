"""
Global visual theme for the Streamlit app: dark, blue-accented,
applied once from app.py so every page inherits it consistently.
"""

import streamlit as st

from src.config import (
    BG_BASE, BG_SURFACE, BG_SURFACE_ALT, BORDER,
    TEXT_PRIMARY, TEXT_MUTED, PRIMARY, PRIMARY_BRIGHT,
    SUCCESS, WARNING, DANGER,
)

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

/* ---- App background: near-black with a faint blue glow ---- */
.stApp {{
    background: {BG_BASE};
    color: {TEXT_PRIMARY};
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: {BG_SURFACE};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {{
    color: {TEXT_MUTED};
}}

/* ---- Headers: display font + subtle gradient underline ---- */
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
}}
h1 {{
    padding-bottom: 0.4rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, {PRIMARY}, transparent 70%);
    border-image-slice: 1;
}}

/* ---- Metric cards (used across pages via .metric-card) ---- */
.metric-card {{
    background: {BG_SURFACE};
    padding: 1rem;
    border-radius: 0.6rem;
    border: 1px solid {BORDER};
    border-left: 3px solid {PRIMARY};
}}

/* ---- Native st.metric widgets ---- */
div[data-testid="stMetric"] {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {PRIMARY};
    border-radius: 0.6rem;
    padding: 0.9rem 1rem;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED};
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY};
    font-family: 'Space Grotesk', sans-serif;
}}

/* ---- Buttons ---- */
.stButton > button {{
    background: {PRIMARY};
    color: #FFFFFF;
    border: none;
    border-radius: 0.5rem;
    font-weight: 600;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}}
.stButton > button:hover {{
    border: 1px solid {PRIMARY_BRIGHT};
    transform: translateY(-1px);
    color: #FFFFFF;
}}

/* ---- Tabs ---- */
button[data-baseweb="tab"] {{
    color: {TEXT_MUTED};
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {PRIMARY_BRIGHT};
    font-weight: 600;
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY};
}}

/* ---- Inputs / selects / sliders ---- */
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{
    background: {BG_SURFACE_ALT};
    border-color: {BORDER};
    color: {TEXT_PRIMARY};
}}
div[data-testid="stSlider"] > div > div > div > div {{
    background-color: {PRIMARY};
}}

/* ---- Dataframes / tables ---- */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 0.5rem;
    overflow: hidden;
}}

/* ---- Alert boxes: keep semantic colors, darker fills ---- */
div[data-testid="stAlert"] {{
    border-radius: 0.5rem;
    border: 1px solid {BORDER};
}}

/* ---- Dividers ---- */
hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, {BORDER}, transparent);
}}

/* ---- Expander ---- */
details {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 0.5rem;
}}
</style>
"""


def apply_theme():
    """Inject the global CSS theme. Call once per script run from app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
