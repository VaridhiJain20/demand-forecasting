"""
Central configuration file for the Retail Demand Forecasting
and Inventory Optimization System.

All constants used across the app live here so nothing is
hardcoded elsewhere.
"""

# ---------------------------------------------------------------
# REAL DATA SETTINGS (Kaggle "Store Item Demand Forecasting Challenge")
# ---------------------------------------------------------------
RAW_DATA_PATH = "data/raw/train.csv"
PROCESSED_DIR = "data/processed"

# ---------------------------------------------------------------
# FORECASTING SETTINGS
# ---------------------------------------------------------------
FORECAST_HORIZONS = [30, 60, 90]

# ---------------------------------------------------------------
# INVENTORY SETTINGS
# ---------------------------------------------------------------
SERVICE_LEVELS = [0.90, 0.95, 0.99]
DEFAULT_LEAD_TIME = 7

# Z-scores for common service levels (used by inventory_optimizer)
Z_SCORES = {
    0.90: 1.28,
    0.95: 1.645,
    0.99: 2.33,
}

# ---------------------------------------------------------------
# COLOR PALETTE — dark, blue-accented
# ---------------------------------------------------------------
BG_BASE = "#0A0E16"        # app background, near-black with a blue tint
BG_SURFACE = "#121A28"     # cards, panels
BG_SURFACE_ALT = "#182338" # nested/alt panels, table headers
BORDER = "#253148"         # hairline borders
TEXT_PRIMARY = "#E8EDF6"
TEXT_MUTED = "#8A96AC"

PRIMARY = "#3B82F6"        # core blue accent
PRIMARY_BRIGHT = "#63A4FF" # hover/glow variant
SUCCESS = "#22C55E"
WARNING = "#F5A524"
DANGER = "#F0465A"
INFO = "#8B7CF6"

# Extra shades used for charts / severity levels
COLOR_HIGH = DANGER
COLOR_MEDIUM = WARNING
COLOR_LOW = "#EAB308"
COLOR_NORMAL = PRIMARY

# ---------------------------------------------------------------
# PLOTLY DEFAULTS
# ---------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT_DEFAULTS = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=60, b=40),
    font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
    colorway=[PRIMARY, INFO, SUCCESS, WARNING, DANGER, "#38BDF8", "#F472B6"],
)
