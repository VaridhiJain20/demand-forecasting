#  Retail Demand Forecasting and Inventory Optimization System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Prophet](https://img.shields.io/badge/Prophet-1.1.5-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Problem Statement

Retailers routinely struggle to balance two conflicting goals: avoiding
stockouts that lose sales, and avoiding overstock that ties up capital
and increases holding costs. Doing this well requires accurate demand
forecasts and sound inventory policy — safety stock, reorder points, and
order quantities that reflect real demand variability.

This project builds an end-to-end system to **explore historical
sales data, forecast future demand with multiple models, detect
unusual sales patterns, and calculate data-driven inventory
policies** — all in a single interactive Streamlit application. It
works with the Kaggle "Store Item Demand Forecasting Challenge"
dataset, or with your own retail sales data in virtually any format.

## Data Source

Upload a CSV file directly in the sidebar, or place one at
`data/raw/train.csv` and reload — no code changes needed either way.

**Supported file format:** CSV only.

**Flexible column names:** your file doesn't need columns named
exactly `date`, `store`, `item`, `sales`. Common alternates are
auto-detected — e.g. `order_date`, `store_id`, `sku`, `units_sold`
all map correctly on their own. If a column genuinely can't be
guessed, the app shows a one-time mapping form asking you which
column is which, with a preview of your data to help you decide.

The sidebar's "Data Source" panel shows the active source, row/store/item
counts, date range, and a short data-quality summary (missing values,
duplicates, negative sales) for whatever's currently loaded.

The store and item dropdowns are populated dynamically from whichever
dataset is active — this works with any number of stores/items and
with either numeric or text-based IDs (e.g. `"Store_A"`).

**Note on `.pkl`/`.pickle` files:** these are intentionally not
accepted, since loading a pickle file can execute arbitrary code from
an untrusted source.

## Features

- 🔍 **EDA Explorer** — interactive filters, rolling averages (window
  adapts to the selected aggregation level), store and item
  comparisons, distribution analysis, and heatmaps.
- 📈 **Forecasting** — Moving Average, XGBoost (recursive multi-step),
  and Prophet models with confidence intervals and automatic model
  comparison.
- 🚨 **Anomaly Detection** — Isolation Forest based detection with an
  adjustable contamination rate, severity scoring (HIGH / MEDIUM /
  LOW), and pattern insights.
- 📦 **Inventory Optimizer** — safety stock, reorder point, and
  Economic Order Quantity (EOQ) calculations with visual inventory
  health gauges and cost analysis.

## Tech Stack

| Layer            | Technology                         |
|-------------------|-------------------------------------|
| UI Framework      | Streamlit                          |
| Data Processing   | Pandas, NumPy                      |
| Visualization     | Plotly                             |
| ML Models         | XGBoost, Prophet, scikit-learn     |
| Optimization      | Custom inventory math (EOQ, safety stock) |

## Project Structure

```
demand-forecasting/
│
├── app.py
│
├── data/
│   ├── raw/            # place your CSV (e.g. Kaggle train.csv) here
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── xgboost_model.py
│   │   └── prophet_model.py
│   ├── evaluator.py
│   ├── anomaly_detector.py
│   ├── inventory_optimizer.py
│   └── theme.py
│
├── app_pages/
│   ├── overview.py
│   ├── eda_explorer.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   └── inventory_optimizer.py
│
├── tests/
│   └── test_data_loader.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# 1. Clone or unzip the project
cd demand-forecasting

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note on Prophet:** Prophet can be tricky to install on some
> systems (it depends on `pystan`/`cmdstanpy`). If installation fails,
> the app will still run — the Forecasting page automatically falls
> back to a moving-average forecast with a warning message when
> Prophet is unavailable.

## How to Run

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

## Screenshots

> _Add screenshots of the Overview, EDA Explorer, Forecasting, Anomaly
> Detection, and Inventory Optimizer pages here._

| Page | Screenshot |
|------|------------|
| Overview | _placeholder_ |
| EDA Explorer | _placeholder_ |
| Forecasting | _placeholder_ |
| Anomaly Detection | _placeholder_ |
| Inventory Optimizer | _placeholder_ |

## Model Results (Example)

Results will vary by store/item combination and random seed, but a
typical comparison on the Kaggle "Store Item Demand Forecasting
Challenge" dataset looks like:

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Moving Average | ~12-18 | ~15-22 | ~15-25% |
| XGBoost | ~6-10 | ~9-14 | ~8-14% |
| Prophet | ~7-11 | ~10-15 | ~9-15% |

XGBoost typically performs best on this dataset due to its ability to
learn from lag/rolling features, while Prophet excels at capturing
explicit yearly/weekly seasonality with less tuning.

## Running Tests

```bash
python -m unittest discover tests
```

## Future Improvements

- Add hyperparameter tuning (Optuna/GridSearch) for XGBoost.
- Support hierarchical forecasting across stores/items.
- Add a multi-item batch mode with portfolio-level inventory dashboards.
- Persist and version trained models with a lightweight model registry.
- Add authentication and role-based views for planners vs. managers.
- Integrate real supplier lead-time and cost data via file upload.

## Author

Built as a complete, self-contained demonstration project for
retail demand forecasting and inventory optimization using Python
and Streamlit.
