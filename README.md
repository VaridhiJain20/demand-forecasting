


# Retail Demand Forecasting and Inventory Optimization System

A Streamlit application for retail demand forecasting and inventory planning. It takes historical store-item sales data and produces exploratory analysis, multi-model demand forecasts, anomaly detection, and inventory policy calculations (safety stock, reorder point, EOQ). The app accepts a CSV upload directly in the browser , and works with the Kaggle Store Item Demand Forecasting Challenge dataset or any sales data with date, store, item, and sales fields. Output is an interactive five-page dashboard rather than a static report.

---

## Problem Statement

Retailers lose sales when they run out of stock and lose money when they hold too much of it. Both problems come down to the same root cause: demand forecasts that are either missing or inaccurate, and inventory rules that don't account for how much daily demand actually varies. This project addresses that by combining forecasting, anomaly detection, and inventory math in one tool that works on a store-item level.

---

## Live Demo

Live application: [https://demand-forecasting-varidhi.streamlit.app/]

Upload your own sales CSV, or select the sample dataset available in the Sample data/ folder of this repository to test the application. Navigate through the Overview, EDA Explorer, Forecasting, Anomaly Detection, and Inventory Optimizer pages using the sidebar.

---

## Features

Data Analysis
- KPI summary (total records, stores, items, average daily sales) and a 12-month trend chart on the Overview page
- Day-of-week and monthly average sales breakdowns
- Interactive EDA Explorer with date-range filtering, daily/weekly/monthly aggregation, and an optional rolling average overlay
- Store comparison and item comparison views for a selected item or store
- Sales distribution histogram and a box plot of sales by store

Forecasting
- Three models: Moving Average baseline, XGBoost, and Prophet
- Automatic side-by-side comparison table across all three models, ranked by MAE
- Forecast horizon selectable at 30, 60, or 90 days
- Forecast output table with predicted value and lower/upper bounds, downloadable as CSV

Anomaly Detection
- Isolation Forest based detection on the daily sales series, with an adjustable contamination rate
- Severity classification into HIGH, MEDIUM, and LOW based on percentage deviation from a rolling expected value
- Anomaly breakdown by month and day of week, and a severity distribution chart
- Anomaly table exportable as CSV

Inventory Optimization
- Safety stock, reorder point, and Economic Order Quantity (EOQ) calculations from historical demand
- Configurable lead time, service level, unit price, ordering cost, and holding cost
- Inventory status classification (CRITICAL, LOW, HEALTHY, OVERSTOCK) with a plain-language recommendation
- Stockout cost risk and monthly holding cost estimates

---

## Tech Stack

| Layer            | Technology                         |
|-------------------|-------------------------------------|
| UI Framework      | Streamlit                          |
| Data Processing   | Pandas, NumPy                      |
| Visualization     | Plotly                             |
| ML Models         | XGBoost, Prophet, scikit-learn     |
| Optimization      | Custom inventory math (EOQ, safety stock) |

---

## Project Structure

```
demand-forecasting/
│
├── app.py
│
├── data/
│   ├── raw/            
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
---

## Dataset

Source: Kaggle Store Item Demand Forecasting Challenge

The app expects four fields per row: a date, a store identifier, an item identifier, and a sales quantity. Column names don't need to match exactly — the loader auto-detects common alternates (for example `order_date`, `store_id`, `sku`, `units_sold`) and asks for manual mapping only if it can't find a confident match.

---

## Setup and Installation

Prerequisites:
- Python 3.10 or above
- pip

Steps:

1. Clone the repository and move into it:
   ```
   git clone <https://github.com/VaridhiJain20/demand-forecasting>
   cd demand-forecasting
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your data:
   ```
   cp /path/to/your/train.csv data/raw/train.csv
   ```
   Alternatively, skip this step and upload a CSV directly in the app's sidebar once it's running.

4. Run the app:
   ```
   streamlit run app.py
   ```

---

## How It Works

The app starts by loading data either from a sidebar CSV upload or from `data/raw/train.csv`. `data_loader.py` parses the file, matches its columns to date/store/item/sales using a synonym list, and coerces types (dates parsed, IDs cast to int where possible, sales coerced to numeric with bad values filled to zero). If a column can't be matched automatically, the app raises a `ColumnMappingRequired` exception and shows a form asking the user to pick the right columns manually.

Once loaded, `validate_raw_data` runs data-quality checks (missing values, duplicates, negative sales, date gaps) that populate the sidebar summary. `preprocess` then adds calendar fields such as year, month, day of week, and quarter, used throughout the Overview and EDA pages.

For forecasting, a single store-item series is extracted with `make_continuous_daily_series`, which fills any missing calendar days with zero sales so the series is continuous. `feature_engineer.create_features` builds lag (1, 7, 14, 30 days), rolling mean/std/max/min, and exponentially weighted features for the XGBoost model. XGBoost is evaluated with `TimeSeriesSplit` cross-validation and then forecasts recursively, feeding each day's prediction back in as input for the next. Prophet is fit directly on the daily series with weekly and yearly seasonality, evaluated on a holdout tail, and falls back to the moving-average baseline if Prophet isn't installed or fails to fit.

Anomaly detection also runs on the continuous daily series: `anomaly_detector.py` computes a rolling 7-day mean and standard deviation, feeds sales plus these rolling statistics into an Isolation Forest, and classifies flagged points as HIGH, MEDIUM, or LOW severity based on percentage deviation from the rolling mean.

Inventory calculations in `inventory_optimizer.py` use the same historical series: safety stock from the standard deviation of daily sales scaled by a service-level z-score and the square root of lead time, reorder point as mean daily demand times lead time plus safety stock, and EOQ from the standard square-root formula using ordering and holding costs.

---

## Model Details

| Model | Approach | Key Parameters | Evaluation |
|-------|----------|-----------------|------------|
| Moving Average | Averages the trailing window of daily sales and holds that average flat across the forecast horizon | 30-day trailing window | MAE, RMSE, MAPE computed on a holdout tail |

| XGBoost | Gradient-boosted trees trained on lag and rolling features, forecasting recursively one day at a time | `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`, `TimeSeriesSplit(n_splits=3)` | MAE, RMSE, MAPE averaged across CV folds |

| Prophet | Additive time-series model with explicit yearly and weekly seasonality | `seasonality_mode="multiplicative"`, `yearly_seasonality=True`, `weekly_seasonality=True` | MAE, RMSE, MAPE, R2 computed on a holdout tail (last 30 days or 10% of the series) |

All three models are run together on the Forecasting page and ranked by MAE in a comparison table.

---

## Results

The app computes MAE, RMSE, and MAPE for every model at run time, per store-item combination and forecast horizon selected by the user. These numbers depend on which store, item, and horizon are chosen, so no fixed results table is included here — running the app against the dataset produces the comparison table live on the Forecasting page.

---

## Limitations and Future Work

Limitations:
- Forecasts and inventory calculations operate on one store-item pair at a time; there is no batch mode across the full catalog.
- The XGBoost model requires at least 60 days of history and Prophet requires at least 30, so new or low-volume store-item combinations may not get a forecast.
- Only CSV input is supported; other formats such as Excel or JSON are rejected at the upload step.

Future Work:
- Add a batch forecasting mode that runs all models across every store-item combination and exports a consolidated report.
- Persist trained models instead of retraining on every page load or selection change.
- Extend inventory optimization to account for multiple suppliers or variable lead times instead of a single fixed lead time.

