"""
Forecasting page: run Moving Average, XGBoost, or Prophet models
and visualize forecasts with confidence intervals.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import PLOTLY_LAYOUT_DEFAULTS, PRIMARY, WARNING, FORECAST_HORIZONS
from src.data_loader import make_continuous_daily_series
from src.models.baseline import moving_average_forecast
from src.models.xgboost_model import xgboost_forecast
from src.models.prophet_model import prophet_forecast, PROPHET_AVAILABLE
from src.evaluator import calculate_metrics, compare_models


MODEL_INFO = {
    "Moving Average": (
        "A simple baseline that averages the last 30 days of sales and "
        "projects that flat average into the future. Fast, but doesn't "
        "capture trend or seasonality."
    ),
    "XGBoost": (
        "A gradient-boosted tree model that learns from lag and rolling "
        "features (recent sales history, day of week, month) to make "
        "recursive day-by-day predictions. Captures complex patterns."
    ),
    "Prophet": (
        "Facebook's Prophet model, designed specifically for business "
        "time series. It explicitly models yearly and weekly seasonality "
        "plus trend changes."
    ),
}


def _run_model(model_choice, df, store, item, horizon):
    if model_choice == "Moving Average":
        subset = make_continuous_daily_series(df, store, item, fill_method="zero")
        if subset.empty:
            return {
                "historical_dates": [], "historical_sales": [],
                "forecast_dates": [], "forecast_values": [],
                "lower_bound": [], "upper_bound": [],
                "metrics": {"mae": None, "rmse": None, "mape": None},
                "error": "No historical data available for this store/item combination.",
            }
        series = subset.set_index("date")["sales"]

        # Compute metrics via a simple holdout (skip if too little history)
        if len(series) >= 15:
            holdout_n = min(30, max(7, int(len(series) * 0.1)))
            train_series = series.iloc[:-holdout_n]
            test_series = series.iloc[-holdout_n:]
            fitted = moving_average_forecast(train_series, holdout_n)
            metrics = calculate_metrics(test_series.values, fitted["forecast"])
        else:
            metrics = {"mae": None, "rmse": None, "mape": None}

        result = moving_average_forecast(series, horizon)
        return {
            "historical_dates": list(series.index),
            "historical_sales": list(series.values),
            "forecast_dates": result["dates"],
            "forecast_values": result["forecast"],
            "lower_bound": result["lower"],
            "upper_bound": result["upper"],
            "metrics": metrics,
        }
    elif model_choice == "XGBoost":
        return xgboost_forecast(df, store, item, horizon)
    else:
        return prophet_forecast(df, store, item, horizon)


def show(store, item):
    df = st.session_state.get("full_data")
    if df is None or df.empty:
        st.warning("No data available. Please reload the app.")
        return

    st.title("Forecasting")
    st.caption(f"Generate demand forecasts for Store {store}, Item {item}.")

    model_options = ["Moving Average", "XGBoost"]
    if PROPHET_AVAILABLE:
        model_options.append("Prophet")

    col1, col2, col3 = st.columns(3)

    with col1:
        model_choice = st.radio("Model", model_options)
        if not PROPHET_AVAILABLE:
            st.caption("Prophet isn't installed in this environment, so it's hidden from the model list.")

    with col2:
        horizon = st.select_slider("Forecast Days", options=FORECAST_HORIZONS)

    with col3:
        st.info(MODEL_INFO[model_choice])

    generate = st.button("Generate Forecast", type="primary")

    if generate:
        st.session_state["forecast_ran"] = True
        st.session_state["forecast_config"] = {
            "model_choice": model_choice,
            "horizon": horizon,
            "store": store,
            "item": item,
        }

    if not st.session_state.get("forecast_ran"):
        st.info("Configure your options above and click **Generate Forecast** to begin.")
        return

    config = st.session_state["forecast_config"]
    model_choice = config["model_choice"]
    horizon = config["horizon"]

    with st.spinner(f"Training {model_choice} and generating forecast..."):
        result = _run_model(model_choice, df, store, item, horizon)

    if result.get("error"):
        st.error(result["error"])
        return
    if result.get("warning"):
        st.warning(result["warning"])

    metrics = result.get("metrics", {})

    st.divider()

    # ---------------------------------------------------------------
    # Metrics row
    # ---------------------------------------------------------------
    m1, m2, m3 = st.columns(3)
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    mape = metrics.get("mape")
    m1.metric("MAE", f"{mae:.2f}" if mae is not None else "N/A")
    m2.metric("RMSE", f"{rmse:.2f}" if rmse is not None else "N/A")
    m3.metric("MAPE", f"{mape:.2f}%" if mape is not None else "N/A")

    # ---------------------------------------------------------------
    # Main forecast chart
    # ---------------------------------------------------------------
    hist_dates = pd.to_datetime(pd.Series(result["historical_dates"]))
    hist_sales = result["historical_sales"]

    tail_n = 90
    hist_dates_tail = hist_dates.tail(tail_n)
    hist_sales_tail = hist_sales[-tail_n:] if len(hist_sales) > tail_n else hist_sales

    forecast_dates = pd.to_datetime(pd.Series(result["forecast_dates"]))
    forecast_values = result["forecast_values"]
    lower_bound = result["lower_bound"]
    upper_bound = result["upper_bound"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_dates_tail, y=hist_sales_tail, mode="lines",
            name="Historical", line=dict(color=PRIMARY, width=2),
        )
    )

    if len(forecast_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates, y=forecast_values, mode="lines",
                name="Forecast", line=dict(color=WARNING, width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill="toself",
                fillcolor="rgba(217,119,6,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Interval",
                showlegend=True,
            )
        )
        today_marker = forecast_dates.min()
        fig.add_vline(x=today_marker, line_width=1, line_dash="dot", line_color="white")

    fig.update_layout(
        title=f"Forecast: Store {store}, Item {item} — {horizon} Day Horizon ({model_choice})",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------
    # Model comparison table (run all 3 automatically)
    # ---------------------------------------------------------------
    st.subheader("Model Comparison")
    with st.spinner("Comparing all models..."):
        all_results = {}
        for name in model_options:
            try:
                r = _run_model(name, df, store, item, horizon)
                all_results[name] = r.get("metrics", {"mae": None, "rmse": None, "mape": None})
            except Exception:
                all_results[name] = {"mae": None, "rmse": None, "mape": None}

        comparison_df = compare_models(all_results)

    def _highlight_best(row):
        return ["background-color: rgba(22,163,74,0.35)" if row["Best"] else "" for _ in row]

    st.dataframe(
        comparison_df.style.apply(_highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ---------------------------------------------------------------
    # Forecast data table
    # ---------------------------------------------------------------
    st.subheader("Forecast Data")
    if len(forecast_dates) > 0:
        forecast_table = pd.DataFrame(
            {
                "Date": forecast_dates.dt.strftime("%Y-%m-%d"),
                "Predicted": [round(v, 2) for v in forecast_values],
                "Lower": [round(v, 2) for v in lower_bound],
                "Upper": [round(v, 2) for v in upper_bound],
            }
        )
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        csv = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Forecast CSV",
            data=csv,
            file_name=f"forecast_store{store}_item{item}.csv",
            mime="text/csv",
        )
    else:
        st.info("No forecast values were generated.")
