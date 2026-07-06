"""
Facebook Prophet based forecasting model.

Falls back gracefully to the moving-average baseline if Prophet
is not installed in the environment.
"""

import pandas as pd

from src.evaluator import calculate_metrics
from src.models.baseline import moving_average_forecast
from src.data_loader import make_continuous_daily_series

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def prophet_forecast(df: pd.DataFrame, store: int, item: int, horizon: int) -> dict:
    """
    Train a Prophet model on a single store/item series and
    forecast `horizon` days into the future.

    Returns
    -------
    dict with keys: historical_dates, historical_sales,
                     forecast_dates, forecast_values,
                     lower_bound, upper_bound, metrics
                     (and optionally 'warning' if Prophet was unavailable)
    """
    subset = make_continuous_daily_series(df, store, item, fill_method="zero")

    if subset.empty or len(subset) < 30:
        return {
            "historical_dates": list(subset["date"]) if not subset.empty else [],
            "historical_sales": list(subset["sales"]) if not subset.empty else [],
            "forecast_dates": [],
            "forecast_values": [],
            "lower_bound": [],
            "upper_bound": [],
            "metrics": {"mae": None, "rmse": None, "mape": None},
            "error": "Not enough history for this store/item to train Prophet (need at least 30 days).",
        }

    if not PROPHET_AVAILABLE:
        series = subset.set_index("date")["sales"]
        fallback = moving_average_forecast(series, horizon)
        return {
            "historical_dates": list(subset["date"]),
            "historical_sales": list(subset["sales"]),
            "forecast_dates": fallback["dates"],
            "forecast_values": fallback["forecast"],
            "lower_bound": fallback["lower"],
            "upper_bound": fallback["upper"],
            "metrics": {"mae": None, "rmse": None, "mape": None},
            "warning": (
                "Prophet is not installed in this environment. "
                "Showing a moving-average fallback forecast instead."
            ),
        }

    try:
        prophet_df = subset[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})

        # Hold out the last 30 days (or 20% of data) to compute metrics
        holdout_n = min(30, max(7, int(len(prophet_df) * 0.1)))
        train_df = prophet_df.iloc[:-holdout_n]
        test_df = prophet_df.iloc[-holdout_n:]

        eval_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode="multiplicative",
        )
        eval_model.fit(train_df)
        eval_future = eval_model.make_future_dataframe(periods=holdout_n)
        eval_forecast = eval_model.predict(eval_future)
        eval_preds = eval_forecast.tail(holdout_n)["yhat"].values
        metrics = calculate_metrics(test_df["y"].values, eval_preds)

        # Final model trained on all data for the real forecast
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode="multiplicative",
        )
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        future_only = forecast.tail(horizon)

        forecast_values = [max(v, 0.0) for v in future_only["yhat"].tolist()]
        lower_bound = [max(v, 0.0) for v in future_only["yhat_lower"].tolist()]
        upper_bound = [max(v, 0.0) for v in future_only["yhat_upper"].tolist()]

        return {
            "historical_dates": list(subset["date"]),
            "historical_sales": list(subset["sales"]),
            "forecast_dates": list(future_only["ds"]),
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "metrics": metrics,
        }

    except Exception as exc:
        series = subset.set_index("date")["sales"]
        fallback = moving_average_forecast(series, horizon)
        return {
            "historical_dates": list(subset["date"]),
            "historical_sales": list(subset["sales"]),
            "forecast_dates": fallback["dates"],
            "forecast_values": fallback["forecast"],
            "lower_bound": fallback["lower"],
            "upper_bound": fallback["upper"],
            "metrics": {"mae": None, "rmse": None, "mape": None},
            "warning": f"Prophet forecasting failed ({exc}). Showing a moving-average fallback instead.",
        }
