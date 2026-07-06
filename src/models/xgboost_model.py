"""
XGBoost-based demand forecasting model with recursive multi-step
forecasting and time-series cross validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.feature_engineer import create_features
from src.evaluator import calculate_metrics
from src.data_loader import make_continuous_daily_series

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30",
    "rolling_std_7", "rolling_max_7", "rolling_min_7",
    "ewm_7", "ewm_30",
    "day_of_week", "month", "is_weekend",
]


def _prep_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def _make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror",
    )


def xgboost_forecast(df: pd.DataFrame, store: int, item: int, horizon: int) -> dict:
    """
    Train an XGBoost model on a single store/item series and
    produce a recursive multi-step forecast.

    Returns
    -------
    dict with keys: historical_dates, historical_sales,
                     forecast_dates, forecast_values,
                     lower_bound, upper_bound, metrics
    """
    try:
        subset = make_continuous_daily_series(df, store, item, fill_method="zero")
        subset = _prep_calendar(subset)

        if len(subset) < 60:
            raise ValueError("Not enough history to train XGBoost (need at least 60 days).")

        featured = create_features(subset[["date", "sales"]].copy())
        featured = _prep_calendar(featured)

        X = featured[FEATURE_COLS]
        y = featured["sales"]

        # Time series cross-validated metrics
        tscv = TimeSeriesSplit(n_splits=3)
        fold_maes, fold_rmses, fold_mapes = [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = _make_model()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            m = calculate_metrics(y_test.values, preds)
            fold_maes.append(m["mae"])
            fold_rmses.append(m["rmse"])
            fold_mapes.append(m["mape"])

        metrics = {
            "mae": float(np.mean(fold_maes)),
            "rmse": float(np.mean(fold_rmses)),
            "mape": float(np.mean(fold_mapes)),
        }

        # Final model trained on all available data
        final_model = _make_model()
        final_model.fit(X, y)

        # Recursive forecasting
        history = subset[["date", "sales"]].copy()
        last_date = history["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        working_series = history.copy()
        forecast_values = []

        for future_date in future_dates:
            # Build a single row of features for the future date manually
            shifted = working_series["sales"]
            lag_1 = shifted.iloc[-1] if len(shifted) >= 1 else 0
            lag_7 = shifted.iloc[-7] if len(shifted) >= 7 else lag_1
            lag_14 = shifted.iloc[-14] if len(shifted) >= 14 else lag_1
            lag_30 = shifted.iloc[-30] if len(shifted) >= 30 else lag_1

            tail7 = shifted.tail(7)
            tail30 = shifted.tail(30)

            row = {
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_30": lag_30,
                "rolling_mean_7": tail7.mean(),
                "rolling_mean_30": tail30.mean(),
                "rolling_std_7": tail7.std() if len(tail7) > 1 else 0.0,
                "rolling_max_7": tail7.max(),
                "rolling_min_7": tail7.min(),
                "ewm_7": shifted.ewm(span=7, adjust=False).mean().iloc[-1],
                "ewm_30": shifted.ewm(span=30, adjust=False).mean().iloc[-1],
                "day_of_week": future_date.dayofweek,
                "month": future_date.month,
                "is_weekend": int(future_date.dayofweek in [5, 6]),
            }
            row_df = pd.DataFrame([row])[FEATURE_COLS]
            pred = float(final_model.predict(row_df)[0])
            pred = max(pred, 0.0)
            forecast_values.append(pred)

            working_series = pd.concat(
                [working_series, pd.DataFrame({"date": [future_date], "sales": [pred]})],
                ignore_index=True,
            )

        lower_bound = [v * 0.88 for v in forecast_values]
        upper_bound = [v * 1.12 for v in forecast_values]

        return {
            "historical_dates": list(history["date"]),
            "historical_sales": list(history["sales"]),
            "forecast_dates": list(future_dates),
            "forecast_values": forecast_values,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "metrics": metrics,
        }

    except Exception as exc:
        return {
            "historical_dates": [],
            "historical_sales": [],
            "forecast_dates": [],
            "forecast_values": [],
            "lower_bound": [],
            "upper_bound": [],
            "metrics": {"mae": None, "rmse": None, "mape": None},
            "error": f"XGBoost forecasting failed: {exc}",
        }
