"""
Simple moving-average baseline forecasting model.
"""

import numpy as np
import pandas as pd


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 30) -> dict:
    """
    Forecast future values using a simple moving average of the
    last `window` observations, held flat for the entire horizon.

    Parameters
    ----------
    series : pd.Series
        Historical sales values, indexed by date (DatetimeIndex)
        or with a parallel dates array handled by the caller.
    horizon : int
        Number of days to forecast.
    window : int
        Number of trailing days to average over.

    Returns
    -------
    dict with keys: forecast, dates, lower, upper
    """
    series = series.dropna()

    if len(series) == 0:
        avg = 0.0
    else:
        tail = series.tail(window) if len(series) >= window else series
        avg = float(tail.mean())

    forecast = [avg for _ in range(horizon)]

    if isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 0:
        last_date = series.index.max()
    else:
        last_date = pd.Timestamp.today().normalize()

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    lower = [v * 0.85 for v in forecast]
    upper = [v * 1.15 for v in forecast]

    return {
        "forecast": forecast,
        "dates": list(future_dates),
        "lower": lower,
        "upper": upper,
    }
