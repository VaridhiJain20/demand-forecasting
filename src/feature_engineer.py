"""
Feature engineering for machine-learning based forecasting models
(lag features, rolling statistics, exponential weighted stats).
"""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag, rolling, and exponentially weighted features from
    a single store/item time series (expects a `sales` column,
    sorted by date).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `date` and `sales` columns, already sorted
        chronologically for a single store/item combination.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional feature columns; rows containing
        NaNs introduced by shifting/rolling are dropped.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    # Lag features
    out["lag_1"] = out["sales"].shift(1)
    out["lag_7"] = out["sales"].shift(7)
    out["lag_14"] = out["sales"].shift(14)
    out["lag_30"] = out["sales"].shift(30)

    # Rolling window features (based on lag_1 to avoid leakage of current day)
    shifted = out["sales"].shift(1)
    out["rolling_mean_7"] = shifted.rolling(window=7).mean()
    out["rolling_mean_30"] = shifted.rolling(window=30).mean()
    out["rolling_std_7"] = shifted.rolling(window=7).std()
    out["rolling_max_7"] = shifted.rolling(window=7).max()
    out["rolling_min_7"] = shifted.rolling(window=7).min()

    # Exponentially weighted features
    out["ewm_7"] = shifted.ewm(span=7, adjust=False).mean()
    out["ewm_30"] = shifted.ewm(span=30, adjust=False).mean()

    out = out.dropna().reset_index(drop=True)

    return out
