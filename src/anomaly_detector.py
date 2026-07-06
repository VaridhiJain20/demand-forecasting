"""
Anomaly detection for a single store/item sales series using
Isolation Forest plus a rolling-statistics based severity score.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data_loader import make_continuous_daily_series


def detect_anomalies(df: pd.DataFrame, store: int, item: int, contamination: float = 0.05) -> pd.DataFrame:
    """
    Detect anomalies in the sales series for a given store/item.

    Parameters
    ----------
    df : pd.DataFrame
        Full sales dataframe with columns date, store, item, sales.
    store : int
    item : int
    contamination : float
        Expected proportion of anomalies (passed to IsolationForest).

    Returns
    -------
    pd.DataFrame
        Columns: date, sales, is_anomaly, severity,
                 deviation_pct, expected_sales
    """
    subset = make_continuous_daily_series(df, store, item, fill_method="zero")

    if len(subset) < 14:
        subset["is_anomaly"] = False
        subset["severity"] = "NONE"
        subset["deviation_pct"] = 0.0
        subset["expected_sales"] = subset["sales"]
        return subset[["date", "sales", "is_anomaly", "severity", "deviation_pct", "expected_sales"]]

    subset["rolling_mean"] = subset["sales"].rolling(window=7, min_periods=1, center=True).mean()
    subset["rolling_std"] = subset["sales"].rolling(window=7, min_periods=1, center=True).std().fillna(0)

    features = subset[["sales", "rolling_mean", "rolling_std"]].fillna(0)

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(features)  # -1 = anomaly, 1 = normal

    subset["is_anomaly"] = predictions == -1
    subset["expected_sales"] = subset["rolling_mean"]

    # Deviation percentage from expected (rolling mean) sales
    safe_expected = subset["expected_sales"].replace(0, np.nan)
    subset["deviation_pct"] = (
        (subset["sales"] - subset["expected_sales"]).abs() / safe_expected * 100
    ).fillna(0)

    def _severity(row):
        if not row["is_anomaly"]:
            return "NONE"
        if row["deviation_pct"] > 60:
            return "HIGH"
        elif row["deviation_pct"] > 30:
            return "MEDIUM"
        else:
            return "LOW"

    subset["severity"] = subset.apply(_severity, axis=1)

    return subset[["date", "sales", "is_anomaly", "severity", "deviation_pct", "expected_sales"]]
