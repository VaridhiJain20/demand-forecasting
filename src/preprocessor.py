"""
Preprocessing utilities: enrich the raw sales dataframe with
calendar-derived features.
"""

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar features from the `date` column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a `date` column (datetime-like).

    Returns
    -------
    pd.DataFrame
        Enriched dataframe with additional calendar columns.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["month_name"] = out["date"].dt.month_name()
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["day_of_month"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_name"] = out["date"].dt.day_name()
    out["quarter"] = out["date"].dt.quarter
    out["is_weekend"] = out["day_of_week"].isin([5, 6])
    out["is_month_start"] = out["date"].dt.is_month_start
    out["is_month_end"] = out["date"].dt.is_month_end

    return out
