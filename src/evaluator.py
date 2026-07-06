"""
Model evaluation utilities: error metrics and model comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def calculate_metrics(actual, predicted) -> dict:
    """
    Compute MAE, RMSE, MAPE, and R2 between actual and predicted
    arrays.

    Parameters
    ----------
    actual : array-like
    predicted : array-like

    Returns
    -------
    dict with keys: mae, rmse, mape, r2
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return {"mae": None, "rmse": None, "mape": None, "r2": None}

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Avoid division by zero in MAPE
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100
        )
    else:
        mape = None

    try:
        r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else None
    except Exception:
        r2 = None

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compare multiple models' metrics and rank them by MAE.

    Parameters
    ----------
    results : dict
        Mapping of model_name -> metrics dict (as returned by
        calculate_metrics or a model's `metrics` field).

    Returns
    -------
    pd.DataFrame
        Ranked comparison table with columns:
        Model, MAE, RMSE, MAPE, Best
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": metrics.get("mae"),
                "RMSE": metrics.get("rmse"),
                "MAPE": metrics.get("mape"),
            }
        )

    comparison_df = pd.DataFrame(rows)

    # Rank by MAE ascending (lower is better); ignore rows without MAE
    valid = comparison_df.dropna(subset=["MAE"])
    if not valid.empty:
        best_model = valid.loc[valid["MAE"].idxmin(), "Model"]
    else:
        best_model = None

    comparison_df["Best"] = comparison_df["Model"] == best_model
    comparison_df = comparison_df.sort_values(
        by="MAE", na_position="last"
    ).reset_index(drop=True)

    return comparison_df
