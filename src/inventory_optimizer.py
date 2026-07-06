"""
Inventory optimization calculations: safety stock, reorder point,
economic order quantity (EOQ), and inventory status/recommendations.
"""

import math
import numpy as np
import pandas as pd

from src.config import Z_SCORES, SUCCESS, WARNING, DANGER, INFO


def calculate_safety_stock(df: pd.DataFrame, service_level: float, lead_time: int) -> float:
    """
    Safety stock = Z * std(daily sales) * sqrt(lead_time)
    """
    if df is None or len(df) == 0:
        return 0.0

    z = Z_SCORES.get(service_level, 1.645)
    std_sales = float(df["sales"].std()) if len(df) > 1 else 0.0
    std_sales = 0.0 if pd.isna(std_sales) else std_sales

    safety_stock = z * std_sales * math.sqrt(max(lead_time, 0))
    return float(max(safety_stock, 0.0))


def calculate_reorder_point(df: pd.DataFrame, lead_time: int, safety_stock: float) -> float:
    """
    Reorder point = mean(daily sales) * lead_time + safety_stock
    """
    if df is None or len(df) == 0:
        return safety_stock

    mean_sales = float(df["sales"].mean())
    mean_sales = 0.0 if pd.isna(mean_sales) else mean_sales

    reorder_point = mean_sales * lead_time + safety_stock
    return float(max(reorder_point, 0.0))


def calculate_eoq(annual_demand: float, ordering_cost: float = 100, holding_cost_per_unit: float = 2) -> float:
    """
    Economic Order Quantity = sqrt(2 * D * S / H)
    """
    if annual_demand <= 0 or holding_cost_per_unit <= 0:
        return 0.0

    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    return float(eoq)


def get_inventory_status(current_stock: float, safety_stock: float, reorder_point: float, max_stock: float) -> dict:
    """
    Classify the current inventory position into a status with a
    color, message, and recommended action.
    """
    if current_stock <= safety_stock:
        return {
            "status": "CRITICAL",
            "color": DANGER,
            "message": "Stock is at or below the safety stock level. Stockout risk is high.",
            "action": "Place an emergency order immediately to avoid running out.",
        }
    elif current_stock <= reorder_point:
        return {
            "status": "LOW",
            "color": WARNING,
            "message": "Stock has fallen to or below the reorder point.",
            "action": "Place a replenishment order now based on the recommended EOQ.",
        }
    elif current_stock <= max_stock:
        return {
            "status": "HEALTHY",
            "color": SUCCESS,
            "message": "Stock level is within the healthy operating range.",
            "action": "No action needed. Continue monitoring regularly.",
        }
    else:
        return {
            "status": "OVERSTOCK",
            "color": INFO,
            "message": "Stock level exceeds the recommended maximum.",
            "action": "Consider slowing down future orders or running a promotion to reduce excess inventory.",
        }


def get_recommendation(current_stock: float, reorder_point: float, eoq: float, safety_stock: float) -> str:
    """
    Produce a human-readable recommendation based on current stock
    relative to the reorder point.
    """
    if current_stock <= safety_stock:
        return (
            f"Stock ({current_stock:.0f} units) is at or below safety stock "
            f"({safety_stock:.0f} units). Order {eoq:.0f} units immediately to avoid a stockout."
        )
    elif current_stock <= reorder_point:
        shortfall = reorder_point - current_stock
        return (
            f"Stock ({current_stock:.0f} units) has reached the reorder point "
            f"({reorder_point:.0f} units). Place an order of {eoq:.0f} units now. "
            f"You are {shortfall:.0f} units below the reorder point."
        )
    else:
        buffer_units = current_stock - reorder_point
        return (
            f"Stock ({current_stock:.0f} units) is healthy, "
            f"{buffer_units:.0f} units above the reorder point ({reorder_point:.0f} units). "
            f"No order needed right now; the recommended order quantity when you do reorder is {eoq:.0f} units."
        )
