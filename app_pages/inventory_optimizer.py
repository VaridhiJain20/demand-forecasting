"""
Inventory Optimizer page: safety stock, reorder point, EOQ, and
inventory health visualization for a selected store/item.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import PLOTLY_LAYOUT_DEFAULTS, SUCCESS, WARNING, DANGER, INFO, SERVICE_LEVELS
from src.data_loader import make_continuous_daily_series
from src.inventory_optimizer import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    get_inventory_status,
    get_recommendation,
)


def show(store, item):
    df = st.session_state.get("full_data")
    if df is None or df.empty:
        st.warning("No data available. Please reload the app.")
        return

    st.title("Inventory Optimizer")
    st.caption(
        f"Calculate safety stock, reorder points, and order quantities for "
        f"Store {store}, Item {item} based on historical demand."
    )

    subset = make_continuous_daily_series(df, store, item, fill_method="zero")
    if subset.empty:
        st.info("No historical data available for this store/item combination.")
        return

    left_col, right_col = st.columns([2, 3])

    with left_col:
        st.subheader("Parameters")

        current_stock = st.number_input(
            "Current Stock Level (units)", min_value=0, value=200, step=10,
            help="The number of units you have in stock right now.",
        )
        lead_time = st.slider(
            "Lead Time (days)", 1, 30, 7,
            help="The number of days between placing an order and receiving it.",
        )
        service_level = st.select_slider(
            "Service Level",
            options=SERVICE_LEVELS,
            value=0.95,
            format_func=lambda x: f"{x*100:.0f}%",
            help="The probability you want of not running out of stock before the next order arrives.",
        )
        unit_price = st.number_input(
            "Unit Price (₹)", min_value=0.0, value=100.0, step=10.0,
            help="The selling price of one unit.",
        )
        ordering_cost = st.number_input(
            "Ordering Cost (₹)", min_value=0.0, value=500.0, step=50.0,
            help="The fixed cost of placing one order, no matter how many units it's for.",
        )
        holding_cost = st.number_input(
            "Holding Cost per unit/day (₹)", min_value=0.0, value=2.0, step=0.5,
            help="The cost of storing one unit for one day.",
        )

        calculate_btn = st.button("Calculate", type="primary", use_container_width=True)

    if calculate_btn:
        st.session_state["inventory_ran"] = True
        st.session_state["inventory_config"] = {
            "current_stock": current_stock,
            "lead_time": lead_time,
            "service_level": service_level,
            "unit_price": unit_price,
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost,
        }

    with right_col:
        if not st.session_state.get("inventory_ran"):
            st.info("Set your parameters on the left and click **Calculate** to see results.")
            return

        cfg = st.session_state["inventory_config"]

        try:
            safety_stock = calculate_safety_stock(subset, cfg["service_level"], cfg["lead_time"])
            reorder_point = calculate_reorder_point(subset, cfg["lead_time"], safety_stock)

            avg_daily_demand = float(subset["sales"].mean())
            annual_demand = avg_daily_demand * 365
            eoq = calculate_eoq(annual_demand, cfg["ordering_cost"], cfg["holding_cost"])

            max_stock = reorder_point + eoq

            status_info = get_inventory_status(cfg["current_stock"], safety_stock, reorder_point, max_stock)
            recommendation = get_recommendation(cfg["current_stock"], reorder_point, eoq, safety_stock)
        except Exception as exc:
            st.error(f"Calculation error: {exc}")
            return

        # -----------------------------------------------------------
        # Status badge
        # -----------------------------------------------------------
        status = status_info["status"]
        message = f"{status_info['message']} — {status_info['action']}"
        if status == "CRITICAL":
            st.error(f"**{status}**: {message}")
        elif status == "LOW":
            st.warning(f"**{status}**: {message}")
        elif status == "HEALTHY":
            st.success(f"**{status}**: {message}")
        else:
            st.info(f"**{status}**: {message}")

        # -----------------------------------------------------------
        # Metric cards
        # -----------------------------------------------------------
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Safety Stock",
            f"{safety_stock:.0f} units",
            help="Extra stock kept on hand to absorb unexpected spikes in demand.",
        )
        m2.metric(
            "Reorder Point",
            f"{reorder_point:.0f} units",
            help="The stock level at which you should place a new order.",
        )
        m3.metric(
            "EOQ",
            f"{eoq:.0f} units",
            help="The order size that costs the least overall, balancing ordering costs against holding costs.",
        )

        # -----------------------------------------------------------
        # Recommendation
        # -----------------------------------------------------------
        st.info(f"**Recommendation:** {recommendation}")

        # -----------------------------------------------------------
        # Inventory gauge visualization
        # -----------------------------------------------------------
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[safety_stock], y=["Inventory"], orientation="h",
            marker_color=DANGER, name="Critical Zone (0 - Safety Stock)",
        ))
        fig.add_trace(go.Bar(
            x=[max(reorder_point - safety_stock, 0)], y=["Inventory"], orientation="h",
            marker_color=WARNING, name="Warning Zone (Safety Stock - Reorder Point)",
        ))
        fig.add_trace(go.Bar(
            x=[max(max_stock - reorder_point, 0)], y=["Inventory"], orientation="h",
            marker_color=SUCCESS, name="Healthy Zone (Reorder Point - Max Stock)",
        ))

        fig.add_vline(
            x=cfg["current_stock"], line_width=3, line_color="white", line_dash="dot",
            annotation_text=f"Current Stock: {cfg['current_stock']:.0f}",
            annotation_position="top",
        )

        fig.update_layout(
            barmode="stack",
            title="Inventory Position",
            xaxis_title="Units",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------------------
        # Cost analysis
        # -----------------------------------------------------------
        st.subheader("Cost Analysis")
        c1, c2 = st.columns(2)
        with c1:
            stockout_units = max(reorder_point - cfg["current_stock"], 0)
            stockout_cost_risk = stockout_units * cfg["unit_price"]
            st.metric("Stockout Cost Risk", f"₹{stockout_cost_risk:,.0f}")
            st.caption("Estimated revenue at risk if stock runs out before the next order arrives.")
        with c2:
            monthly_holding_cost = cfg["current_stock"] * cfg["holding_cost"] * 30
            st.metric("Monthly Holding Cost", f"₹{monthly_holding_cost:,.0f}")
            st.caption("Estimated cost of holding the current stock level for 30 days.")

        # -----------------------------------------------------------
        # Historical demand stats
        # -----------------------------------------------------------
        st.subheader("Historical Demand Stats")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Avg Daily Demand", f"{subset['sales'].mean():.1f}")
        h2.metric("Std Deviation", f"{subset['sales'].std():.1f}")
        h3.metric("Max Single Day", f"{subset['sales'].max():.0f}")
        h4.metric("Min Single Day", f"{subset['sales'].min():.0f}")
