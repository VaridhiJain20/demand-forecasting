"""
Overview page: high-level KPIs and trends across the whole dataset.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PLOTLY_LAYOUT_DEFAULTS


def show():
    df = st.session_state.get("full_data")
    if df is None or df.empty:
        st.warning("No data available. Please reload the app.")
        return

    st.title("Overview")
    st.caption("A high-level snapshot of sales performance across all stores and items.")

    # ---------------------------------------------------------------
    # KPI cards
    # ---------------------------------------------------------------
    total_records = len(df)
    total_stores = df["store"].nunique()
    total_items = df["item"].nunique()
    avg_daily_sales = df.groupby("date")["sales"].sum().mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Total Stores", f"{total_stores}")
    col3.metric("Total Items", f"{total_items}")
    col4.metric("Avg Daily Sales", f"{avg_daily_sales:,.0f}")

    st.divider()

    # ---------------------------------------------------------------
    # Full-width trend chart (last 365 days, all stores combined)
    # ---------------------------------------------------------------
    daily_totals = df.groupby("date", as_index=False)["sales"].sum()
    daily_totals = daily_totals.sort_values("date")
    last_year = daily_totals.tail(365)

    fig_trend = px.line(
        last_year,
        x="date",
        y="sales",
        title="Overall Sales Trend (Last 12 Months)",
        labels={"date": "Date", "sales": "Total Sales"},
    )
    fig_trend.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ---------------------------------------------------------------
    # Top stores / top items
    # ---------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        store_totals = (
            df.groupby("store", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=True)
        )
        store_totals["store"] = "Store " + store_totals["store"].astype(str)
        fig_stores = px.bar(
            store_totals,
            x="sales",
            y="store",
            orientation="h",
            title="Top Stores by Total Sales",
            labels={"sales": "Total Sales", "store": "Store"},
        )
        fig_stores.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_stores, use_container_width=True)

    with col_right:
        item_totals = (
            df.groupby("item", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
            .head(10)
            .sort_values("sales", ascending=True)
        )
        item_totals["item"] = "Item " + item_totals["item"].astype(str)
        fig_items = px.bar(
            item_totals,
            x="sales",
            y="item",
            orientation="h",
            title="Top Items by Total Sales",
            labels={"sales": "Total Sales", "item": "Item"},
        )
        fig_items.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_items, use_container_width=True)

    # ---------------------------------------------------------------
    # Day-of-week / month averages
    # ---------------------------------------------------------------
    col_left2, col_right2 = st.columns(2)

    df_calendar = df.copy()
    df_calendar["day_name"] = pd.to_datetime(df_calendar["date"]).dt.day_name()
    df_calendar["month_name"] = pd.to_datetime(df_calendar["date"]).dt.month_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    with col_left2:
        dow_avg = df_calendar.groupby("day_name")["sales"].mean().reindex(day_order).reset_index()
        fig_dow = px.bar(
            dow_avg,
            x="day_name",
            y="sales",
            title="Average Sales by Day of Week",
            labels={"day_name": "Day of Week", "sales": "Average Sales"},
        )
        fig_dow.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_right2:
        month_avg = df_calendar.groupby("month_name")["sales"].mean().reindex(month_order).reset_index()
        fig_month = px.bar(
            month_avg,
            x="month_name",
            y="sales",
            title="Average Sales by Month",
            labels={"month_name": "Month", "sales": "Average Sales"},
        )
        fig_month.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_month, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------
    # Summary statistics table
    # ---------------------------------------------------------------
    st.subheader("Summary Statistics")
    summary = {
        "Metric": ["Minimum", "Maximum", "Mean", "Std Dev", "Total", "Median"],
        "Value": [
            df["sales"].min(),
            df["sales"].max(),
            round(df["sales"].mean(), 2),
            round(df["sales"].std(), 2),
            df["sales"].sum(),
            df["sales"].median(),
        ],
    }
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
