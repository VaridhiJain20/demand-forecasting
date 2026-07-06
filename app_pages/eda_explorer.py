"""
EDA Explorer page: interactive filters and pattern analysis for a
selected store/item combination (and cross-store/item comparisons).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PLOTLY_LAYOUT_DEFAULTS


def show(store, item):
    df = st.session_state.get("full_data")
    if df is None or df.empty:
        st.warning("No data available. Please reload the app.")
        return

    st.title("EDA Explorer")
    st.caption(f"Exploring sales patterns for Store {store}, Item {item}.")

    subset = df[(df["store"] == store) & (df["item"] == item)].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date")

    if subset.empty:
        st.info("No data for this store/item combination.")
        return

    # ---------------------------------------------------------------
    # Filters
    # ---------------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        min_date = subset["date"].min().to_pydatetime()
        max_date = subset["date"].max().to_pydatetime()
        date_range = st.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
        )
    with col2:
        aggregation = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"])

    filtered = subset[(subset["date"] >= date_range[0]) & (subset["date"] <= date_range[1])].copy()

    if aggregation == "Weekly":
        plot_df = (
            filtered.set_index("date")["sales"].resample("W").sum().reset_index()
        )
    elif aggregation == "Monthly":
        plot_df = (
            filtered.set_index("date")["sales"].resample("ME").sum().reset_index()
        )
    else:
        plot_df = filtered[["date", "sales"]].copy()

    # ---------------------------------------------------------------
    # Filtered trend chart with optional rolling average
    # ---------------------------------------------------------------
    rolling_window = 7 if aggregation == "Daily" else 4 if aggregation == "Weekly" else 3
    rolling_unit = "day" if aggregation == "Daily" else "week" if aggregation == "Weekly" else "month"
    show_rolling = st.checkbox(
        f"Show rolling average ({rolling_window}-{rolling_unit})",
        value=True,
    )

    fig = px.line(plot_df, x="date", y="sales", title=f"Sales Trend ({aggregation})", labels={"date": "Date", "sales": "Sales"})
    if show_rolling and len(plot_df) > 1:
        plot_df["rolling"] = plot_df["sales"].rolling(window=rolling_window, min_periods=1).mean()
        fig.add_scatter(x=plot_df["date"], y=plot_df["rolling"], mode="lines", name="Rolling Avg")
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Time Patterns", "Store Analysis", "Item Analysis", "Distributions"]
    )

    # ---- Tab 1: Time Patterns ----
    with tab1:
        cal = filtered.copy()
        cal["day_name"] = cal["date"].dt.day_name()
        cal["month_name"] = cal["date"].dt.month_name()
        cal["quarter"] = cal["date"].dt.quarter
        cal["year"] = cal["date"].dt.year

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]

        c1, c2 = st.columns(2)
        with c1:
            dow = cal.groupby("day_name")["sales"].mean().reindex(day_order).reset_index()
            fig_dow = px.bar(dow, x="day_name", y="sales", title="Sales by Day of Week")
            fig_dow.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_dow, use_container_width=True)
        with c2:
            mon = cal.groupby("month_name")["sales"].mean().reindex(month_order).reset_index()
            fig_mon = px.bar(mon, x="month_name", y="sales", title="Sales by Month")
            fig_mon.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_mon, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            q = cal.groupby("quarter")["sales"].mean().reset_index()
            q["quarter"] = "Q" + q["quarter"].astype(str)
            fig_q = px.bar(q, x="quarter", y="sales", title="Quarter Comparison")
            fig_q.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_q, use_container_width=True)
        with c4:
            yoy = cal.groupby("year")["sales"].sum().reset_index()
            fig_yoy = px.line(yoy, x="year", y="sales", markers=True, title="Year over Year Sales")
            fig_yoy.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_yoy, use_container_width=True)

    # ---- Tab 2: Store Analysis ----
    with tab2:
        item_all_stores = df[df["item"] == item].copy()
        item_all_stores["date"] = pd.to_datetime(item_all_stores["date"])
        item_all_stores["store_label"] = "Store " + item_all_stores["store"].astype(str)

        fig_all_stores = px.line(
            item_all_stores,
            x="date",
            y="sales",
            color="store_label",
            title=f"All Stores Comparison (Item {item})",
        )
        fig_all_stores.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_all_stores, use_container_width=True)

        store_ranking = (
            item_all_stores.groupby("store_label")["sales"]
            .agg(["sum", "mean", "std"])
            .reset_index()
            .rename(columns={"sum": "Total Sales", "mean": "Avg Sales", "std": "Std Dev"})
            .sort_values("Total Sales", ascending=False)
        )
        st.dataframe(store_ranking, use_container_width=True, hide_index=True)

        best_store = store_ranking.iloc[0]
        worst_store = store_ranking.iloc[-1]
        c1, c2 = st.columns(2)
        c1.success(f"Best Performing: {best_store['store_label']} ({best_store['Total Sales']:.0f} total units)")
        c2.error(f"Worst Performing: {worst_store['store_label']} ({worst_store['Total Sales']:.0f} total units)")

    # ---- Tab 3: Item Analysis ----
    with tab3:
        store_all_items = df[df["store"] == store].copy()
        store_all_items["date"] = pd.to_datetime(store_all_items["date"])
        store_all_items["item_label"] = "Item " + store_all_items["item"].astype(str)

        top_items = (
            store_all_items.groupby("item_label")["sales"].sum().sort_values(ascending=False).head(8).index
        )
        fig_all_items = px.line(
            store_all_items[store_all_items["item_label"].isin(top_items)],
            x="date",
            y="sales",
            color="item_label",
            title=f"Top Items Comparison (Store {store})",
        )
        fig_all_items.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_all_items, use_container_width=True)

        item_ranking = (
            store_all_items.groupby("item_label")["sales"]
            .agg(["sum", "mean", "std"])
            .reset_index()
            .rename(columns={"sum": "Total Sales", "mean": "Avg Sales", "std": "Std Dev"})
            .sort_values("Total Sales", ascending=False)
        )
        st.dataframe(item_ranking, use_container_width=True, hide_index=True)

        most_volatile = item_ranking.sort_values("Std Dev", ascending=False).iloc[0]
        st.info(f"Most Volatile Item: {most_volatile['item_label']} (Std Dev: {most_volatile['Std Dev']:.1f})")

    # ---- Tab 4: Distributions ----
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(filtered, x="sales", nbins=40, title="Sales Distribution")
            fig_hist.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            box_df = df[df["item"] == item].copy()
            box_df["store_label"] = "Store " + box_df["store"].astype(str)
            fig_box = px.box(box_df, x="store_label", y="sales", title="Sales Distribution by Store")
            fig_box.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_box, use_container_width=True)

        heat_df = df[df["store"] == store].copy()
        heat_df["date"] = pd.to_datetime(heat_df["date"])
        heat_df["month"] = heat_df["date"].dt.month
        pivot = heat_df.pivot_table(index="item", columns="month", values="sales", aggfunc="mean")
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Month", y="Item", color="Avg Sales"),
            title=f"Store {store}: Item x Month Heatmap",
            aspect="auto",
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
