"""
Anomaly Detection page: run Isolation Forest based anomaly detection
and visualize/explore the results.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import PLOTLY_LAYOUT_DEFAULTS, COLOR_HIGH, COLOR_MEDIUM, COLOR_LOW, COLOR_NORMAL
from src.anomaly_detector import detect_anomalies


def _severity_badge(severity):
    colors = {"HIGH": COLOR_HIGH, "MEDIUM": COLOR_MEDIUM, "LOW": COLOR_LOW, "NONE": "#334155"}
    color = colors.get(severity, "#334155")
    return f"<span style='background:{color};padding:2px 8px;border-radius:8px;color:white;font-size:0.8em;'>{severity}</span>"


def show(store, item):
    df = st.session_state.get("full_data")
    if df is None or df.empty:
        st.warning("No data available. Please reload the app.")
        return

    st.title("Anomaly Detection")
    st.caption(f"Detecting unusual sales patterns for Store {store}, Item {item}.")

    with st.expander("Detection Settings", expanded=True):
        contamination = st.slider(
            "Contamination (expected anomaly rate)", min_value=0.01, max_value=0.15, value=0.05, step=0.01
        )
        st.caption(
            "Contamination tells the Isolation Forest algorithm roughly what "
            "fraction of days are expected to be anomalous. A higher value "
            "flags more points as anomalies."
        )

    run_detection = st.button("Run Detection", type="primary")

    if run_detection:
        st.session_state["anomaly_ran"] = True
        st.session_state["anomaly_config"] = {"store": store, "item": item, "contamination": contamination}

    if not st.session_state.get("anomaly_ran"):
        st.info("Configure the contamination rate and click **Run Detection** to begin.")
        return

    cfg = st.session_state["anomaly_config"]

    with st.spinner("Running Isolation Forest..."):
        result = detect_anomalies(df, cfg["store"], cfg["item"], cfg["contamination"])

    if result.empty:
        st.info("Not enough data to run anomaly detection.")
        return

    anomalies = result[result["is_anomaly"]]

    st.divider()

    # ---------------------------------------------------------------
    # Summary cards
    # ---------------------------------------------------------------
    total_anomalies = len(anomalies)
    high_count = (anomalies["severity"] == "HIGH").sum()
    medium_count = (anomalies["severity"] == "MEDIUM").sum()
    low_count = (anomalies["severity"] == "LOW").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Anomalies", total_anomalies)
    c2.metric("High Severity", high_count)
    c3.metric("Medium Severity", medium_count)
    c4.metric("Low Severity", low_count)

    # ---------------------------------------------------------------
    # Main chart
    # ---------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result["date"], y=result["sales"], mode="lines+markers",
            name="Sales", line=dict(color=COLOR_NORMAL, width=1.5),
            marker=dict(size=4, color=COLOR_NORMAL),
            hovertemplate="Date: %{x}<br>Sales: %{y}<extra></extra>",
        )
    )

    for severity, color, symbol in [
        ("HIGH", COLOR_HIGH, "triangle-up"),
        ("MEDIUM", COLOR_MEDIUM, "circle"),
        ("LOW", COLOR_LOW, "square"),
    ]:
        sev_df = anomalies[anomalies["severity"] == severity]
        if not sev_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sev_df["date"], y=sev_df["sales"], mode="markers",
                    name=f"{severity} Anomaly",
                    marker=dict(size=11, color=color, symbol=symbol, line=dict(width=1, color="white")),
                    customdata=sev_df[["expected_sales", "deviation_pct"]],
                    hovertemplate=(
                        "Date: %{x}<br>Sales: %{y}<br>Expected: %{customdata[0]:.1f}"
                        "<br>Deviation: %{customdata[1]:.1f}%<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"Anomaly Detection: Store {store}, Item {item}",
        xaxis_title="Date",
        yaxis_title="Sales",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------------
    # Details tabs
    # ---------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Anomaly Table", "Distribution Charts", "Insights"])

    with tab1:
        if anomalies.empty:
            st.info("No anomalies detected with the current settings.")
        else:
            table = anomalies.copy()
            table["Deviation"] = table["deviation_pct"].round(1).astype(str) + "%"
            table["Severity"] = table["severity"].apply(_severity_badge)
            display_table = table[["date", "sales", "expected_sales", "Deviation", "Severity"]].rename(
                columns={"date": "Date", "sales": "Sales", "expected_sales": "Expected"}
            )
            display_table["Expected"] = display_table["Expected"].round(1)
            display_table["Date"] = pd.to_datetime(display_table["Date"]).dt.strftime("%Y-%m-%d")
            st.write(display_table.to_html(escape=False, index=False), unsafe_allow_html=True)

            csv = anomalies.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Anomalies CSV",
                data=csv,
                file_name=f"anomalies_store{store}_item{item}.csv",
                mime="text/csv",
            )

    with tab2:
        if anomalies.empty:
            st.info("No anomalies to visualize.")
        else:
            anomalies_calendar = anomalies.copy()
            anomalies_calendar["date"] = pd.to_datetime(anomalies_calendar["date"])
            anomalies_calendar["month_name"] = anomalies_calendar["date"].dt.month_name()
            anomalies_calendar["day_name"] = anomalies_calendar["date"].dt.day_name()

            c1, c2 = st.columns(2)
            with c1:
                by_month = anomalies_calendar.groupby("month_name").size().reset_index(name="count")
                fig_month = px.bar(by_month, x="month_name", y="count", title="Anomalies by Month")
                fig_month.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                st.plotly_chart(fig_month, use_container_width=True)
            with c2:
                by_dow = anomalies_calendar.groupby("day_name").size().reset_index(name="count")
                fig_dow = px.bar(by_dow, x="day_name", y="count", title="Anomalies by Day of Week")
                fig_dow.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
                st.plotly_chart(fig_dow, use_container_width=True)

            severity_counts = anomalies["severity"].value_counts().reset_index()
            severity_counts.columns = ["severity", "count"]
            fig_pie = px.pie(
                severity_counts, names="severity", values="count", title="Severity Distribution",
                color="severity",
                color_discrete_map={"HIGH": COLOR_HIGH, "MEDIUM": COLOR_MEDIUM, "LOW": COLOR_LOW},
            )
            fig_pie.update_layout(**PLOTLY_LAYOUT_DEFAULTS)
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        if anomalies.empty:
            st.info("No anomalies detected — insights unavailable.")
        else:
            anomalies_calendar = anomalies.copy()
            anomalies_calendar["date"] = pd.to_datetime(anomalies_calendar["date"])
            anomalies_calendar["month_name"] = anomalies_calendar["date"].dt.month_name()

            top_month = anomalies_calendar["month_name"].value_counts().idxmax()
            top_month_count = anomalies_calendar["month_name"].value_counts().max()

            anomaly_rate = len(anomalies) / len(result) * 100
            avg_deviation = anomalies["deviation_pct"].mean()

            st.markdown(
                f"""
                - **Anomaly rate:** {anomaly_rate:.1f}% of days for Store {store}, Item {item} were flagged as anomalous.
                - **Most anomalous month:** {top_month} ({top_month_count} anomalies).
                - **Average deviation:** anomalous days deviated from expected sales by {avg_deviation:.1f}% on average.
                - **Severity mix:** {high_count} high, {medium_count} medium, and {low_count} low severity anomalies were found.
                """
            )

            if high_count > 0:
                st.warning(
                    f"{high_count} high-severity anomalies were detected — these represent the largest "
                    "deviations from expected demand and may warrant investigation (e.g. promotions, "
                    "stockouts, data entry errors, or supply disruptions)."
                )
