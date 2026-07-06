"""
Retail Demand Forecasting and Inventory Optimization System
Main Streamlit entry point.

Data can come from a direct CSV upload in the sidebar, or from a local
file at RAW_DATA_PATH as a fallback. If a file's columns don't match
date/store/item/sales automatically, the user is prompted to map them
manually.
"""

import os

import streamlit as st

from src.data_loader import (
    load_raw_data,
    load_uploaded_data,
    peek_uploaded_file,
    validate_raw_data,
    ColumnMappingRequired,
)
from src.preprocessor import preprocess
from src.config import RAW_DATA_PATH
from src.theme import apply_theme

from app_pages import overview, eda_explorer, forecasting, anomaly_detection, inventory_optimizer as inventory_page


st.set_page_config(
    page_title="Demand Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.cache_data(show_spinner=False)
def _process_data(raw_df):
    report = validate_raw_data(raw_df)
    processed_df = preprocess(raw_df)
    return processed_df, report


@st.cache_data(show_spinner=False)
def _load_from_bytes(file_bytes: bytes, file_name: str, manual_mapping=None):
    return load_uploaded_data(file_bytes, file_name, manual_mapping=manual_mapping)


@st.cache_data(show_spinner=False)
def _load_from_path(path: str, manual_mapping=None):
    return load_raw_data(path, manual_mapping=manual_mapping)


def _render_column_mapping_ui(mapping_error: ColumnMappingRequired, source_label: str):
    """
    Shown when a file was parsed but its columns couldn't be
    auto-matched to date/store/item/sales. Lets the user pick the
    right column for each field, then reruns with that mapping applied.
    """
    st.warning(
        f"Couldn't automatically match all required columns in **{source_label}**. "
        f"Please map them below — this only needs to happen once per file."
    )

    columns = list(mapping_error.df.columns)
    placeholder = "-- select a column --"
    options = [placeholder] + [str(c) for c in columns]

    chosen = {}
    cols_ui = st.columns(4)
    for i, target in enumerate(["date", "store", "item", "sales"]):
        guess = mapping_error.mapping.get(target)
        default_index = options.index(str(guess)) if guess is not None and str(guess) in options else 0
        with cols_ui[i]:
            chosen[target] = st.selectbox(
                f"Column for '{target}'", options, index=default_index, key=f"colmap_{target}"
            )

    with st.expander("Preview uploaded data (first 20 rows)"):
        st.dataframe(mapping_error.df.head(20), use_container_width=True)

    if st.button("Confirm column mapping", type="primary"):
        if any(v == placeholder for v in chosen.values()):
            st.error("Please select a column for every field before continuing.")
        else:
            st.session_state["_manual_mapping"] = chosen
            st.rerun()


def _reset_mapping_state_if_new_file(current_key: str):
    """Clear any saved mapping choice when a different file is loaded."""
    if st.session_state.get("_active_source_key") != current_key:
        st.session_state["_active_source_key"] = current_key
        st.session_state.pop("_manual_mapping", None)


def main():
    with st.sidebar:
        st.title("Demand Forecasting")
        st.caption("A complete retail demand forecasting and inventory optimization toolkit.")
        st.divider()

        st.subheader("Data Source")
        uploaded_file = st.file_uploader(
            "Upload sales data",
            type=["csv"],
            help=(
                "Supported format: CSV only. Column names don't need to match "
                "exactly — you'll be asked to map them if they can't be "
                "auto-detected."
            ),
        )
        
        

    df = None
    report = None
    data_source_label = None

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        _reset_mapping_state_if_new_file(f"upload::{filename}::{len(file_bytes)}")

        try:
            peek_uploaded_file(file_bytes, filename)
        except Exception as exc:
            st.sidebar.error(f"Couldn't read '{filename}':\n\n{exc}")
            st.stop()

        manual_mapping = st.session_state.get("_manual_mapping")

        try:
            with st.spinner(f"Loading {filename}..."):
                raw_df = _load_from_bytes(file_bytes, filename, manual_mapping)
                df, report = _process_data(raw_df)
            data_source_label = f"Uploaded: {filename}"
        except ColumnMappingRequired as mapping_error:
            _render_column_mapping_ui(mapping_error, filename)
            st.stop()
        except Exception as exc:
            st.error(f"Failed to load '{filename}':\n\n{exc}")
            st.stop()

    elif os.path.exists(RAW_DATA_PATH):
        _reset_mapping_state_if_new_file(f"path::{RAW_DATA_PATH}")
        manual_mapping = st.session_state.get("_manual_mapping")
        try:
            with st.spinner(f"Loading data from {RAW_DATA_PATH}..."):
                raw_df = _load_from_path(RAW_DATA_PATH, manual_mapping)
                df, report = _process_data(raw_df)
            data_source_label = f"Local file: {RAW_DATA_PATH}"
        except ColumnMappingRequired as mapping_error:
            _render_column_mapping_ui(mapping_error, RAW_DATA_PATH)
            st.stop()
        except Exception as exc:
            st.error(f"Failed to load '{RAW_DATA_PATH}':\n\n{exc}")
            st.stop()
    else:
        st.error(
            "No data loaded yet. Upload a CSV file in the sidebar, or place one "
            f"at `{RAW_DATA_PATH}` and reload."
        )
        st.stop()

    if df is None or df.empty:
        st.error("The loaded dataset is empty. Please check your file.")
        st.stop()

    with st.sidebar:
        st.markdown(f"**Source:** {data_source_label}")
        st.caption(
            f"{len(df):,} rows · {df['store'].nunique()} stores · {df['item'].nunique()} items"
        )
        
        
        total_missing = sum(report.get("missing_values", {}).values())
        st.caption(
            f"Missing: {total_missing} · "
            f"Duplicates: {report.get('duplicate_rows', 0)} · "
            f"Negative sales: {report.get('negative_sales', 0)}"
        )

        st.divider()

        store_options = sorted(df["store"].unique().tolist())
        item_options = sorted(df["item"].unique().tolist())

        store = st.selectbox("Select Store", store_options, key="selected_store")
        item = st.selectbox("Select Item", item_options, key="selected_item")

        st.divider()

        page = st.radio(
            "Navigate",
            [
                "Overview",
                "EDA Explorer",
                "Forecasting",
                "Anomaly Detection",
                "Inventory Optimizer",
            ],
        )

        st.divider()

    st.session_state["full_data"] = df
    st.session_state["data_source"] = data_source_label

    if page == "Overview":
        overview.show()
    elif page == "EDA Explorer":
        eda_explorer.show(store, item)
    elif page == "Forecasting":
        forecasting.show(store, item)
    elif page == "Anomaly Detection":
        anomaly_detection.show(store, item)
    elif page == "Inventory Optimizer":
        inventory_page.show(store, item)


if __name__ == "__main__":
    main()
