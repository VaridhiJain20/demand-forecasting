"""
Flexible data loading, validation, and continuity utilities.

Supports loading sales data from CSV files and from files whose column
names don't exactly match "date, store, item, sales" via automatic
synonym matching or an explicit manual mapping.
"""

import io
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

REQUIRED_COLUMNS = ["date", "store", "item", "sales"]

# File extensions we know how to parse into a dataframe.
SUPPORTED_TABULAR_EXTENSIONS = {"csv"}

# Common alternate names people use for each required column, used to
# auto-detect a mapping when a file's headers don't match exactly.
COLUMN_SYNONYMS = {
    "date": [
        "date", "day", "order_date", "sale_date", "sales_date",
        "transaction_date", "ds", "period", "week", "week_date", "timestamp",
    ],
    "store": [
        "store", "store_id", "storeid", "shop", "shop_id", "location",
        "location_id", "branch", "branch_id", "outlet", "outlet_id", "site",
    ],
    "item": [
        "item", "item_id", "itemid", "product", "product_id", "productid",
        "sku", "sku_id", "article", "article_id", "product_name",
    ],
    "sales": [
        "sales", "sale", "sales_qty", "salesqty", "quantity", "qty",
        "units", "units_sold", "unitssold", "demand", "y", "amount",
        "revenue", "value", "sold",
    ],
}

# Explicitly unsupported / unsafe formats, with a specific message for each.
_EXPLICITLY_BLOCKED_EXTENSIONS = {
    "pkl": "Pickle files can execute arbitrary code when loaded and are not accepted for security reasons.",
    "pickle": "Pickle files can execute arbitrary code when loaded and are not accepted for security reasons.",
}


class ColumnMappingRequired(Exception):
    """
    Raised when a file was parsed successfully but its columns could
    not be automatically matched to date/store/item/sales. Carries the
    raw dataframe and the best-guess partial mapping so the caller
    (e.g. the Streamlit UI) can ask the user to complete the mapping.
    """

    def __init__(self, df: pd.DataFrame, mapping: dict, message: str):
        super().__init__(message)
        self.df = df
        self.mapping = mapping


def _normalize_col(col) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def suggest_column_mapping(columns) -> dict:
    """
    Guess which of the given raw column names maps to each of
    date/store/item/sales, based on common naming synonyms.

    Parameters
    ----------
    columns : iterable of str
        The raw dataframe's column names.

    Returns
    -------
    dict
        {"date": original_col_or_None, "store": ..., "item": ..., "sales": ...}
    """
    normalized_to_original = {_normalize_col(c): c for c in columns}
    used = set()
    mapping = {}

    for target, synonyms in COLUMN_SYNONYMS.items():
        found = None
        for syn in synonyms:
            if syn in normalized_to_original and normalized_to_original[syn] not in used:
                found = normalized_to_original[syn]
                break
        mapping[target] = found
        if found:
            used.add(found)

    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Rename df's columns according to mapping {target: source_column}.
    """
    rename_map = {source: target for target, source in mapping.items() if source is not None}
    return df.rename(columns=rename_map)


def peek_uploaded_file(file_bytes: bytes, filename: str) -> dict:
    """
    Inspect a file's bytes before fully parsing it and detect whether it
    is a supported CSV upload.

    Returns
    -------
    dict: {"kind": "tabular", "ext": str}

    Raises
    ------
    ValueError
        If the file extension isn't supported.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in _EXPLICITLY_BLOCKED_EXTENSIONS:
        raise ValueError(f"'.{ext}' files are not supported: {_EXPLICITLY_BLOCKED_EXTENSIONS[ext]}")

    if ext != "csv":
        raise ValueError(
            f"Unsupported file type '.{ext}' in '{filename}'. Only CSV files are supported."
        )

    return {"kind": "tabular", "ext": ext}


def _read_dataframe_from_buffer(buf: io.BytesIO, ext: str, filename: str) -> pd.DataFrame:
    """
    Parse a file-like buffer into a raw dataframe based on its
    (effective) extension. No column renaming/validation happens here.
    """
    try:
        if ext == "csv":
            return pd.read_csv(buf)
        else:
            raise ValueError(
                f"Unsupported or unrecognized file type '.{ext}' in '{filename}'. "
                "Only CSV files are supported."
            )
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Failed to parse %s", filename)
        raise ValueError(f"Could not parse '{filename}' as a .{ext} file: {exc}") from exc


def load_any_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse raw bytes of a CSV file into a dataframe.

    This does NOT rename or validate columns — call
    `load_uploaded_data` for the full pipeline including column
    mapping and cleaning.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in _EXPLICITLY_BLOCKED_EXTENSIONS:
        raise ValueError(f"'.{ext}' files are not supported: {_EXPLICITLY_BLOCKED_EXTENSIONS[ext]}")

    if ext != "csv":
        raise ValueError(
            f"Unsupported file type '.{ext}' in '{filename}'. Only CSV files are supported."
        )

    buf = io.BytesIO(file_bytes)
    return _read_dataframe_from_buffer(buf, ext, filename)


def _finalize_columns(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Final dtype coercion + sorting once columns are already named
    exactly date/store/item/sales.
    """
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as exc:
        logger.exception("Failed to parse the 'date' column")
        raise ValueError(f"Could not parse 'date' column as dates: {exc}") from exc

    try:
        df["store"] = df["store"].astype(str).str.strip()
        # Prefer integer store/item IDs when possible, but fall back to
        # string labels (e.g. "Store_A") rather than failing outright.
        if df["store"].str.match(r"^-?\d+$").all():
            df["store"] = df["store"].astype(int)
    except Exception:
        pass

    try:
        df["item"] = df["item"].astype(str).str.strip()
        if df["item"].str.match(r"^-?\d+$").all():
            df["item"] = df["item"].astype(int)
    except Exception:
        pass

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    n_bad_sales = int(df["sales"].isna().sum())
    if n_bad_sales > 0:
        logger.warning("%d rows had non-numeric sales values; filling with 0", n_bad_sales)
        df["sales"] = df["sales"].fillna(0)

    df = df.sort_values(["date", "store", "item"]).reset_index(drop=True)

    logger.info(
        "Loaded %d rows from '%s': %d stores, %d items, date range %s to %s",
        len(df), source_label, df["store"].nunique(), df["item"].nunique(),
        df["date"].min().date(), df["date"].max().date(),
    )

    return df


def load_uploaded_data(file_bytes: bytes, filename: str, manual_mapping: dict = None) -> pd.DataFrame:
    """
    Full pipeline: parse a CSV file's bytes, map its columns to
    date/store/item/sales (automatically or via an explicit manual
    mapping), and clean/validate the result.

    Parameters
    ----------
    file_bytes : bytes
    filename : str
    manual_mapping : dict, optional
        {"date": col, "store": col, "item": col, "sales": col} — if
        provided, used instead of automatic synonym matching.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ColumnMappingRequired
        If columns can't be auto-matched and no manual mapping was given.
    ValueError
        If the file format is unsupported or parsing/cleaning fails.
    """
    raw_df = load_any_file(file_bytes, filename)

    if raw_df.empty or len(raw_df.columns) == 0:
        raise ValueError(f"'{filename}' parsed successfully but contains no data.")

    mapping = manual_mapping if manual_mapping else suggest_column_mapping(raw_df.columns)

    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ColumnMappingRequired(
            df=raw_df,
            mapping=mapping,
            message=(
                f"Could not automatically detect a column for: {missing} in '{filename}'. "
                f"Available columns: {list(raw_df.columns)}."
            ),
        )

    mapped_df = apply_column_mapping(raw_df, mapping)
    # Keep only the four columns we need, in case of extra columns.
    mapped_df = mapped_df[REQUIRED_COLUMNS].copy()

    return _finalize_columns(mapped_df, source_label=filename)


def load_raw_data(path: str, manual_mapping: dict = None) -> pd.DataFrame:
    """
    Load and clean sales data from a local CSV file path. Supports the
    same flexible column mapping as `load_uploaded_data` (it delegates
    to it internally).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at `path`.
    """
    if not os.path.exists(path):
        logger.error("Data file not found at %s", path)
        raise FileNotFoundError(f"Could not find a data file at '{path}'.")

    with open(path, "rb") as f:
        file_bytes = f.read()

    return load_uploaded_data(file_bytes, os.path.basename(path), manual_mapping=manual_mapping)


def validate_raw_data(df: pd.DataFrame) -> dict:
    """
    Run a set of data-quality checks on the cleaned dataframe.

    Returns
    -------
    dict
        {
          "missing_values": {col: count, ...},
          "duplicate_rows": int,
          "date_range": (min_date, max_date),
          "n_stores": int,
          "n_items": int,
          "negative_sales": int,
          "global_date_gaps": int,
          "store_item_gap_summary": {
              "combinations_with_gaps": int,
              "combinations_checked": int,
              "total_combinations": int,
          },
        }
    """
    report = {}

    report["missing_values"] = {col: int(df[col].isna().sum()) for col in df.columns}
    report["duplicate_rows"] = int(df.duplicated(subset=["date", "store", "item"]).sum())
    report["date_range"] = (df["date"].min(), df["date"].max())
    report["n_stores"] = int(df["store"].nunique())
    report["n_items"] = int(df["item"].nunique())
    report["negative_sales"] = int((df["sales"] < 0).sum())

    full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    dates_present = pd.to_datetime(df["date"].unique())
    report["global_date_gaps"] = int(len(full_range) - len(dates_present))

    combos = df[["store", "item"]].drop_duplicates()
    sample_combos = combos if len(combos) <= 500 else combos.sample(500, random_state=42)

    gap_count = 0
    for _, row in sample_combos.iterrows():
        sub_dates = df[(df["store"] == row["store"]) & (df["item"] == row["item"])]["date"]
        if len(sub_dates) == 0:
            continue
        expected_days = (sub_dates.max() - sub_dates.min()).days + 1
        if len(sub_dates) < expected_days:
            gap_count += 1

    report["store_item_gap_summary"] = {
        "combinations_with_gaps": gap_count,
        "combinations_checked": len(sample_combos),
        "total_combinations": len(combos),
    }

    return report


def make_continuous_daily_series(df: pd.DataFrame, store, item, fill_method: str = "zero") -> pd.DataFrame:
    """
    Build a continuous daily sales series for a single store-item pair,
    filling any missing calendar days between its first and last
    observed date.

    Parameters
    ----------
    df : pd.DataFrame
    store, item : the store/item identifiers to filter on (int or str)
    fill_method : str
        "zero" (default) fills missing days with 0 sales.
        "ffill" forward-fills the last known value.
        "interpolate" linearly interpolates missing values.

    Returns
    -------
    pd.DataFrame
        Columns: date, store, item, sales.
    """
    subset = df[(df["store"] == store) & (df["item"] == item)].copy()

    if subset.empty:
        return pd.DataFrame(columns=["date", "store", "item", "sales"])

    subset["date"] = pd.to_datetime(subset["date"])
    subset = subset.sort_values("date").drop_duplicates(subset="date", keep="last")
    subset = subset.set_index("date")

    full_range = pd.date_range(start=subset.index.min(), end=subset.index.max(), freq="D")
    subset = subset.reindex(full_range)
    subset.index.name = "date"

    if fill_method == "ffill":
        subset["sales"] = subset["sales"].ffill().fillna(0)
    elif fill_method == "interpolate":
        subset["sales"] = subset["sales"].interpolate(method="linear").ffill().bfill()
    else:
        subset["sales"] = subset["sales"].fillna(0)

    subset["store"] = store
    subset["item"] = item

    subset = subset.reset_index()
    return subset[["date", "store", "item", "sales"]]
