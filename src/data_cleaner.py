"""
Data Cleaner Module
====================
Handles missing values, outlier detection/removal, duplicate removal,
data type validation, and generates a cleaning summary report.
"""

import pandas as pd
import numpy as np
from src.config import NUMERIC_FEATURES, DB_PATH
from src.db_manager import (
    query_to_dataframe, insert_dataframe, clear_table,
    init_database, log_cleaning_action
)


def load_raw_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load raw driving data from SQLite."""
    return query_to_dataframe("SELECT * FROM driving_data", db_path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - Numeric columns: impute with column median (robust to outliers).
    - Categorical columns: fill with mode.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Fill categorical
    if "behavior_label" in df.columns and df["behavior_label"].isnull().any():
        df["behavior_label"] = df["behavior_label"].fillna(df["behavior_label"].mode()[0])

    missing_after = df.isnull().sum().sum()
    log_cleaning_action(
        "missing_values",
        str(missing_before),
        str(missing_after),
        "Imputed numeric with median, categorical with mode"
    )
    return df


def detect_and_remove_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect and remove outliers using the IQR method.

    Args:
        df: Input DataFrame.
        method: Detection method ('iqr' or 'zscore').
        threshold: IQR multiplier (default 1.5) or Z-score threshold.

    Returns:
        DataFrame with outliers removed.
    """
    df = df.copy()
    rows_before = len(df)
    outlier_mask = pd.Series([False] * len(df), index=df.index)

    cols_to_check = ["speed_kmh", "acceleration_ms2", "braking_force", "rpm"]

    if method == "iqr":
        for col in cols_to_check:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_outliers = (df[col] < lower) | (df[col] > upper)
                outlier_mask = outlier_mask | col_outliers
    elif method == "zscore":
        for col in cols_to_check:
            if col in df.columns:
                z = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = outlier_mask | (z > threshold)

    df = df[~outlier_mask].reset_index(drop=True)
    rows_after = len(df)

    log_cleaning_action(
        "outlier_removal",
        str(rows_before),
        str(rows_after),
        f"Removed {rows_before - rows_after} outlier rows via {method.upper()} method"
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on trip_id + timestamp."""
    df = df.copy()
    rows_before = len(df)

    df = df.drop_duplicates(
        subset=["trip_id", "timestamp"],
        keep="first"
    )
    df = df.reset_index(drop=True)
    rows_after = len(df)

    log_cleaning_action(
        "duplicate_removal",
        str(rows_before),
        str(rows_after),
        f"Removed {rows_before - rows_after} duplicate rows"
    )
    return df


def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce data types for all columns."""
    df = df.copy()

    numeric_cols = NUMERIC_FEATURES
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "behavior_label" in df.columns:
        df["behavior_label"] = df["behavior_label"].astype(str)

    log_cleaning_action("type_validation", "mixed", "validated", "Coerced all columns to correct types")
    return df


def generate_cleaning_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Generate a summary report comparing before/after cleaning.

    Returns:
        dict with cleaning statistics.
    """
    report = {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed": len(df_before) - len(df_after),
        "removal_pct": round((len(df_before) - len(df_after)) / len(df_before) * 100, 2),
        "nulls_before": int(df_before.isnull().sum().sum()),
        "nulls_after": int(df_after.isnull().sum().sum()),
        "duplicates_before": int(df_before.duplicated(subset=["trip_id", "timestamp"]).sum()),
        "duplicates_after": int(df_after.duplicated(subset=["trip_id", "timestamp"]).sum()),
        "columns": len(df_after.columns),
        "unique_trips": int(df_after["trip_id"].nunique()) if "trip_id" in df_after.columns else 0,
        "label_distribution": df_after["behavior_label"].value_counts().to_dict() if "behavior_label" in df_after.columns else {}
    }

    print("\n📊 Cleaning Report")
    print("=" * 50)
    for key, val in report.items():
        print(f"  {key:25s}: {val}")
    print("=" * 50)

    return report


def clean_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Execute the full data cleaning pipeline:
    1. Load raw data
    2. Validate data types
    3. Remove duplicates
    4. Handle missing values
    5. Remove outliers
    6. Generate report

    Returns:
        Cleaned pd.DataFrame.
    """
    print("🧹 Starting data cleaning pipeline...")
    init_database(db_path)

    # Load raw data
    df_raw = load_raw_data(db_path)
    print(f"  Loaded {len(df_raw)} raw records")

    # Pipeline steps
    df = validate_data_types(df_raw)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = detect_and_remove_outliers(df)

    # Generate report
    report = generate_cleaning_report(df_raw, df)

    # Save cleaned data back (replace driving_data with clean version)
    clear_table("driving_data", db_path)
    insert_dataframe(df, "driving_data", db_path)
    print(f"✅ Cleaned data saved: {len(df)} records")

    return df


if __name__ == "__main__":
    clean_data()
