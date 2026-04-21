"""
Feature Engineering Module
===========================
Computes statistical aggregates, derived features, rolling-window features,
and applies feature scaling for ML model consumption.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from src.config import (
    DB_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH,
    HARSH_BRAKING_THRESHOLD, RAPID_ACCELERATION_THRESHOLD,
    ENGINEERED_FEATURES
)
from src.db_manager import (
    query_to_dataframe, insert_dataframe, clear_table, init_database
)


def compute_trip_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical aggregates per trip.
    Groups by trip_id and calculates mean, max, min, std for sensor columns.

    Args:
        df: Cleaned driving data DataFrame.

    Returns:
        DataFrame with one row per trip and aggregated features.
    """
    grouped = df.groupby("trip_id")

    features = pd.DataFrame()
    features["trip_id"] = grouped["trip_id"].first().values

    # Speed features
    features["avg_speed"] = grouped["speed_kmh"].mean().values
    features["max_speed"] = grouped["speed_kmh"].max().values
    features["speed_std"] = grouped["speed_kmh"].std().fillna(0).values

    # Acceleration features
    features["avg_acceleration"] = grouped["acceleration_ms2"].mean().values
    features["max_acceleration"] = grouped["acceleration_ms2"].max().values
    features["acceleration_std"] = grouped["acceleration_ms2"].std().fillna(0).values

    # Braking features
    features["avg_braking"] = grouped["braking_force"].mean().values
    features["max_braking"] = grouped["braking_force"].max().values
    features["braking_std"] = grouped["braking_force"].std().fillna(0).values

    # Steering features
    features["avg_steering_angle"] = grouped["steering_angle_deg"].mean().values
    features["max_steering_angle"] = grouped["steering_angle_deg"].max().values
    features["steering_variability"] = grouped["steering_angle_deg"].std().fillna(0).values

    # RPM features
    features["avg_rpm"] = grouped["rpm"].mean().values
    features["max_rpm"] = grouped["rpm"].max().values

    # Throttle features
    features["avg_throttle"] = grouped["throttle_pct"].mean().values
    features["throttle_variability"] = grouped["throttle_pct"].std().fillna(0).values

    # Event counts
    features["harsh_braking_count"] = grouped.apply(
        lambda g: (g["braking_force"] > HARSH_BRAKING_THRESHOLD).sum()
    ).values
    features["rapid_acceleration_count"] = grouped.apply(
        lambda g: (g["acceleration_ms2"] > RAPID_ACCELERATION_THRESHOLD).sum()
    ).values

    # Fuel & distance
    features["avg_fuel_consumption"] = grouped["fuel_consumption"].mean().values
    features["total_distance"] = grouped["distance_km"].sum().values

    # Behavior label (one per trip)
    features["behavior_label"] = grouped["behavior_label"].first().values

    return features


def compute_correlation_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and return the Pearson correlation matrix for numeric features.

    Args:
        features_df: Feature DataFrame.

    Returns:
        Correlation matrix as DataFrame.
    """
    numeric_cols = [c for c in ENGINEERED_FEATURES if c in features_df.columns]
    return features_df[numeric_cols].corr()


def apply_scaling(features_df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Apply StandardScaler to numeric features.

    Args:
        features_df: Feature DataFrame.
        fit: If True, fits a new scaler and saves it. If False, loads existing scaler.

    Returns:
        DataFrame with scaled numeric features (trip_id and label preserved).
    """
    numeric_cols = [c for c in ENGINEERED_FEATURES if c in features_df.columns]
    df = features_df.copy()

    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(numeric_cols, FEATURE_COLUMNS_PATH)
        print(f"  💾 Scaler saved to {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def engineer_features(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline:
    1. Load cleaned data from SQLite
    2. Compute trip-level aggregates
    3. Apply scaling
    4. Save features to SQLite

    Returns:
        Unscaled feature DataFrame (scaled version saved separately).
    """
    print("⚙️  Starting feature engineering pipeline...")
    init_database(db_path)

    # Load cleaned data
    df = query_to_dataframe("SELECT * FROM driving_data", db_path)
    print(f"  Loaded {len(df)} cleaned records ({df['trip_id'].nunique()} trips)")

    # Compute trip-level features
    features = compute_trip_aggregates(df)
    print(f"  Computed {len(features.columns) - 2} features for {len(features)} trips")

    # Save unscaled features to SQLite
    clear_table("features", db_path)
    insert_dataframe(features, "features", db_path)
    print(f"  ✅ Features saved to SQLite ({len(features)} trips)")

    # Apply scaling (saves scaler and scaled data)
    scaled_features = apply_scaling(features, fit=True)

    # Compute correlation matrix for later use
    corr_matrix = compute_correlation_matrix(features)
    print(f"  📊 Correlation matrix computed ({len(corr_matrix)}x{len(corr_matrix)})")

    return features


if __name__ == "__main__":
    engineer_features()
