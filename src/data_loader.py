"""
Data Generator & Loader Module
===============================
Generates realistic synthetic driving data and loads CSV data into SQLite.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import (
    RAW_CSV_PATH, RAW_COLUMNS, BEHAVIOR_LABELS, RANDOM_STATE
)
from src.db_manager import init_database, insert_dataframe, clear_table


def generate_synthetic_data(num_trips: int = 500, records_per_trip: int = 10, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate realistic synthetic driving behavior data.

    Each trip has multiple timestamped records. Driving behavior labels
    (Safe, Normal, Aggressive) influence the distribution of sensor values.

    Args:
        num_trips: Number of unique trips to generate.
        records_per_trip: Number of records per trip.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with synthetic driving data.
    """
    np.random.seed(seed)
    rows = []
    base_time = datetime(2025, 1, 1, 6, 0, 0)

    # Distribution of behavior labels
    label_weights = [0.35, 0.40, 0.25]  # Safe, Normal, Aggressive
    labels = np.random.choice(BEHAVIOR_LABELS, size=num_trips, p=label_weights)

    for i in range(num_trips):
        trip_id = f"TRIP_{i+1:04d}"
        label = labels[i]
        trip_start = base_time + timedelta(hours=np.random.randint(0, 720))

        for j in range(records_per_trip):
            ts = trip_start + timedelta(seconds=j * 30 + np.random.randint(0, 10))

            if label == "Safe":
                speed = np.clip(np.random.normal(50, 10), 10, 90)
                accel = np.clip(np.random.normal(1.0, 0.5), 0, 3.0)
                brake = np.clip(np.random.normal(2.0, 1.0), 0, 5.0)
                steer = np.clip(np.random.normal(5, 3), 0, 20)
                rpm_val = np.clip(np.random.normal(2000, 300), 800, 3500)
                throttle = np.clip(np.random.normal(30, 8), 5, 55)
                fuel = np.clip(np.random.normal(6.0, 1.0), 3.0, 10.0)
                dist = np.clip(np.random.normal(2.0, 0.5), 0.5, 4.0)

            elif label == "Normal":
                speed = np.clip(np.random.normal(70, 15), 20, 120)
                accel = np.clip(np.random.normal(2.0, 1.0), 0, 5.0)
                brake = np.clip(np.random.normal(4.0, 1.5), 0, 8.0)
                steer = np.clip(np.random.normal(12, 6), 0, 40)
                rpm_val = np.clip(np.random.normal(3000, 500), 1000, 5000)
                throttle = np.clip(np.random.normal(50, 12), 10, 80)
                fuel = np.clip(np.random.normal(8.0, 1.5), 4.0, 14.0)
                dist = np.clip(np.random.normal(3.5, 1.0), 1.0, 7.0)

            else:  # Aggressive
                speed = np.clip(np.random.normal(100, 20), 40, 180)
                accel = np.clip(np.random.normal(4.0, 1.5), 0.5, 8.0)
                brake = np.clip(np.random.normal(7.0, 2.0), 1.0, 12.0)
                steer = np.clip(np.random.normal(25, 10), 0, 70)
                rpm_val = np.clip(np.random.normal(4500, 700), 2000, 7000)
                throttle = np.clip(np.random.normal(75, 10), 30, 100)
                fuel = np.clip(np.random.normal(12.0, 2.0), 6.0, 20.0)
                dist = np.clip(np.random.normal(5.0, 1.5), 1.5, 10.0)

            # Inject occasional noise / anomalies (~2% chance)
            if np.random.random() < 0.02:
                speed = np.random.uniform(0, 200)

            rows.append({
                "trip_id": trip_id,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "speed_kmh": round(speed, 2),
                "acceleration_ms2": round(accel, 2),
                "braking_force": round(brake, 2),
                "steering_angle_deg": round(steer, 2),
                "rpm": round(rpm_val, 2),
                "throttle_pct": round(throttle, 2),
                "fuel_consumption": round(fuel, 2),
                "distance_km": round(dist, 2),
                "behavior_label": label
            })

    df = pd.DataFrame(rows)

    # Inject some missing values (~1.5% of cells in numeric columns)
    numeric_cols = ["speed_kmh", "acceleration_ms2", "braking_force", "rpm", "throttle_pct"]
    for col in numeric_cols:
        mask = np.random.random(len(df)) < 0.015
        df.loc[mask, col] = np.nan

    # Inject some duplicates (~1%)
    n_dupes = int(len(df) * 0.01)
    dupes = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


def save_csv(df: pd.DataFrame, path: str = RAW_CSV_PATH):
    """Save a DataFrame as CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved {len(df)} records to {path}")


def load_csv_to_sqlite(csv_path: str = RAW_CSV_PATH, replace: bool = True):
    """
    Load CSV data into the SQLite driving_data table.

    Args:
        csv_path: Path to the CSV file.
        replace: If True, clears existing data before inserting.
    """
    init_database()
    df = pd.read_csv(csv_path)

    if replace:
        clear_table("driving_data")

    insert_dataframe(df, "driving_data")
    print(f"✅ Loaded {len(df)} records into SQLite")


def generate_and_load(num_trips: int = 500):
    """Generate synthetic data, save as CSV, and load into SQLite."""
    print("🔄 Generating synthetic driving data...")
    df = generate_synthetic_data(num_trips=num_trips)
    save_csv(df)
    load_csv_to_sqlite()
    return df


if __name__ == "__main__":
    generate_and_load()
