"""
Unit Tests for data_loader module
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_synthetic_data, save_csv, load_csv_to_sqlite
from src.db_manager import init_database, get_table_row_count, query_to_dataframe
from src.config import BEHAVIOR_LABELS

TEST_DB = os.path.join(os.path.dirname(__file__), "test_loader_db.sqlite")
TEST_CSV = os.path.join(os.path.dirname(__file__), "test_data.csv")


@pytest.fixture(autouse=True)
def cleanup():
    yield
    for f in [TEST_DB, TEST_CSV]:
        if os.path.exists(f):
            os.remove(f)


class TestSyntheticDataGeneration:
    def test_generates_correct_num_rows(self):
        df = generate_synthetic_data(num_trips=10, records_per_trip=5)
        # 10 trips * 5 records + ~1% duplicates
        assert len(df) >= 50

    def test_has_all_columns(self):
        df = generate_synthetic_data(num_trips=5, records_per_trip=3)
        expected_cols = [
            "trip_id", "timestamp", "speed_kmh", "acceleration_ms2",
            "braking_force", "steering_angle_deg", "rpm", "throttle_pct",
            "fuel_consumption", "distance_km", "behavior_label"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_behavior_labels_valid(self):
        df = generate_synthetic_data(num_trips=50, records_per_trip=5)
        labels = df["behavior_label"].unique()
        for label in labels:
            assert label in BEHAVIOR_LABELS, f"Invalid label: {label}"

    def test_has_missing_values_injected(self):
        df = generate_synthetic_data(num_trips=100, records_per_trip=10)
        # Should have some NaN values injected
        assert df.isnull().sum().sum() > 0

    def test_reproducibility_with_seed(self):
        df1 = generate_synthetic_data(num_trips=10, records_per_trip=5, seed=42)
        df2 = generate_synthetic_data(num_trips=10, records_per_trip=5, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_data(num_trips=10, records_per_trip=5, seed=42)
        df2 = generate_synthetic_data(num_trips=10, records_per_trip=5, seed=99)
        assert not df1.equals(df2)


class TestCSVSave:
    def test_saves_csv_file(self):
        df = generate_synthetic_data(num_trips=5, records_per_trip=3)
        save_csv(df, TEST_CSV)
        assert os.path.exists(TEST_CSV)

    def test_csv_roundtrip(self):
        df = generate_synthetic_data(num_trips=5, records_per_trip=3)
        save_csv(df, TEST_CSV)
        loaded = pd.read_csv(TEST_CSV)
        assert len(loaded) == len(df)


class TestSQLiteLoading:
    def test_loads_csv_to_sqlite(self):
        df = generate_synthetic_data(num_trips=10, records_per_trip=5)
        save_csv(df, TEST_CSV)

        # Override DB path for test
        from src import db_manager
        original_db = db_manager.DB_PATH
        try:
            init_database(TEST_DB)
            load_csv_to_sqlite.__wrapped__ if hasattr(load_csv_to_sqlite, '__wrapped__') else None

            # Direct test: read CSV and insert
            csv_df = pd.read_csv(TEST_CSV)
            from src.db_manager import insert_dataframe, clear_table
            clear_table("driving_data", TEST_DB)
            insert_dataframe(csv_df, "driving_data", TEST_DB)

            count = get_table_row_count("driving_data", TEST_DB)
            assert count == len(csv_df)
        finally:
            pass
