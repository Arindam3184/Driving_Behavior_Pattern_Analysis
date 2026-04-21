"""
Unit Tests for db_manager module
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_manager import (
    init_database, insert_dataframe, query_to_dataframe,
    get_table_row_count, clear_table, table_exists,
    log_cleaning_action, get_connection
)

TEST_DB = os.path.join(os.path.dirname(__file__), "test_db.sqlite")


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Create a fresh test database for each test."""
    init_database(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestDatabaseInit:
    def test_tables_created(self):
        assert table_exists("driving_data", TEST_DB)
        assert table_exists("features", TEST_DB)
        assert table_exists("cleaning_log", TEST_DB)

    def test_nonexistent_table(self):
        assert not table_exists("nonexistent_table", TEST_DB)


class TestInsertAndQuery:
    def test_insert_dataframe(self):
        df = pd.DataFrame({
            "trip_id": ["T001", "T002"],
            "timestamp": ["2025-01-01 08:00:00", "2025-01-01 09:00:00"],
            "speed_kmh": [60.0, 80.0],
            "acceleration_ms2": [1.5, 2.0],
            "braking_force": [3.0, 4.0],
            "steering_angle_deg": [5.0, 10.0],
            "rpm": [2000.0, 3000.0],
            "throttle_pct": [30.0, 50.0],
            "fuel_consumption": [6.0, 8.0],
            "distance_km": [2.0, 3.0],
            "behavior_label": ["Safe", "Normal"]
        })
        insert_dataframe(df, "driving_data", TEST_DB)
        count = get_table_row_count("driving_data", TEST_DB)
        assert count == 2

    def test_query_to_dataframe(self):
        df = pd.DataFrame({
            "trip_id": ["T001"],
            "timestamp": ["2025-01-01 08:00:00"],
            "speed_kmh": [60.0],
            "acceleration_ms2": [1.5],
            "braking_force": [3.0],
            "steering_angle_deg": [5.0],
            "rpm": [2000.0],
            "throttle_pct": [30.0],
            "fuel_consumption": [6.0],
            "distance_km": [2.0],
            "behavior_label": ["Safe"]
        })
        insert_dataframe(df, "driving_data", TEST_DB)
        result = query_to_dataframe("SELECT * FROM driving_data WHERE trip_id = ?", TEST_DB, params=("T001",))
        assert len(result) == 1
        assert result.iloc[0]["speed_kmh"] == 60.0

    def test_clear_table(self):
        df = pd.DataFrame({
            "trip_id": ["T001"],
            "timestamp": ["2025-01-01 08:00:00"],
            "speed_kmh": [60.0],
            "acceleration_ms2": [1.5],
            "braking_force": [3.0],
            "steering_angle_deg": [5.0],
            "rpm": [2000.0],
            "throttle_pct": [30.0],
            "fuel_consumption": [6.0],
            "distance_km": [2.0],
            "behavior_label": ["Safe"]
        })
        insert_dataframe(df, "driving_data", TEST_DB)
        assert get_table_row_count("driving_data", TEST_DB) == 1
        clear_table("driving_data", TEST_DB)
        assert get_table_row_count("driving_data", TEST_DB) == 0


class TestCleaningLog:
    def test_log_cleaning_action(self):
        log_cleaning_action("test_metric", "100", "95", "removed 5 rows", TEST_DB)
        result = query_to_dataframe("SELECT * FROM cleaning_log", TEST_DB)
        assert len(result) == 1
        assert result.iloc[0]["metric"] == "test_metric"
        assert result.iloc[0]["action_taken"] == "removed 5 rows"


class TestConnection:
    def test_context_manager(self):
        with get_connection(TEST_DB) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM driving_data")
            count = cursor.fetchone()[0]
            assert count == 0
