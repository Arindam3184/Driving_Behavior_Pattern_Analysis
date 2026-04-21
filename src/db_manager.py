"""
Database Manager Module
=======================
Handles all SQLite operations — connection management, table creation,
CRUD operations, and query execution.
"""

import sqlite3
import pandas as pd
from contextlib import contextmanager
from src.config import DB_PATH


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """Context manager for database connections with auto-commit and close."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(db_path: str = DB_PATH):
    """
    Initialize the database with required tables and indexes.
    Creates 'driving_data' and 'features' tables if they don't exist.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Raw driving data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS driving_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trip_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                speed_kmh REAL,
                acceleration_ms2 REAL,
                braking_force REAL,
                steering_angle_deg REAL,
                rpm REAL,
                throttle_pct REAL,
                fuel_consumption REAL,
                distance_km REAL,
                behavior_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Engineered features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trip_id TEXT UNIQUE NOT NULL,
                avg_speed REAL, max_speed REAL, speed_std REAL,
                avg_acceleration REAL, max_acceleration REAL, acceleration_std REAL,
                avg_braking REAL, max_braking REAL, braking_std REAL,
                avg_steering_angle REAL, max_steering_angle REAL, steering_variability REAL,
                avg_rpm REAL, max_rpm REAL,
                avg_throttle REAL, throttle_variability REAL,
                harsh_braking_count INTEGER,
                rapid_acceleration_count INTEGER,
                avg_fuel_consumption REAL,
                total_distance REAL,
                behavior_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Cleaning log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cleaning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                before_value TEXT,
                after_value TEXT,
                action_taken TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trip_id ON driving_data(trip_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON driving_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_behavior ON driving_data(behavior_label)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_trip ON features(trip_id)")


def insert_dataframe(df: pd.DataFrame, table_name: str, db_path: str = DB_PATH, if_exists: str = "append"):
    """
    Insert a pandas DataFrame into the specified table.

    Args:
        df: DataFrame to insert.
        table_name: Target table name.
        db_path: Path to the database.
        if_exists: 'append', 'replace', or 'fail'.
    """
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()


def query_to_dataframe(query: str, db_path: str = DB_PATH, params: tuple = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.

    Args:
        query: SQL query string.
        db_path: Path to the database.
        params: Optional query parameters.

    Returns:
        pd.DataFrame with query results.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def get_table_row_count(table_name: str, db_path: str = DB_PATH) -> int:
    """Return the number of rows in a table."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]


def clear_table(table_name: str, db_path: str = DB_PATH):
    """Delete all rows from a table."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")


def table_exists(table_name: str, db_path: str = DB_PATH) -> bool:
    """Check if a table exists in the database."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None


def log_cleaning_action(metric: str, before: str, after: str, action: str, db_path: str = DB_PATH):
    """Log a data cleaning action to the cleaning_log table."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO cleaning_log (metric, before_value, after_value, action_taken) VALUES (?, ?, ?, ?)",
            (metric, str(before), str(after), action)
        )
