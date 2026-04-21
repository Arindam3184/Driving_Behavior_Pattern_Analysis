"""
Configuration module for the Driving Behavior Pattern Analysis project.
Centralizes all paths, constants, and settings.
"""

import os

# ─── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ─── File Paths ─────────────────────────────────────────────────────────────────
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "driving_data.csv")
DB_PATH = os.path.join(DATA_DIR, "driving_behavior.db")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")

# ─── Data Schema ────────────────────────────────────────────────────────────────
RAW_COLUMNS = [
    "trip_id", "timestamp", "speed_kmh", "acceleration_ms2",
    "braking_force", "steering_angle_deg", "rpm", "throttle_pct",
    "fuel_consumption", "distance_km", "behavior_label"
]

BEHAVIOR_LABELS = ["Safe", "Normal", "Aggressive"]

# ─── Feature Engineering ────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "speed_kmh", "acceleration_ms2", "braking_force",
    "steering_angle_deg", "rpm", "throttle_pct",
    "fuel_consumption", "distance_km"
]

ENGINEERED_FEATURES = [
    "avg_speed", "max_speed", "speed_std",
    "avg_acceleration", "max_acceleration", "acceleration_std",
    "avg_braking", "max_braking", "braking_std",
    "avg_steering_angle", "max_steering_angle", "steering_variability",
    "avg_rpm", "max_rpm",
    "avg_throttle", "throttle_variability",
    "harsh_braking_count", "rapid_acceleration_count",
    "avg_fuel_consumption", "total_distance"
]

# ─── Model Training ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # 15% of remaining after test split
CV_FOLDS = 5

# ─── Thresholds ─────────────────────────────────────────────────────────────────
HARSH_BRAKING_THRESHOLD = 7.0       # braking_force > 7 is harsh
RAPID_ACCELERATION_THRESHOLD = 3.5  # acceleration > 3.5 is rapid

# ─── Ensure directories exist ───────────────────────────────────────────────────
for d in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)
