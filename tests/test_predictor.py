"""
Unit Tests for predictor module
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_PATH, SCALER_PATH


# Skip all tests if model hasn't been trained yet
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model not trained yet. Run pipeline first."
)


@pytest.fixture
def predictor():
    from src.predictor import DrivingBehaviorPredictor
    p = DrivingBehaviorPredictor()
    p.load()
    return p


class TestPredictorLoading:
    def test_loads_successfully(self, predictor):
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.label_encoder is not None

    def test_feature_names_not_empty(self, predictor):
        names = predictor.get_feature_names()
        assert len(names) > 0

    def test_class_labels_correct(self, predictor):
        labels = predictor.get_class_labels()
        assert "Safe" in labels
        assert "Normal" in labels
        assert "Aggressive" in labels


class TestSinglePrediction:
    def test_safe_driver_profile(self, predictor):
        features = {
            "avg_speed": 50.0, "max_speed": 70.0, "speed_std": 8.0,
            "avg_acceleration": 1.0, "max_acceleration": 2.5, "acceleration_std": 0.5,
            "avg_braking": 2.0, "max_braking": 4.0, "braking_std": 0.8,
            "avg_steering_angle": 5.0, "max_steering_angle": 15.0, "steering_variability": 3.0,
            "avg_rpm": 2000.0, "max_rpm": 3000.0,
            "avg_throttle": 30.0, "throttle_variability": 6.0,
            "harsh_braking_count": 0.0, "rapid_acceleration_count": 0.0,
            "avg_fuel_consumption": 6.0, "total_distance": 20.0,
        }
        result = predictor.predict(features)
        assert result["label"] is not None
        assert result["confidence"] > 0
        assert "error" not in result or not result["error"]

    def test_aggressive_driver_profile(self, predictor):
        features = {
            "avg_speed": 120.0, "max_speed": 170.0, "speed_std": 30.0,
            "avg_acceleration": 5.0, "max_acceleration": 8.0, "acceleration_std": 2.5,
            "avg_braking": 8.0, "max_braking": 12.0, "braking_std": 3.0,
            "avg_steering_angle": 30.0, "max_steering_angle": 60.0, "steering_variability": 15.0,
            "avg_rpm": 5000.0, "max_rpm": 7000.0,
            "avg_throttle": 80.0, "throttle_variability": 18.0,
            "harsh_braking_count": 8.0, "rapid_acceleration_count": 7.0,
            "avg_fuel_consumption": 15.0, "total_distance": 55.0,
        }
        result = predictor.predict(features)
        assert result["label"] is not None
        assert result["confidence"] > 0

    def test_prediction_has_probabilities(self, predictor):
        features = {
            "avg_speed": 70.0, "max_speed": 100.0, "speed_std": 15.0,
            "avg_acceleration": 2.0, "max_acceleration": 4.0, "acceleration_std": 1.0,
            "avg_braking": 4.0, "max_braking": 7.0, "braking_std": 1.5,
            "avg_steering_angle": 12.0, "max_steering_angle": 30.0, "steering_variability": 6.0,
            "avg_rpm": 3000.0, "max_rpm": 4500.0,
            "avg_throttle": 50.0, "throttle_variability": 12.0,
            "harsh_braking_count": 2.0, "rapid_acceleration_count": 2.0,
            "avg_fuel_consumption": 8.0, "total_distance": 30.0,
        }
        result = predictor.predict(features)
        assert "probabilities" in result
        probs = result["probabilities"]
        assert len(probs) == 3
        # Probabilities should sum to ~1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_prediction_label_in_valid_set(self, predictor):
        features = {col: 50.0 for col in predictor.get_feature_names()}
        result = predictor.predict(features)
        assert result["label"] in ["Safe", "Normal", "Aggressive"]

    def test_handles_missing_features_gracefully(self, predictor):
        """Predict with partial features — should default missing to 0."""
        result = predictor.predict({"avg_speed": 60.0})
        assert result["label"] is not None


class TestBatchPrediction:
    def test_batch_returns_correct_count(self, predictor):
        features_list = [
            {col: 50.0 for col in predictor.get_feature_names()},
            {col: 100.0 for col in predictor.get_feature_names()},
            {col: 20.0 for col in predictor.get_feature_names()},
        ]
        results = predictor.predict_batch(features_list)
        assert len(results) == 3
        for r in results:
            assert r["label"] is not None
