"""
Integration Test — Full Pipeline
==================================
Tests the end-to-end flow: data generation → cleaning → features → training → prediction.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEndToEndPipeline:
    """
    Integration test that runs the full pipeline on a small dataset.
    Uses the default DB and model paths from config.
    """

    @pytest.mark.slow
    def test_full_pipeline_runs(self):
        """Verify the entire pipeline executes without errors."""
        from src.data_loader import generate_and_load
        from src.data_cleaner import clean_data
        from src.feature_engineer import engineer_features
        from src.model_trainer import ModelTrainer
        from src.predictor import DrivingBehaviorPredictor
        from src.config import MODEL_PATH, SCALER_PATH, METRICS_PATH

        # Step 1: Generate data (small set for speed)
        generate_and_load(num_trips=100)

        # Step 2: Clean
        cleaned_df = clean_data()
        assert len(cleaned_df) > 0

        # Step 3: Features
        features_df = engineer_features()
        assert len(features_df) > 0
        assert "behavior_label" in features_df.columns

        # Step 4: Train
        trainer = ModelTrainer()
        metrics = trainer.run_full_pipeline()
        assert metrics["test_accuracy"] > 0.5  # Should be better than random
        assert metrics["test_f1"] > 0.5
        assert os.path.exists(MODEL_PATH)
        assert os.path.exists(SCALER_PATH)
        assert os.path.exists(METRICS_PATH)

        # Step 5: Predict
        predictor = DrivingBehaviorPredictor()
        predictor.load()
        result = predictor.predict({
            "avg_speed": 70.0, "max_speed": 100.0, "speed_std": 15.0,
            "avg_acceleration": 2.0, "max_acceleration": 4.0, "acceleration_std": 1.0,
            "avg_braking": 4.0, "max_braking": 7.0, "braking_std": 1.5,
            "avg_steering_angle": 12.0, "max_steering_angle": 30.0, "steering_variability": 6.0,
            "avg_rpm": 3000.0, "max_rpm": 4500.0,
            "avg_throttle": 50.0, "throttle_variability": 12.0,
            "harsh_braking_count": 2.0, "rapid_acceleration_count": 2.0,
            "avg_fuel_consumption": 8.0, "total_distance": 30.0,
        })
        assert result["label"] in ["Safe", "Normal", "Aggressive"]
        assert result["confidence"] > 0
