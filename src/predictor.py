"""
Prediction Service Module
==========================
Loads the trained model and scaler, provides single and batch prediction
functions for driving behavior classification.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Union

from src.config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH, ENGINEERED_FEATURES


class DrivingBehaviorPredictor:
    """
    Prediction service for driving behavior classification.

    Loads the trained model, scaler, and label encoder on initialization.
    Provides predict() for single predictions and predict_batch() for bulk.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self._loaded = False

    def load(self):
        """Load model, scaler, and label encoder from disk."""
        if self._loaded:
            return

        # Validate files exist
        model_path = MODEL_PATH
        scaler_path = SCALER_PATH
        encoder_path = MODEL_PATH.replace("best_model", "label_encoder")
        feature_cols_path = FEATURE_COLUMNS_PATH

        for path, name in [(model_path, "Model"), (scaler_path, "Scaler"), (encoder_path, "Label Encoder")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{name} file not found at {path}. Run model training first."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

        if os.path.exists(feature_cols_path):
            self.feature_columns = joblib.load(feature_cols_path)
        else:
            self.feature_columns = ENGINEERED_FEATURES

        self._loaded = True

    def predict(self, features: dict) -> dict:
        """
        Predict driving behavior from a feature dictionary.

        Args:
            features: Dict with feature names as keys and numeric values.
                      Expected keys match ENGINEERED_FEATURES.

        Returns:
            dict with:
                - label: Predicted behavior label (Safe/Normal/Aggressive)
                - confidence: Prediction confidence (0-1)
                - probabilities: Per-class probabilities
        """
        self.load()

        # Validate input
        errors = self._validate_input(features)
        if errors:
            return {"error": errors, "label": None, "confidence": None}

        # Build feature vector in correct column order
        feature_vector = np.array([
            float(features.get(col, 0.0)) for col in self.feature_columns
        ]).reshape(1, -1)

        # Scale features
        feature_scaled = self.scaler.transform(feature_vector)

        # Predict
        prediction = self.model.predict(feature_scaled)[0]
        label = self.label_encoder.inverse_transform([prediction])[0]

        # Confidence
        probabilities = {}
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(feature_scaled)[0]
            confidence = float(max(proba))
            for cls, prob in zip(self.label_encoder.classes_, proba):
                probabilities[cls] = round(float(prob), 4)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities
        }

    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        """
        Predict behavior for multiple feature sets.

        Args:
            features_list: List of feature dictionaries.

        Returns:
            List of prediction result dicts.
        """
        self.load()
        return [self.predict(f) for f in features_list]

    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict behavior for rows in a DataFrame.

        Args:
            df: DataFrame with feature columns.

        Returns:
            DataFrame with added 'predicted_label' and 'confidence' columns.
        """
        self.load()
        result_df = df.copy()

        # Ensure correct column order
        available_cols = [c for c in self.feature_columns if c in df.columns]
        if len(available_cols) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_cols)
            for col in missing:
                result_df[col] = 0.0

        X = result_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        result_df["predicted_label"] = self.label_encoder.inverse_transform(predictions)

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_scaled)
            result_df["confidence"] = probas.max(axis=1).round(4)

        return result_df

    def _validate_input(self, features: dict) -> list[str]:
        """Validate input feature dictionary."""
        errors = []

        if not isinstance(features, dict):
            return ["Input must be a dictionary of features."]

        for col in self.feature_columns:
            if col not in features:
                # Allow missing features (default to 0), but warn
                continue
            val = features[col]
            if val is not None:
                try:
                    float(val)
                except (ValueError, TypeError):
                    errors.append(f"Feature '{col}' must be numeric, got: {type(val).__name__}")

        return errors

    def get_feature_names(self) -> list[str]:
        """Return the expected feature column names."""
        self.load()
        return list(self.feature_columns)

    def get_class_labels(self) -> list[str]:
        """Return the class labels."""
        self.load()
        return list(self.label_encoder.classes_)
