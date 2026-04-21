"""
Model Trainer Module
=====================
Trains, tunes, evaluates, and persists ML models for driving behavior classification.
Supports Logistic Regression, Random Forest, and Gradient Boosting classifiers.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.config import (
    DB_PATH, MODEL_PATH, SCALER_PATH, METRICS_PATH, FEATURE_COLUMNS_PATH,
    ENGINEERED_FEATURES, RANDOM_STATE, TEST_SIZE, VAL_SIZE, CV_FOLDS, ARTIFACTS_DIR
)
from src.db_manager import query_to_dataframe, init_database


class ModelTrainer:
    """
    Handles the full ML training lifecycle:
    - Data splitting (train/val/test with stratification)
    - Baseline model training and comparison
    - Hyperparameter tuning via GridSearchCV
    - Evaluation with confusion matrix, classification report, ROC-AUC
    - Model persistence
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_columns = None

    def load_features(self) -> pd.DataFrame:
        """Load engineered features from SQLite."""
        init_database(self.db_path)
        df = query_to_dataframe("SELECT * FROM features", self.db_path)
        if df.empty:
            raise ValueError("No features found in database. Run feature engineering first.")
        return df

    def prepare_data(self):
        """
        Load features and split into train/val/test sets with stratification.
        Encodes labels to integers.
        """
        print("📦 Preparing data splits...")
        df = self.load_features()

        # Determine available feature columns
        self.feature_columns = [c for c in ENGINEERED_FEATURES if c in df.columns]
        X = df[self.feature_columns].values
        y = self.label_encoder.fit_transform(df["behavior_label"])

        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Second split: train and validation from remaining
        val_ratio = VAL_SIZE / (1 - TEST_SIZE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
        )

        print(f"  Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")

    def train_baselines(self):
        """Train baseline models: Logistic Regression, Random Forest, Gradient Boosting."""
        print("\n🏋️ Training baseline models...")

        model_configs = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_STATE
            )
        }

        for name, model in model_configs.items():
            print(f"\n  Training {name}...")
            model.fit(self.X_train, self.y_train)

            # Evaluate on validation set
            y_pred = model.predict(self.X_val)
            metrics = {
                "accuracy": round(accuracy_score(self.y_val, y_pred), 4),
                "precision": round(precision_score(self.y_val, y_pred, average="weighted"), 4),
                "recall": round(recall_score(self.y_val, y_pred, average="weighted"), 4),
                "f1_score": round(f1_score(self.y_val, y_pred, average="weighted"), 4),
            }

            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=CV_FOLDS, scoring="f1_weighted")
            metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
            metrics["cv_f1_std"] = round(cv_scores.std(), 4)

            self.models[name] = model
            self.results[name] = metrics
            print(f"    Accuracy: {metrics['accuracy']} | F1: {metrics['f1_score']} | CV F1: {metrics['cv_f1_mean']}±{metrics['cv_f1_std']}")

    def select_best_model(self) -> str:
        """Select the best model based on F1 score."""
        self.best_model_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        self.best_model = self.models[self.best_model_name]
        print(f"\n🏆 Best model: {self.best_model_name} (F1: {self.results[self.best_model_name]['f1_score']})")
        return self.best_model_name

    def tune_best_model(self):
        """Hyperparameter tuning for the best model using GridSearchCV."""
        print(f"\n🔧 Tuning {self.best_model_name}...")

        param_grids = {
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "newton-cg"],
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
            }
        }

        param_grid = param_grids.get(self.best_model_name, {})
        if not param_grid:
            print("  ⚠️ No param grid defined, skipping tuning.")
            return

        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=CV_FOLDS,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(self.X_train, self.y_train)

        # Compare tuned vs baseline
        tuned_model = grid_search.best_estimator_
        y_pred_tuned = tuned_model.predict(self.X_val)
        tuned_f1 = round(f1_score(self.y_val, y_pred_tuned, average="weighted"), 4)
        baseline_f1 = self.results[self.best_model_name]["f1_score"]

        print(f"  Baseline F1: {baseline_f1}")
        print(f"  Tuned F1:    {tuned_f1}")
        print(f"  Best params: {grid_search.best_params_}")

        if tuned_f1 >= baseline_f1:
            self.best_model = tuned_model
            self.results[self.best_model_name]["f1_score_tuned"] = tuned_f1
            self.results[self.best_model_name]["best_params"] = grid_search.best_params_
            self.models[self.best_model_name] = tuned_model
            print(f"  ✅ Tuned model adopted (Δ F1: {tuned_f1 - baseline_f1:+.4f})")
        else:
            print("  ⚠️ Baseline was better, keeping original.")

    def evaluate_on_test(self) -> dict:
        """
        Final evaluation on the held-out test set.
        Generates confusion matrix, classification report, and ROC-AUC.
        """
        print(f"\n📊 Evaluating {self.best_model_name} on test set...")

        y_pred = self.best_model.predict(self.X_test)
        y_proba = None
        if hasattr(self.best_model, "predict_proba"):
            y_proba = self.best_model.predict_proba(self.X_test)

        # Core metrics
        eval_metrics = {
            "model_name": self.best_model_name,
            "test_accuracy": round(accuracy_score(self.y_test, y_pred), 4),
            "test_precision": round(precision_score(self.y_test, y_pred, average="weighted"), 4),
            "test_recall": round(recall_score(self.y_test, y_pred, average="weighted"), 4),
            "test_f1": round(f1_score(self.y_test, y_pred, average="weighted"), 4),
        }

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        eval_metrics["confusion_matrix"] = cm.tolist()
        eval_metrics["class_labels"] = list(self.label_encoder.classes_)

        # Classification report
        report = classification_report(
            self.y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        eval_metrics["classification_report"] = report

        # ROC-AUC (One-vs-Rest)
        if y_proba is not None:
            y_test_bin = label_binarize(self.y_test, classes=list(range(len(self.label_encoder.classes_))))
            try:
                roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")
                eval_metrics["roc_auc"] = round(roc_auc, 4)
            except ValueError:
                eval_metrics["roc_auc"] = None

        # Feature importance (if available)
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            feature_imp = dict(zip(self.feature_columns, [round(float(x), 4) for x in importances]))
            eval_metrics["feature_importances"] = dict(
                sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
            )

        # Model comparison summary
        eval_metrics["model_comparison"] = self.results

        # Print summary
        print(f"  Test Accuracy:  {eval_metrics['test_accuracy']}")
        print(f"  Test F1:        {eval_metrics['test_f1']}")
        print(f"  Test Precision: {eval_metrics['test_precision']}")
        print(f"  Test Recall:    {eval_metrics['test_recall']}")
        if eval_metrics.get("roc_auc"):
            print(f"  ROC-AUC:        {eval_metrics['roc_auc']}")

        return eval_metrics

    def save_model(self, eval_metrics: dict):
        """Save the best model, label encoder, and evaluation metrics."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # Save model
        joblib.dump(self.best_model, MODEL_PATH)
        print(f"  💾 Model saved to {MODEL_PATH}")

        # Save label encoder
        encoder_path = MODEL_PATH.replace("best_model", "label_encoder")
        joblib.dump(self.label_encoder, encoder_path)
        print(f"  💾 Label encoder saved to {encoder_path}")

        # Save feature columns list
        joblib.dump(self.feature_columns, FEATURE_COLUMNS_PATH)

        # Save evaluation metrics as JSON
        metrics_serializable = self._make_serializable(eval_metrics)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"  💾 Metrics saved to {METRICS_PATH}")

    def _make_serializable(self, obj):
        """Recursively convert numpy types to Python native types for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def run_full_pipeline(self) -> dict:
        """Execute the full training pipeline end-to-end."""
        print("=" * 60)
        print("🚀 ML Training Pipeline — Driving Behavior Classification")
        print("=" * 60)

        self.prepare_data()
        self.train_baselines()
        self.select_best_model()
        self.tune_best_model()
        eval_metrics = self.evaluate_on_test()
        self.save_model(eval_metrics)

        print("\n" + "=" * 60)
        print("✅ Pipeline complete!")
        print("=" * 60)

        return eval_metrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_full_pipeline()
