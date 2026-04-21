"""
Pipeline Runner
================
Orchestrates the full data-to-model pipeline.
Run this script to execute everything end-to-end.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import generate_and_load
from src.data_cleaner import clean_data
from src.feature_engineer import engineer_features
from src.model_trainer import ModelTrainer


def run_pipeline():
    """Execute the full pipeline: generate data → clean → features → train."""
    print("=" * 70)
    print("🚗  DRIVING BEHAVIOR PATTERN ANALYSIS — FULL PIPELINE")
    print("=" * 70)

    # Step 1: Generate and load data
    print("\n" + "─" * 70)
    print("STEP 1/4: Data Generation & Ingestion")
    print("─" * 70)
    generate_and_load(num_trips=500)

    # Step 2: Clean data
    print("\n" + "─" * 70)
    print("STEP 2/4: Data Cleaning")
    print("─" * 70)
    clean_data()

    # Step 3: Feature engineering
    print("\n" + "─" * 70)
    print("STEP 3/4: Feature Engineering")
    print("─" * 70)
    engineer_features()

    # Step 4: Model training
    print("\n" + "─" * 70)
    print("STEP 4/4: Model Training & Evaluation")
    print("─" * 70)
    trainer = ModelTrainer()
    metrics = trainer.run_full_pipeline()

    print("\n" + "=" * 70)
    print("🎉  PIPELINE COMPLETE — All artifacts saved!")
    print("=" * 70)
    print("\nNext step: Run the dashboard with:")
    print("  streamlit run dashboard/app.py")

    return metrics


if __name__ == "__main__":
    run_pipeline()
