# 🚗 Driving Behavior Pattern Analysis

An ML-powered predictive analytics system that analyzes driving behavior patterns and displays insights through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Database-green?logo=sqlite)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Dashboard Pages](#dashboard-pages)
- [Testing](#testing)
- [Tech Stack](#tech-stack)

---

## 🎯 Overview

This project classifies driving behavior into three categories — **Safe**, **Normal**, and **Aggressive** — using telemetry data from vehicle sensors. It features:

- **Synthetic data generation** with realistic distributions
- **Automated data cleaning** (outlier removal, imputation, deduplication)
- **Feature engineering** (20 derived features per trip)
- **Model comparison** (Logistic Regression, Random Forest, Gradient Boosting)
- **Hyperparameter tuning** with GridSearchCV
- **Interactive Streamlit dashboard** with 4 pages

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 Data Pipeline | End-to-end automated: generation → cleaning → features → training |
| 🧠 ML Models | 3 classifiers compared, best one auto-tuned and persisted |
| 📊 Dashboard | 4-page Streamlit UI with dark theme and rich visualizations |
| 🎯 Live Predictions | Real-time behavior classification with confidence scores |
| 🔍 Data Explorer | Interactive filters, distributions, correlations, parallel coordinates |
| 📈 Model Insights | Confusion matrix, feature importance, radar charts, ROC-AUC |
| ✅ Testing | Unit + integration tests with pytest |

---

## 🏗️ Architecture

```
CSV Data → Data Cleaning → Feature Engineering → Model Training → Prediction API
                                                                        ↓
                                                              Streamlit Dashboard
                                                         (Overview | Explorer | Insights | Predictions)
```

---

## 🚀 Quick Start

### Option 1: One-Click (Windows)

```bash
# Double-click or run:
run.bat
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ML pipeline (generates data, trains models)
python src/run_pipeline.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard will open at **http://localhost:8501**

---

## 📁 Project Structure

```
driving-behavior/
├── dashboard/                  # Streamlit dashboard
│   ├── app.py                  # Main app (Overview page)
│   └── pages/
│       ├── 1_Data_Explorer.py  # Data exploration & visualization
│       ├── 2_Model_Insights.py # Model evaluation & metrics
│       └── 3_Live_Predictions.py # Interactive prediction UI
├── src/                        # Core source code
│   ├── config.py               # Centralized configuration
│   ├── db_manager.py           # SQLite database operations
│   ├── data_loader.py          # Data generation & CSV loading
│   ├── data_cleaner.py         # Data cleaning pipeline
│   ├── feature_engineer.py     # Feature engineering
│   ├── model_trainer.py        # ML training, tuning, evaluation
│   ├── predictor.py            # Prediction service
│   └── run_pipeline.py         # End-to-end pipeline runner
├── tests/                      # Test suite
│   ├── test_db_manager.py
│   ├── test_data_loader.py
│   ├── test_predictor.py
│   └── test_integration.py
├── data/                       # Generated data (gitignored)
├── models/                     # Saved models (gitignored)
├── artifacts/                  # Evaluation artifacts (gitignored)
├── requirements.txt
├── run.bat                     # One-click launcher (Windows)
└── README.md
```

---

## 🧠 ML Pipeline

### Data Generation
- 500 trips × 10 records each = 5,000+ records
- Realistic distributions per behavior class
- Injected noise (~2%) and missing values (~1.5%) for real-world simulation

### Data Cleaning
- Missing values → median imputation
- Outliers → IQR-based detection and removal
- Duplicates → detected and removed
- Cleaning report generated with before/after statistics

### Feature Engineering (20 features)
| Category | Features |
|----------|----------|
| Speed | avg_speed, max_speed, speed_std |
| Acceleration | avg_acceleration, max_acceleration, acceleration_std |
| Braking | avg_braking, max_braking, braking_std |
| Steering | avg_steering_angle, max_steering_angle, steering_variability |
| Engine | avg_rpm, max_rpm, avg_throttle, throttle_variability |
| Events | harsh_braking_count, rapid_acceleration_count |
| Other | avg_fuel_consumption, total_distance |

### Model Training
- **Algorithms:** Logistic Regression, Random Forest, Gradient Boosting
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Tuning:** GridSearchCV with 5-fold cross-validation
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

---

## 📊 Dashboard Pages

1. **📊 Overview** — KPI cards, behavior distribution pie chart, feature histograms, model summary
2. **🔍 Data Explorer** — Filterable data table, distributions, correlation heatmap, parallel coordinates
3. **🧠 Model Insights** — Confusion matrix, classification report, feature importance, model radar chart
4. **🎯 Live Predictions** — Slider-based input form, preset profiles, confidence gauge, probability breakdown

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run only fast tests (skip integration)
pytest tests/ -v -m "not slow"
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| ML | scikit-learn, NumPy, pandas |
| Dashboard | Streamlit, Plotly |
| Database | SQLite |
| Testing | pytest, pytest-cov |
| Serialization | joblib |

---

## 📄 License

This project is for educational and demonstration purposes.
