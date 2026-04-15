# 🏭 SECOM MSPC – Industry 4.0 Multivariate Statistical Process Control

**Semiconductor Manufacturing Quality Prediction using the SECOM Dataset**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Business Problem & Motivation

In semiconductor manufacturing, a **single failed wafer batch costs $50,000–$500,000**. Traditional quality inspection catches defects *after* the fact—when the wafer is already scrapped.

This project implements a **real-time Multivariate Statistical Process Control (MSPC)** system that monitors 591 process sensors simultaneously and detects abnormal process behaviour *before* wafers fail final inspection.

### Key Capabilities

| Capability | Method |
|---|---|
| Detect large sudden shifts | Hotelling T² chart |
| Detect small sustained drifts | MEWMA chart |
| Root-cause diagnosis | MYT T² decomposition |
| Predictive classification | Random Forest, XGBoost, SVM |
| Real-time monitoring | Streamlit dashboard |

---

## 2. Dataset

| Property | Value |
|---|---|
| **Source** | [UCI Machine Learning Repository – SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM) |
| **Observations** | 1,567 production runs |
| **Features** | 591 process sensor variables |
| **Target** | Binary: Pass (−1 → 0) / Fail (+1 → 1) |
| **Imbalance** | ~93.3 % Pass / ~6.7 % Fail |
| **Missing Data** | ~5.8 % of all values |

---

## 3. Project Architecture

```
secom_mspc/
│
├── main.py                          ← Master runner (runs everything)
├── config.py                        ← All configuration parameters
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
│
├── data/
│   ├── loader.py                    ← Data downloading & loading
│   └── raw/                         ← Auto-created for raw files
│
├── preprocessing/
│   ├── quality_checker.py           ← Data quality assessment
│   ├── cleaner.py                   ← Missing values, outliers, scaling
│   └── feature_selector.py          ← Feature reduction pipeline
│
├── statistical_analysis/
│   ├── eda.py                       ← Exploratory data analysis
│   ├── normality_tests.py           ← Distribution testing
│   └── correlation_analysis.py      ← Multicollinearity analysis
│
├── dimensionality_reduction/
│   ├── pca_engine.py                ← PCA implementation & analysis
│   └── component_selector.py        ← Optimal component selection
│
├── mspc/
│   ├── phase_manager.py             ← Phase I / Phase II setup
│   ├── hotelling_t2.py              ← Hotelling T² chart
│   ├── mewma.py                     ← MEWMA chart
│   ├── combined_mspc.py             ← Integrated MSPC system
│   ├── arl_simulator.py             ← ARL computation engine
│   └── fault_diagnosis.py           ← T² decomposition & OCAP
│
├── predictive_model/
│   ├── imbalance_handler.py         ← SMOTE & class weights
│   ├── model_trainer.py             ← Multi-model training
│   ├── model_evaluator.py           ← Performance evaluation
│   └── feature_importance.py        ← Feature importance analysis
│
├── visualization/
│   ├── control_chart_plots.py       ← SPC chart visualisations
│   ├── pca_plots.py                 ← PCA visualisation suite
│   ├── performance_plots.py         ← Model performance plots
│   └── report_generator.py          ← Auto HTML report
│
├── dashboard/
│   ├── app.py                       ← Streamlit dashboard
│   ├── chart_components.py          ← Reusable chart widgets
│   └── alert_system.py              ← Real-time alert engine
│
├── outputs/
│   ├── figures/                     ← All saved plots (PNG, HTML)
│   ├── models/                      ← Saved ML & MSPC models
│   ├── reports/                     ← Generated reports
│   ├── tables/                      ← CSV result tables
│   └── logs/                        ← Execution logs
│
└── tests/
    ├── test_preprocessing.py        ← Preprocessing unit tests
    ├── test_mspc.py                 ← SPC tests
    └── test_models.py               ← Model tests
```

---

## 4. Installation

### Option A – pip (recommended)

```bash
# Clone or download the project
cd secom_mspc

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Option B – conda

```bash
conda create -n secom python=3.10 -y
conda activate secom
pip install -r requirements.txt
```

---

## 5. Quick Start

```bash
# 1. Run the complete pipeline (downloads data automatically)
python main.py

# 2. Launch the interactive dashboard
streamlit run dashboard/app.py

# 3. Run unit tests
python -m pytest tests/ -v --tb=short
```

---

## 6. Detailed Usage

### Run Only MSPC Phases

```bash
python main.py --models-only   # Skip MSPC, run only ML models
python main.py --phases-only   # Skip ML, run only MSPC charts
python main.py --skip-download # Use already-downloaded data
```

### Individual Modules

```python
from data.loader import SECOMDataLoader
from preprocessing.cleaner import SECOMCleaner
from dimensionality_reduction.pca_engine import SECOMPCAEngine
from mspc.hotelling_t2 import HotellingT2Chart
from mspc.mewma import MEWMAChart

# Load
loader = SECOMDataLoader()
loader.download_data()
X, y, df = loader.load_data()

# Preprocess
cleaner = SECOMCleaner()
X_clean = cleaner.fit_transform(X)

# PCA
pca = SECOMPCAEngine()
X_pca = pca.fit_transform(X_clean, y)

# T² Chart
t2 = HotellingT2Chart()
t2.fit_phase1(X_pca[:800])  # Phase I
results = t2.monitor_phase2(X_pca[800:])
```

---

## 7. Configuration Guide

All parameters are in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `missing_threshold` | 0.50 | Drop features with >50 % missing |
| `correlation_threshold` | 0.95 | Remove features with \|r\| > 0.95 |
| `selected_variance` | 0.95 | PCA variance retention target |
| `alpha` | 0.0027 | SPC false-alarm rate (≈ 3σ) |
| `phase1_ratio` | 0.60 | Fraction of PASS samples for Phase I |
| `mewma_lambda` | 0.10 | MEWMA smoothing parameter |
| `test_size` | 0.30 | ML test set fraction (temporal split) |
| `random_seed` | 42 | Reproducibility seed |

---

## 8. Output Files

After running the pipeline, you will find:

| Directory | Contents |
|---|---|
| `outputs/figures/` | PNG plots (300 DPI) for EDA, PCA, MSPC, models |
| `outputs/models/` | Serialised models (`.pkl`) – cleaner, PCA, T², MEWMA, classifiers |
| `outputs/reports/` | `secom_mspc_complete_report.html`, `run_parameters.json` |
| `outputs/tables/` | CSV tables: descriptive stats, model comparison, ARL, MSPC results |
| `outputs/logs/` | `run.log` with timestamped execution log |

---

## 9. Mathematical Background

### Hotelling T²

$$T^2_i = (\\mathbf{x}_i - \\bar{\\mathbf{x}})^T \\mathbf{S}^{-1} (\\mathbf{x}_i - \\bar{\\mathbf{x}})$$

**Phase I UCL** (exact *F*-distribution):

$$UCL_I = \\frac{p(m-1)(m+1)}{m(m-p)} \\cdot F_{p,\\,m-p,\\,\\alpha}$$

### MEWMA

$$\\mathbf{Z}_i = \\lambda(\\mathbf{X}_i - \\boldsymbol{\\mu}_0) + (1-\\lambda)\\mathbf{Z}_{i-1}$$

with time-varying covariance:

$$\\Sigma_{Z,i} = \\frac{\\lambda}{2-\\lambda}\\left[1-(1-\\lambda)^{2i}\\right]\\Sigma$$

### MYT Decomposition

For fault diagnosis, the contribution of variable *j* is:

$$d_j = T^2_{\\text{full}} - T^2_{-j}$$

---

## 10. Dashboard

The Streamlit dashboard provides five pages:

1. **Home** – KPI cards, mini T² chart, class distribution
2. **Control Charts** – Interactive zoomable T² and MEWMA charts
3. **PCA Analysis** – Scree plot, 3-D biplot, loadings
4. **ML Models** – Comparison table, ROC curves, threshold slider
5. **Alert Center** – Sortable alert log with drill-down details

Launch: `streamlit run dashboard/app.py`

---

## 11. References

1. Mason, R. L., & Young, J. C. (2002). *Multivariate Statistical Process Control with Industrial Applications*. SIAM.
2. Lowry, C. A., et al. (1992). A multivariate exponentially weighted moving average control chart. *Technometrics*, 34(1), 46–53.
3. UCI Machine Learning Repository – SECOM Dataset.
4. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.

---

## 12. License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```
