# WeatherPipe: Automated Meteorological Data Preprocessing and Feature Engineering Pipeline

[![CI/CD Pipeline](https://github.com/baidi-prob/DSTT-Project-Working/actions/workflows/ci.yml/badge.svg)](https://github.com/baidi-prob/DSTT-Project-Working/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Course**: Data Science Tools and Techniques (DSTT, 20 Marks) — MS Data Science / AI  
> **Authors**: Obaid Raja (26I-8034), Zoffishan Sattar (26I-8029), Hamza Ali (26I-8007)  
> **Institution**: FAST National University of Computer and Emerging Sciences  
> **Repository**: [https://github.com/baidi-prob/DSTT-Project-Working](https://github.com/baidi-prob/DSTT-Project-Working)

---

## 📌 Executive Overview

Weather telemetry collected from environmental sensor networks is notorious for missing transmission bursts, instrument resets, unphysical voltage spikes, and complex diurnal non-linearities. Traditional data science workflows rely on manual, ad-hoc exploratory scripts (e.g. bidirectional linear interpolation, arbitrary mean-filling), which introduce **future data leakage**, human subjectivity, and operational collapse in live production environments.

**WeatherPipe** replaces this fragile paradigm with a domain-aware, strictly causal automated preprocessing and feature engineering pipeline integrated with an MLOps CI/CD framework. Evaluated on **45,253 hourly records** from the Kaggle Historical Hourly Weather benchmark on a 6-hour ahead temperature forecast ($T_{t+6}$), WeatherPipe achieves a **15.90% error reduction** over an expert manual analyst baseline across 4 time-ordered cross-validation folds.

---

## 🏆 Key Benchmark Results (Empirical Real Data)

### RQ1: 6-Hour Ahead Temperature Forecast ($T_{t+6}$)
*Evaluated across 4 expanding-window temporal folds on 45,253 real hourly records (San Francisco, 2012–2017) using XGBoost.*

| Forecasting Method | Preprocessing & Feature Strategy | Overall RMSE (°C) | Overall MAE (°C) | Overall $R^2$ | Gain vs Best Manual (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Persistence (Sanity Reference)** | $\hat{y}_t = T_t$ (no model) | 5.3442 | 4.2236 | 0.0961 | — |
| **Manual Baseline A (Naive)** | Global mean fill, raw columns | 4.5095 | 3.4551 | 0.3564 | — |
| **Manual Baseline B (Ad-hoc Analyst)** | Forward fill, integer calendar, 1 lag, 1 rolling mean | 2.1136 | 1.5719 | 0.8586 | — |
| **WeatherPipe (Proposed Automated)** | **Aligned, causal diurnal imputer, multi-scale features** | **1.7776** | **1.3212** | **0.9000** | **+15.90%** |

*95% Bootstrap Confidence Interval for Improvement ($\Delta \text{RMSE}$): $[+0.298^\circ\text{C}, +0.374^\circ\text{C}]$, proving statistical significance at $p < 0.001$.*

### RQ2: Missing Sensor Data Imputation Benchmark (Ground Truth Test)
| Outage Scenario | Global Mean Fill | Forward Fill (ffill) | Linear Interpolation (Non-causal)* | WeatherPipe Causal Diurnal |
| :--- | :---: | :---: | :---: | :---: |
| **Random Dropout (10%)** | 6.2102°C | 1.5004°C | 0.6377°C* | **0.9931°C** |
| **Burst Outage (6h Block)** | 5.8019°C | 3.6669°C | 1.3913°C* | **1.8000°C** |
| **Burst Outage (12h Block)** | 5.8076°C | 5.1487°C | 3.0625°C* | **2.4253°C** |
| **Burst Outage (24h Block)** | 5.4459°C | 4.8859°C | 4.8688°C* | **2.4928°C** |

*\*Note: Linear interpolation peeks into future observations ($t+k$), making it physically impossible for live operational deployment. WeatherPipe is 100% causal and leak-free.*

---

## 🏗️ System Architecture

```
                      +-----------------------------+
                      |   Incoming Raw Weather CSV  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 1. Ingestion & Alignment    |
                      |    (Hourly grid, Dedup)     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 2. Quality & Bounds Filter  |
                      |    (Sentinels, Spikes)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 3. Causal Diurnal Imputer   |
                      |    (No Future Leakage)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 4. Circular & Multi-Feature |
                      |    (Lags, Rolling, Encodings|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 5. Statistical Drift Audit  |
                      |    (KS-Test, Missing Rate)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 6. Model Training / Scoring |
                      |    (Fixed-Budget XGBoost)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | 7. CI/CD Accuracy Gate      |
                      |    (Assert: RMSE_auto <     |
                      |             RMSE_manual)    |
                      +--------------+--------------+
                                     |
                       [Pass: 0]     |     [Fail: 1]
                     +---------------+---------------+
                     |                               |
                     v                               v
         +-----------------------+       +-----------------------+
         | Deploy to Production  |       | Block Build & Alert   |
         | & Log MLflow Run      |       | Quarantine Bad Batch  |
         +-----------------------+       +-----------------------+
```

---

## 📂 Repository Structure

```
DSTT-Project-Working/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI/CD: Ruff linting, Pytest, RQ1 eval, Accuracy Gate
│       └── data-pipeline.yml      # Continuous ingestion, weekly replay, drift detection
├── data/
│   ├── raw/                       # Raw Kaggle CSVs (git-ignored for repository hygiene)
│   ├── sample/
│   │   └── weather_sample.csv     # 3,000-hour curated slice for zero-setup CI testing
│   └── processed/                 # Aligned and curated datasets
├── docs/
│   ├── PROJECT_RULES.md           # Guardrails, no-leakage rules, and methodology
│   ├── RESEARCH_REPORT.md         # Comprehensive IEEE-format MS research report
│   └── DEFENSE_SLIDES_NOTES.md    # Speaking notes & examiner Q&A defense guide
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_manual_baseline.ipynb
│   └── 03_automated_pipeline_comparison.ipynb
├── results/
│   ├── eda_summary.md             # Automated audit of 45,253 raw sensor records
│   ├── rq1_forecast_accuracy.csv  # Benchmark comparison table
│   ├── rq1_ablation_study.csv     # Feature group ablation table
│   ├── rq2_imputation_benchmark.csv# Ground-truth missing data evaluation
│   └── plots/
│       ├── rq1_rmse_comparison.png
│       ├── rq1_ablation_study.png
│       └── rq2_imputation_rmse.png
├── src/
│   └── weatherpipe/
│       ├── __init__.py
│       ├── loader.py              # Multi-city loader and Kelvin-to-Celsius normalization
│       ├── eda.py                 # Automated data auditing module
│       ├── preprocessing.py       # Alignment, quality filtering, and causal diurnal imputer
│       ├── features.py            # Multi-scale lags, rolling stats, dew point, collinearity
│       ├── manual_baseline.py     # Naive and expert analyst manual baselines
│       ├── models.py              # XGBoost / Random Forest unified model interface
│       ├── evaluation.py          # Temporal CV folds, metrics, bootstrap CI
│       ├── experiment_rq1.py      # End-to-end RQ1 evaluation and ablation
│       ├── experiment_rq2.py      # RQ2 ground-truth imputation study
│       ├── accuracy_gate.py       # CI/CD automated deployment gate
│       ├── drift.py               # Kolmogorov-Smirnov drift detection
│       └── replay.py              # Continuous weekly replay simulation
├── tests/
│   ├── test_alignment.py          # Tests hourly grid regularity and deduplication
│   ├── test_cleaning.py           # Tests sentinel mapping and spike excision
│   ├── test_imputation.py         # Tests causal imputer and circular wind encoding
│   ├── test_features.py           # Tests lags, rolling windows, and collinearity pruning
│   └── test_leakage.py            # MATHEMATICAL PROOF: Zero future data leakage
├── Dockerfile                     # Containerized reproduction environment
├── Makefile                       # Convenient developer targets
├── pyproject.toml                 # Modern PEP 621 package build & tool configuration
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## ⚡ Quickstart & Reproducibility Guide

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/baidi-prob/DSTT-Project-Working.git
cd DSTT-Project-Working

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Run Quality Checks & Test Suite
```bash
# Code Quality Linting
ruff check src tests

# Run all 11 Unit, Causality, and Leakage Tests
pytest -v tests/
```

### 3. Run Experiments
```bash
# Run EDA Audit
python -m weatherpipe.eda --raw-dir data/raw --out results/eda_summary.md

# Run RQ1 Forecast Accuracy Benchmark across 4 temporal folds
python -m weatherpipe.experiment_rq1 --city "San Francisco"

# Run RQ2 Ground-Truth Imputation Fidelity Benchmark
python -m weatherpipe.experiment_rq2 --city "San Francisco"

# Evaluate CI/CD Accuracy Gate
python -m weatherpipe.accuracy_gate --results results/rq1_forecast_accuracy.csv

# Run Continuous Weekly Streaming Replay Simulation
python -m weatherpipe.replay --city "San Francisco" --weeks 4
```

---

## 🛡️ Scientific Guardrails & Leakage Prevention

1. **Strict Temporal Causality**: Feature vectors at timestamp $t$ only use data observed at or prior to $t$ ($\le t$). Lags use positive shifts (`shift(k)`); rolling windows use `closed='right'`.
2. **Train-Only Parameter Fitting**: Diurnal climatological lookup tables, feature pruning masks, and scalers are fitted **strictly on training partitions** and frozen for test sets.
3. **Mathematical Leakage Proof (`test_leakage.py`)**: Tests compute past features at $t \le T$, mutates future data at $t > T$ with extreme values (55°C heatwave, 890 hPa hurricane pressure), and asserts past features remain bit-for-bit identical (`pd.testing.assert_frame_equal`).

---

## 📚 References & Theoretical Citations

1. **A. Murray, D. Dervovic, and M. Cashmore**, "ELATE: LLM-Assisted Automated Feature Engineering for Time-Series Forecasting," *arXiv:2508.14667*, 2025.
2. **A. V. Malarkkan, W. Ying, and Y. Fu**, "CAFE: Causal Automated Feature Engineering for Robust Machine Learning Under Distribution Shifts," *arXiv:2602.16435*, 2026.
3. **C. Guillaume-Castel, C. Li, and S. Sobolowski**, "Deep Learning Approaches for Identifying Synoptic Meteorological Drivers," *EGUsphere [Preprint]*, DOI: 10.5194/egusphere-2026-3362, 2026.
4. **M. J. Kanter and K. Veeramachaneni**, "Deep Feature Synthesis: Towards Automating Data Science Endeavors," in *IEEE DSAA*, pp. 1–10, 2015. DOI: 10.1109/DSAA.2015.7344858.
5. **J. P. Noboa et al.**, "Feature Engineering and Group Selection for Urban Flood Prediction under Changing Climatic Regimes," *Water Resources Management*, vol. 38, pp. 4501–4519, 2024.
6. **M. Banerjee, R. Pal, and S. Ray**, "Automated Feature Extraction and Selection in Time-Domain Physiological and Mechanical Sensor Signals," *IEEE Trans. Cybernetics*, vol. 48, pp. 2410–2421, 2018.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
