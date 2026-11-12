# Automated Weather Data Preprocessing & Feature Engineering Pipeline
## Project Operational Rules & Scientific Guardrails

### Project Context
- **Title**: Automated Weather Data Preprocessing and Feature Engineering for Machine Learning: Improving Predictive Robustness
- **Course**: Data Science Tools and Techniques (DSTT) - MS Artificial Intelligence / Data Science
- **Authors**: Obaid Raja (26I-8034), Zoffishan Sattar (26I-8029), Hamza Ali (26I-8007)
- **Primary Objective**: Replace manual, ad-hoc, error-prone time-series weather preprocessing with a modular, automated, CI/CD and MLOps driven pipeline and rigorously demonstrate that it outperforms traditional manual workflows without data leakage.

### Research Questions
- **RQ1 (Predictive Accuracy & Engineering Gain)**: To what extent does an automated, domain-aware preprocessing and feature engineering pipeline improve the forecast accuracy (RMSE, MAE, R²) of downstream machine learning models over manual/ad-hoc preprocessing?
- **RQ2 (Imputation Fidelity & Bias Preservation)**: How effectively does a domain-aware, causal imputation mechanism (seasonal diurnal climatology + decaying anomaly) handle missing records compared to standard baselines (mean, forward-fill, linear interpolation) without introducing synthetic bias or future leakage?

### Fixed Architectural Decisions
1. **Target Task**: Forecast ambient temperature 6 hours ahead ($T_{t+6}$) on a regular 1-hour interval grid.
2. **Benchmark Dataset**: Kaggle Historical Hourly Weather Data (2012–2017) across 36 cities.
   - Primary City: **San Francisco** (coastal marine layer, microclimates, genuine sensor anomalies).
   - Robustness Validation Cities: **Chicago** (continental seasonal extremes) and **Phoenix** (arid desert diurnal swings).
3. **Primary Model**: XGBoost Regressor with fixed hyperparameters across all comparisons to isolate the effect of preprocessing and feature engineering.
4. **Baseline Comparisons**:
   - *Persistence Reference*: $T_{t+6} = T_t$ (sanity benchmark).
   - *Manual Baseline A (Naive)*: Sentinel filtering, column mean imputation, raw features.
   - *Manual Baseline B (Ad-hoc Analyst)*: Forward-fill, raw calendar features, 1 lag, 1 rolling mean.
   - *Proposed Automated Pipeline*: Temporal alignment, physical sanity bounds, spike filtering, causal diurnal climatology gap-filling, circular wind encodings ($\sin/\cos$), multi-scale lags (1, 2, 3, 6, 12, 24), rolling statistics (3h, 24h mean & std), cyclical encodings, meteorological interactions (dew point, 3h pressure tendency), and collinearity pruning.

### Strict Scientific Guardrails
1. **Strict Causality (No Future Peeking)**:
   - Any feature computed at timestamp $t$ may ONLY utilize observations $\le t$.
   - Rolling windows must end at timestamp $t$ (`closed='right'`).
   - Lags must strictly use positive shifts (`shift(k)` where $k \ge 1$).
2. **Train-Only Parameter Fitting**:
   - All scalers, climatological lookup tables, and drift baselines must be fit strictly on training splits.
   - Test splits are transformed using frozen parameters learned on training data.
3. **Temporal Cross-Validation**:
   - Never use randomized `train_test_split` with shuffling for time-series.
   - Use expanding-window temporal cross-validation (4 folds).
4. **Honest Reporting**:
   - Report actual metrics from real executions; do not hardcode or falsify results.
   - Report per-fold metrics, mean, and standard deviation / confidence intervals.
5. **Continuous Integration & Quality Gate**:
   - Automated testing verifies schema, range, causality, and temporal leakage.
   - Accuracy gate enforces that automated pipeline RMSE must be lower than manual baseline RMSE.

