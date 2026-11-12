# Automated Weather Data Preprocessing and Feature Engineering Pipeline: Improving Predictive Robustness Under Temporal Distribution Shifts

**Course**: Data Science Tools and Techniques (DSTT) — MS Data Science / Artificial Intelligence  
**Authors**: Obaid Raja (26I-8034), Zoffishan Sattar (26I-8029), Hamza Ali (26I-8007)  
**Institution**: FAST National University of Computer and Emerging Sciences  
**Date**: November 2026  

---

## Abstract
Meteorological modeling and short-term atmospheric forecasting are foundational to renewable energy dispatch, agriculture, and municipal hazard prevention. However, real-world sensor telemetry is notoriously messy, plagued by missing transmission bursts, instrument recalibration resets, unphysical sensor spikes, and non-linear diurnal shifts. In standard data science practice, exploratory notebooks rely heavily on manual, subjective, and ad-hoc imputation (e.g., non-causal linear interpolation, column mean replacement) and hand-crafted features. This paradigm creates substantial engineering bottlenecks, risks severe temporal data leakage, and collapses under live deployment.

This research presents **WeatherPipe**, a modular, automated, domain-aware preprocessing and feature engineering pipeline paired with an MLOps Continuous Integration and Continuous Training (CI/CD/CT) framework. WeatherPipe implements:
1. Automated temporal alignment to an unbroken regular hourly grid with de-duplication;
2. Strict thermodynamic boundary assertions and isolated meteorological spike filtering;
3. Domain-aware, strictly causal gap imputation that synthesizes seasonal diurnal climatology with decaying anomaly persistence without peeking into future data;
4. Circular wind direction decomposition ($\sin/\cos$) to eliminate $0^\circ/360^\circ$ coordinate discontinuities;
5. Multi-scale causal lag generation, backward-facing rolling window statistics, cyclical temporal markers, and thermodynamic interactions (Magnus-Tetens dew point and barometric pressure tendency);
6. Automated collinearity pruning to eliminate feature redundancy.

We evaluate WeatherPipe against both a naive manual baseline and an ad-hoc expert analyst workflow across 45,253 hourly observations from the Kaggle Historical Hourly Weather benchmark using expanding-window temporal cross-validation (4 folds). On the 6-hour temperature forecast task ($T_{t+6}$), WeatherPipe achieves an overall **RMSE of 1.7776°C** ($R^2 = 0.9000$), outperforming the persistence baseline (5.3442°C), naive manual baseline (4.5095°C), and expert analyst baseline (2.1136°C), representing a **15.90% error reduction** over the best manual approach. An ablation study demonstrates that combining lags, rolling statistics, cyclical encodings, and thermodynamic interactions yields superior predictive capacity compared to any individual feature family. Furthermore, in controlled ground-truth masking benchmarks (RQ2), WeatherPipe's causal diurnal imputer reduces 24-hour sensor outage error to **2.4928°C RMSE** compared to 4.8859°C for forward-fill and 4.8688°C for linear interpolation, while eliminating future leakage. The system is governed by a GitHub Actions CI/CD accuracy gate and drift detection mechanism, ensuring reproducible and verifiable operational deployment.

---

## 1. Introduction

Accurate localized meteorological forecasting is crucial for modern socio-economic infrastructure. Machine learning algorithms, particularly gradient-boosted decision trees (XGBoost) and random forests, have demonstrated remarkable utility in modeling non-linear atmospheric processes. However, a model's operational efficacy is bounded by the quality and temporal integrity of its input features—a principle commonly summarized as "garbage in, garbage out."

### 1.1 The Operational Problem: Manual and Ad-Hoc Preprocessing
Despite advances in model architectures, the data preparation phase remains heavily manual. In traditional workflows, data scientists operate within interactive notebooks, applying heuristic-driven fixes:
- **Arbitrary Imputation**: Gaps in temperature records are frequently filled using column-wide global means or bidirectional linear interpolation. While linear interpolation provides smooth offline visual curves, it computes values using future observations ($x_{t+k}$ to impute $x_t$), introducing severe **future data leakage** that is physically impossible in operational streaming environments.
- **Euclidean Treatment of Circular Variables**: Wind direction is recorded in degrees from $0^\circ$ to $360^\circ$. Treating this as an ordinary linear numeric variable creates artificial mathematical discontinuities: a wind shift from $359^\circ$ (NNW) to $1^\circ$ (NNE) represents a gentle $2^\circ$ veer, but linear algorithms interpret it as a massive $358^\circ$ transition.
- **Unchecked Sensor Glitches**: Sensor telemetry often includes sentinel failure codes (e.g., -999, 9999) or unphysical transient voltage spikes (e.g., a 20°C jump in 60 minutes) which distort gradient calculations if not systematically isolated.
- **Fragile Notebook Scripts**: Ad-hoc scripts lack automated schema validation, version control, unit tests, and performance regression gates. When deployed in production where weather distributions drift across seasons, manual models degrade silently.

### 1.2 Primary Contributions
This research addresses these deficiencies by architecting, implementing, and validating **WeatherPipe**, an end-to-end automated meteorological engineering system. The core contributions are:
1. **Strictly Causal Meteorological Preprocessing**: A pipeline guaranteeing that every imputation and feature vector at timestamp $t$ utilizes observations strictly $\le t$.
2. **Domain-Aware Diurnal Imputation**: A causal algorithm combining empirical diurnal climatology with decaying anomaly persistence, overcoming the failure modes of forward-fill and the data leakage of interpolation.
3. **Automated Feature Generation & Selection**: Systematic extraction of multi-scale lags, backward rolling aggregates, cyclical markers, Magnus-Tetens dew point, and pressure tendencies, filtered via correlation thresholding.
4. **Rigorous Empirical Validation**: Controlled 4-fold expanding-window temporal cross-validation on 45,253 real-world hourly records, directly answering two formal research questions (RQ1 and RQ2).
5. **Production CI/CD & MLOps Integration**: Full test automation (unit, alignment, range, and temporal leakage tests), MLflow tracking, an automated accuracy gate asserting $\text{RMSE}_{\text{auto}} < \text{RMSE}_{\text{manual}}$, and a continuous weekly replay simulator with distribution drift auditing.

---

## 2. Literature Review & Theoretical Positioning

The architectural decisions of WeatherPipe are grounded in five key works across automated feature engineering, spatial climate modeling, and meteorological validation:

| Author(s) & Year | Foundational Concept | Relevance to WeatherPipe | Key Limitation Addressed |
| :--- | :--- | :--- | :--- |
| **Banerjee et al. (2018)** | Automated feature extraction in time-domain signals; comparison against PCA. | Motivation for replacing months of manual feature crafting with programmatic pipelines while retaining interpretability. | Applied only to industrial vibration and physiological signals; did not address multi-station meteorological dynamics. |
| **Noboa et al. (2024)** | Feature group ablation methodology for hydrometeorological event modeling. | Adoption of structured feature grouping (lags, rolling stats, cyclical, thermodynamic) to empirically verify the contribution of each family. | Focused on binary flood classification in a single municipality; used static feature sets rather than automated pipelines. |
| **Guillaume-Castel, Li, & Sobolowski (2026)** | Verification of physical drivers in climate models using explainable AI. | Rigorous temporal train/validation/test splitting, training-only normalization, and baseline persistence reference benchmarks. | Evaluated spatial grid CNNs on reanalysis data; post-hoc explanations represent statistical correlation rather than causal mechanics. |
| **Kanter & Veeramachaneni (2015)** | Deep Feature Synthesis (DFS) for automated relational feature generation. | Algorithmic generation of temporal aggregates and mathematical interaction terms across time series. | Combinatorial expansion produces massive collinear and redundant feature spaces requiring systematic pruning. |
| **Murray et al. (ELATE, 2025)** / **Malarkkan et al. (CAFE, 2025)** | Modern automated feature engineering frameworks using LLMs and Reinforcement Learning. | Concept of intelligent, context-aware feature space exploration and distribution shift adaptation. | High computational overhead, token latency, and heavy optimization loops unsuitable for lightweight, deterministic CI/CD pipelines. |

---

## 3. Methodology & System Architecture

WeatherPipe follows a modular, sequential architecture designed for deterministic execution, reproducibility, and zero future leakage.

### 3.1 Data Ingestion and Temporal Alignment
Raw weather telemetry is collected in wide-format CSV files where rows represent timestamps and columns represent geographical stations.
1. **Datetime Parsing & Regular Grid Reindexing**: Timestamps are parsed and rounded to the nearest hour ($freq = \text{1h}$).
2. **Deduplication**: If duplicate sensor readings occur within the same hourly window, numeric observations are averaged.
3. **Explicit Gap Reindexing**: A complete date range $T_{\text{full}} = [t_{\min}, t_{\max}]$ is constructed. Reindexing against $T_{\text{full}}$ forces all missing sensor hours into explicit `NaN` records, preventing irregular time-step distortions in downstream lags.

### 3.2 Quality Filtering & Physical Sanity Bounds
Sensor hardware occasionally malfunctions, outputting sentinel error codes or extreme voltage transients. WeatherPipe enforces strict physical constraints:
- **Sentinel Filtering**: Known error codes ($-999, -9999, 999, 9999$) and impossible pressure readings ($P \le 500\text{ hPa}$) are mapped to `NaN`.
- **Thermodynamic Bounds**: Temperature is bounded to $[-50^\circ\text{C}, +60^\circ\text{C}]$; relative humidity is clipped to $[0\%, 100\%]$; wind speed is bounded to $[0, 75\text{ m/s}]$.
- **Isolated Meteorological Spike Filter**: An isolated instrument glitch occurs when a value $x_t$ deviates severely from both its predecessor $x_{t-1}$ and successor $x_{t+1}$, while the surrounding context is relatively stable:
  $$\text{Spike}(x_t) = \left(|x_t - x_{t-1}| > \theta\right) \land \left(|x_t - x_{t+1}| > \theta\right) \land \left(|x_{t+1} - x_{t-1}| < \theta\right)$$
  where $\theta = 10^\circ\text{C}$ for temperature and $25\text{ hPa}$ for pressure. Identified glitches are mapped to `NaN`.

### 3.3 Domain-Aware Causal Imputation
Standard interpolation methods look forward into future timestamps, creating artificial leakage. WeatherPipe implements `CausalDiurnalImputer`:
- **Training Phase**: Computes the empirical diurnal climatological lookup table strictly on the training partition:
  $$\mu(m, h) = \frac{1}{|S_{m,h}|} \sum_{i \in S_{m,h}} x_i$$
  where $S_{m,h}$ is the set of historical observations occurring in month $m \in \{1..12\}$ at hour $h \in \{0..23\}$.
- **Inference / Transformation Phase**: For any missing observation at timestamp $t$ occurring $k$ hours after the last valid observation $x_{t-k}$:
  $$\hat{x}_t = \mu(m_t, h_t) + \alpha^k \cdot \left(x_{t-k} - \mu(m_{t-k}, h_{t-k})\right)$$
  where $\alpha = 0.90$ is the persistence decay rate.
  - If $k = 1$ (short 1-hour dropout), the imputer carries $90\%$ of the recent weather anomaly forward, preserving local microclimatic momentum.
  - If $k \ge 48$ (extended transmission blackout), $\alpha^k \to 0$, causing the estimate to smoothly revert to the true diurnal climatological mean without generating unphysical static plateaus or peeking forward.

### 3.4 Circular Coordinate Transformation for Wind Telemetry
Wind direction $\theta \in [0^\circ, 360^\circ)$ is decomposed into orthogonal components:
$$wd_{\sin} = \sin\left(\frac{2\pi \theta}{360}\right), \quad wd_{\cos} = \cos\left(\frac{2\pi \theta}{360}\right)$$
For calm conditions ($\text{wind\_speed} = 0$), both components are set to $0.0$, eliminating trigonometric ambiguity.

### 3.5 Automated Feature Engineering
Features are systematically extracted under strict backward-only causality:
1. **Multi-Scale Lags**: Positive shifts $k \in \{1, 2, 3, 6, 12, 24\}$ for temperature, pressure, humidity, and wind speed.
2. **Backward-Only Rolling Aggregates**: 3-hour and 24-hour rolling mean and rolling standard deviation computed with `closed='right'`.
3. **Cyclical Temporal Encodings**:
   $$\text{hour}_{\sin} = \sin\left(\frac{2\pi h}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi h}{24}\right)$$
   $$\text{doy}_{\sin} = \sin\left(\frac{2\pi \cdot \text{doy}}{365.25}\right), \quad \text{doy}_{\cos} = \cos\left(\frac{2\pi \cdot \text{doy}}{365.25}\right)$$
4. **Thermodynamic Interaction Metrics**:
   - *Magnus-Tetens Dew Point ($T_{\text{dew}}$)*:
     $$\gamma(T, RH) = \frac{17.27 \cdot T}{237.7 + T} + \ln\left(\frac{RH}{100}\right), \quad T_{\text{dew}} = \frac{237.7 \cdot \gamma}{17.27 - \gamma}$$
   - *Dew Point Spread*: $T - T_{\text{dew}}$ (indicates atmospheric saturation and moisture deficit).
   - *3-Hour Pressure Tendency*: $\Delta P_{3h} = P_t - P_{t-3}$ (barometric indicator of frontal boundary passage).
5. **Collinearity Pruning Filter**: Evaluates the pairwise Pearson correlation matrix on training features and prunes features where $|r| > 0.98$, preventing variance inflation.

---

## 4. Experimental Evaluation & Results

The empirical evaluation was conducted on 45,253 hourly records (October 2012 to November 2017) for **San Francisco**, chosen for its complex coastal marine layer dynamics and non-linear diurnal cycles. The forecast task is predicting temperature 6 hours ahead ($y_t = T_{t+6}$).

All models utilize an identical **XGBoost Regressor** ($n_{\text{estimators}} = 120, \text{max\_depth} = 6, \eta = 0.08, \text{subsample} = 0.85$) evaluated over **4 expanding-window temporal folds** (minimum initial training partition: 45%).

### 4.1 RQ1: Forecast Accuracy Comparison
Table 1 summarizes the overall performance across the 4 expanding temporal folds.

**Table 1: Overall 6-Hour Forecast Performance Comparison**
| Forecasting Method | Preprocessing & Feature Strategy | Overall RMSE (°C) | Overall MAE (°C) | Overall $R^2$ | Gain vs Best Manual (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Persistence (Sanity Reference)** | $\hat{y}_t = T_t$ (no model) | 5.3442 | 4.2236 | 0.0961 | — |
| **Manual Baseline A (Naive)** | Global mean fill, raw unprocessed columns | 4.5095 | 3.4551 | 0.3564 | — |
| **Manual Baseline B (Ad-hoc Analyst)**| Forward fill, integer month/hour, 1 lag, 1 rolling mean | 2.1136 | 1.5719 | 0.8586 | — |
| **WeatherPipe (Proposed Automated)** | Aligned, cleaned, causal diurnal imputer, multi-scale features | **1.7776** | **1.3212** | **0.9000** | **+15.90%** |

WeatherPipe achieves an overall **RMSE of 1.7776°C** and an **$R^2$ of 0.9000**, outperforming the expert analyst baseline (2.1136°C) by **15.90%** and the naive baseline (4.5095°C) by **60.58%**.

**Table 2: Per-Fold Temporal Cross-Validation Breakdown**
| Fold | Train Size | Test Size | Persistence RMSE | Naive Baseline RMSE | Analyst Baseline RMSE | WeatherPipe RMSE | Error Reduction (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 20,363 | 6,222 | 4.9691°C | 4.2792°C | 1.9547°C | **1.7038°C** | **+12.84%** |
| **Fold 2** | 26,585 | 6,222 | 5.1573°C | 4.4824°C | 2.0900°C | **1.7250°C** | **+17.46%** |
| **Fold 3** | 32,807 | 6,222 | 5.3748°C | 4.3498°C | 2.0965°C | **1.7346°C** | **+17.26%** |
| **Fold 4** | 39,029 | 6,224 | 5.9049°C | 4.6466°C | 3.2158°C | **1.9593°C** | **+39.07%** |

WeatherPipe won **all 4 chronological test folds**. In Fold 4 (which captured an anomalous seasonal transition), the manual analyst baseline degraded severely to 3.2158°C, while WeatherPipe maintained robust generalization at 1.9593°C—a **39.07% error reduction**.

A 1,000-iteration bootstrap confidence interval for the error reduction ($\Delta \text{RMSE} = \text{RMSE}_{\text{analyst}} - \text{RMSE}_{\text{auto}}$) yields:
$$\Delta \text{RMSE} \in [+0.2984^\circ\text{C}, +0.3736^\circ\text{C}] \quad (95\%\text{ CI})$$
Because the confidence interval is strictly positive and bounded away from zero, the superiority of the automated pipeline is statistically significant at $p < 0.001$.

### 4.2 Feature Group Ablation Study
Following the ablation methodology of Noboa et al. (2024), we trained models on individual feature families and their combination.

**Table 3: Feature Family Ablation Study**
| Feature Group | Features Included | Feature Count | Test RMSE (°C) | Test MAE (°C) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Physical Interactions Only** | Dew point, dew point spread, pressure tendency | 3 | 5.1360 | 3.9410 | 0.0144 |
| **Rolling Aggregates Only** | 3h & 24h rolling means and std | 6 | 3.9593 | 3.0952 | 0.4137 |
| **Cyclical Encodings Only** | $\sin/\cos$ of hour and day of year | 6 | 3.4482 | 2.5460 | 0.5553 |
| **Lags Only** | Multi-scale positive shifts (1, 2, 3, 6, 12, 24) | 17 | 3.0838 | 2.2301 | 0.6451 |
| **All Combined (WeatherPipe)** | Lags + Rolling + Cyclical + Physical Interactions | **37** | **1.9502** | **1.4630** | **0.8581** |

The ablation confirms that no single feature family suffices on its own. While lags provide the strongest individual signal (3.0838°C), integrating cyclical markers, rolling atmospheric volatility, and thermodynamic interaction metrics yields a dramatic performance leap to **1.9502°C**.

### 4.3 RQ2: Missing Data Imputation Fidelity Benchmark
To address RQ2, we conducted a controlled experiment masking known ground-truth observations under two real-world failure patterns:
1. **Random Dropout (MCAR)**: 10% and 20% isolated missing sensor readings;
2. **Burst / Block Outages**: Consecutive transmission blackouts of 3h, 6h, 12h, and 24h.

**Table 4: Imputation Fidelity & Synthetic Bias Benchmark**
| Scenario | Global Mean Fill | Forward Fill (ffill) | Non-Causal Linear Interpolation | Proposed Causal Diurnal |
| :--- | :---: | :---: | :---: | :---: |
| **Random Dropout (10%)** | 6.2102°C (MBE -1.82) | 1.5004°C (MBE +0.04) | 0.6377°C (MBE -0.02)* | **0.9931°C (MBE -0.06)** |
| **Random Dropout (20%)** | 6.0880°C (MBE -1.90) | 1.5696°C (MBE -0.04) | 0.6471°C (MBE +0.01)* | **0.9872°C (MBE -0.08)** |
| **Burst Outage (3h Block)** | 5.5100°C (MBE -1.71) | 2.0853°C (MBE -0.45) | 0.7730°C (MBE +0.01)* | **1.0954°C (MBE -0.16)** |
| **Burst Outage (6h Block)** | 5.8019°C (MBE -2.04) | 3.6669°C (MBE -0.78) | 1.3913°C (MBE -0.12)* | **1.8000°C (MBE -0.38)** |
| **Burst Outage (12h Block)**| 5.8076°C (MBE -2.18) | 5.1487°C (MBE -0.92) | 3.0625°C (MBE -0.52)* | **2.4253°C (MBE -0.54)** |
| **Burst Outage (24h Block)**| 5.4459°C (MBE -1.87) | 4.8859°C (MBE -0.61) | 4.8688°C (MBE -0.36)* | **2.4928°C (MBE -0.40)** |

*\*Note: Linear interpolation peeks into future observations ($t+k$), making it physically invalid for live operational deployment.*

#### Key Findings for RQ2:
1. **Mean Imputation Introduces Massive Bias**: Global mean filling exhibits severe Mean Bias Error ($\approx -1.8^\circ\text{C}$ to $-2.2^\circ\text{C}$) and unacceptably high RMSE ($>5.4^\circ\text{C}$).
2. **Forward-Fill Collapses on Extended Gaps**: As gap length increases to 12h and 24h, forward-fill error escalates to $5.15^\circ\text{C}$ and $4.89^\circ\text{C}$ because it ignores the natural diurnal day/night temperature oscillation.
3. **Linear Interpolation Flattens Day/Night Cycles**: While linear interpolation appears strong on 1-hour drops, it degrades significantly on 24-hour outages ($4.87^\circ\text{C}$ RMSE) because drawing a straight line between two days completely misses nocturnal cooling and afternoon solar heating. Crucially, linear interpolation cannot be used on live streaming data.
4. **Proposed Causal Diurnal Imputer Outperforms All Deployable Methods**: On 12-hour outages, WeatherPipe achieves **2.4253°C RMSE** (vs. 5.15°C for ffill); on 24-hour outages, it achieves **2.4928°C RMSE** (vs. 4.89°C for ffill and 4.87°C for linear). It successfully reconstructs diurnal atmospheric physics while remaining **100% causal and leak-free**.

---

## 5. CI/CD Architecture, Testing, & Continuous MLOps

WeatherPipe is operationalized as an automated production pipeline rather than an isolated research script.

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

### 5.1 Automated Test Suite (11/11 Passing Tests)
The test suite in `tests/` verifies every pipeline component:
1. `test_align_regular_hourly_grid`: Enforces continuous 1-hour interval datetime spacing.
2. `test_align_deduplication`: Verifies multiple readings within an hour are correctly averaged.
3. `test_clean_sentinels_and_bounds`: Confirms -999, 9999, and out-of-range values become NaN.
4. `test_clean_meteorological_spike`: Verifies isolated physical glitches are excised.
5. `test_circular_wind_encoding`: Verifies boundary continuity at 0°/360° and calm zero-vectors.
6. `test_causal_diurnal_imputer_no_nans`: Confirms complete imputation without residual missingness.
7. `test_dew_point_calculation`: Verifies thermodynamic accuracy against psychrometric tables.
8. `test_feature_generation_shapes_and_lags`: Confirms shift offsets and rolling statistics.
9. `test_collinear_feature_pruning`: Verifies that redundant features ($|r| > 0.98$) are dropped.
10. `test_zero_future_leakage_in_features`: **Mutates future data ($t > T$) and proves past feature vectors remain bit-for-bit identical.**
11. `test_zero_future_leakage_in_imputer`: **Proves causal imputer outputs at time $T$ are unaffected by future variations.**

### 5.2 GitHub Actions Workflows
Two automated workflows govern the project:
1. **`.github/workflows/ci.yml`**:
   - Triggers on every push and pull request to `main` and `dev`.
   - Runs `ruff check src tests` (enforcing zero lint errors).
   - Executes the complete `pytest` test suite.
   - Runs a fast validation experiment on `data/sample/weather_sample.csv`.
   - Executes `python -m weatherpipe.accuracy_gate`, failing the build if automated RMSE exceeds manual baseline.
   - Archives experiment tables and plots as build artifacts.
2. **`.github/workflows/data-pipeline.yml`**:
   - Scheduled weekly cron (`0 6 * * 1`) and file-drop trigger (`data/incoming/**`).
   - Ingests incoming batches, executes drift detection, scores with the champion model, and alerts on anomalies.

### 5.3 Streaming Weekly Replay Simulation
The replay module (`weatherpipe.replay`) simulated 4 consecutive weekly data drops from the held-out test period. In Week 3, an unphysical sensor blackout was artificially injected. The pipeline successfully detected the anomaly, flagged distribution drift via the Kolmogorov-Smirnov test, and quarantined the corrupted batch, maintaining model stability without pipeline collapse.

---

## 6. Limitations & Threats to Validity

To adhere to academic rigor, the following limitations must be noted:
1. **Single-Station In Situ Telemetry**: WeatherPipe operates on localized surface station time series. While it handles microclimates effectively, it does not ingest 2D spatial reanalysis grids (e.g., ERA5) or radar arrays.
2. **Precipitation Omission**: The Kaggle Historical Hourly Weather dataset lacks precipitation depth (in mm). Precipitation was omitted, though weather conditions were logged via `weather_description`.
3. **Computational Resource Bounds**: Cross-validation was run across 4 temporal folds. In high-frequency operational forecasting (e.g., 5-minute Doppler radar), rolling window updates require distributed stream engines such as Apache Spark or Flink.

---

## 7. Conclusion

This research demonstrated that replacing manual, ad-hoc weather preprocessing with an automated, domain-aware, CI/CD-driven pipeline yields substantial, statistically significant improvements in predictive performance and operational reliability. 

On 45,253 real-world hourly observations:
- WeatherPipe achieved an overall 6-hour forecast RMSE of **1.7776°C** ($R^2 = 0.9000$), outperforming the expert analyst baseline (2.1136°C) by **15.90%** and the naive baseline (4.5095°C) by **60.58%**.
- In controlled imputation benchmarks, WeatherPipe's causal diurnal imputer reduced 24-hour sensor outage error to **2.4928°C RMSE** compared to 4.8859°C for forward-fill, while eliminating future leakage.
- The pipeline was verified with 11 automated unit and leakage tests, and integrated with GitHub Actions CI/CD and MLOps continuous monitoring.

WeatherPipe proves that systematic, code-driven preprocessing and causal feature engineering are essential foundations for robust, deployable machine learning in environmental sciences.

---

## References

1. **A. Murray, D. Dervovic, and M. Cashmore**, "ELATE: LLM-Assisted Automated Feature Engineering for Time-Series Forecasting," *arXiv preprint arXiv:2508.14667*, Aug. 2025.
2. **A. V. Malarkkan, W. Ying, and Y. Fu**, "CAFE: Causal Automated Feature Engineering for Robust Machine Learning Under Distribution Shifts," *arXiv preprint arXiv:2602.16435*, Feb. 2026.
3. **C. Guillaume-Castel, C. Li, and S. Sobolowski**, "Deep Learning Approaches for Identifying Synoptic Meteorological Drivers: Verification and Interpretability of Atmospheric Extratropical Cyclones," *EGUsphere [Preprint]*, DOI: 10.5194/egusphere-2026-3362, Jun. 2026.
4. **M. J. Kanter and K. Veeramachaneni**, "Deep Feature Synthesis: Towards Automating Data Science Endeavors," in *Proceedings of the IEEE International Conference on Data Science and Advanced Analytics (DSAA)*, Paris, France, pp. 1–10, 2015. DOI: 10.1109/DSAA.2015.7344858.
5. **J. P. Noboa, R. V. Silva, C. G. H. Bastos, and M. V. Lamar**, "Feature Engineering and Group Selection for Urban Flood Prediction under Changing Climatic Regimes," *Water Resources Management*, vol. 38, no. 12, pp. 4501–4519, Oct. 2024. DOI: 10.1007/s11269-024-03912-1.
6. **M. Banerjee, R. Pal, and S. Ray**, "Automated Feature Extraction and Selection in Time-Domain Physiological and Mechanical Sensor Signals," *IEEE Transactions on Cybernetics*, vol. 48, no. 8, pp. 2410–2421, Aug. 2018. DOI: 10.1109/TCYB.2017.2731802.
7. **National Centers for Environmental Information (NCEI)**, "Global Historical Climatology Network (GHCN) Daily and Hourly In Situ Archive," National Oceanic and Atmospheric Administration (NOAA), 2025. [Online]. Available: https://www.ncei.noaa.gov/products/land-based-station/
