# Automating Weather Data Preprocessing and Feature Engineering in a CI/CD Pipeline for Short-Term Temperature Forecasting

**Authors**:
- **Obaid Raja** (`26I-8034`, `i268034@nu.edu.pk`) — Dept. of Computer Science, FAST-NUCES, Islamabad
- **Zoffishan Sattar** (`26I-8029`, `i268029@nu.edu.pk`) — Dept. of Computer Science, FAST-NUCES, Islamabad
- **Hamza Ali** (`26I-8007`, `i268007@nu.edu.pk`) — Dept. of Computer Science, FAST-NUCES, Islamabad

**Course**: Data Science Tools and Techniques (DSTT) — Fall 2026
**GitHub Implementation**: `https://github.com/baidi-prob/DSTT-Project-Working`

---

## Abstract
When working with real-world weather station data, most of the time goes into cleaning messy records rather than training the actual model. Sensor logs almost always have missing hours, sudden hardware spikes, broken error codes like `-999`, and unit mismatches. In normal practice, people handle these steps manually inside a Jupyter notebook by running simple forward-fill or linear interpolation, adding a couple of lag columns by hand, and training a model. While that works for a one-time assignment, it breaks down when new data arrives every week, and methods like linear interpolation actually leak future values into past rows.

In this project, we built **WeatherPipe**, an automated preprocessing and feature engineering pipeline connected to a full GitHub Actions CI/CD workflow. Instead of manual notebook edits, our pipeline automatically aligns timestamps to a regular hourly grid, filters out physical sensor glitches, converts wind direction into sine and cosine components, and fills missing gaps using only past data by combining historical hourly averages with recent temperature trends. It also generates multi-scale lags, rolling window statistics, dew point, and pressure changes over time, while dropping duplicate correlated features. We tested our setup on 45,253 hourly records from San Francisco (2012–2017) using 4 time-ordered cross-validation folds to predict temperature 6 hours ahead ($T_{t+6}$) with XGBoost. Our automated pipeline brought the overall RMSE down to **1.7776°C** ($R^2 = 0.9000$), compared to **2.1136°C** for a careful manual analyst baseline (a **15.90% improvement**) and **4.5095°C** for a basic mean-fill baseline. We also tested our gap-filling method by hiding real data for up to 24 hours at a time, where our approach reached **2.4928°C** RMSE compared to **4.8859°C** for forward-fill. Finally, we added 11 automated pytest checks—including a test that proves no future data leaks into past features—and an accuracy gate that stops the CI/CD build if a new update performs worse than the baseline.

---

## 1. Introduction
Weather forecasting models depend heavily on how clean the input data is. Whether the goal is predicting local temperature for agriculture, managing power grid loads, or tracking storms, raw weather station logs are rarely ready to be fed straight into a machine learning model [1]. If you open a raw CSV file from a weather station, you immediately run into missing timestamps, duplicate readings in the same hour, impossible numbers when a sensor resets, and temperature recorded in Kelvin instead of Celsius.

### 1.1 Why the Usual Manual Workflow is a Problem
In most student projects and even many practical workflows, data preprocessing is done manually. Someone opens a notebook, looks at a few plots, and writes quick fixes. After reviewing how weather time-series data is usually prepared, we noticed three main problems with doing this by hand:
1. **Future Data Leakage During Imputation**: When there is a gap in temperature readings, the most common quick fix in pandas is `df.interpolate(method='linear')`. On a graph, this looks great because it draws a straight line between the point before the gap ($t-1$) and the point after the gap ($t+k$). The problem is that in a real forecasting setup, you are standing at time $t$ trying to predict the future—you do not have the reading at $t+k$ yet. Using linear interpolation during preprocessing lets the model peek at future weather trends, which gives fake high accuracy during testing that disappears in real use.
2. **Treating Wind Direction Like a Normal Number**: Wind direction is measured in degrees from $0^\circ$ to $360^\circ$. If the wind shifts slightly from $359^\circ$ to $1^\circ$, it barely moved at all. But if you pass that raw column into a regression tree or neural network, the model sees a huge jump of 358 units. Without converting angles properly, the model learns the wrong splits.
3. **No Automation When New Data Arrives**: A manual notebook is static. If next week's or next month's weather log comes in with a broken sensor column or a shift in weather patterns, someone has to manually re-run all the cells and check if the model is still accurate. There are no unit tests to catch bugs and no automated checks to block a bad model from being deployed.

### 1.2 What We Built in This Project
Our goal in this project was simple: take all these manual preprocessing and feature engineering steps and turn them into a clean, repeatable, automated CI/CD pipeline that actually beats the manual approach in a fair test:
- **Causal Preprocessing**: Every cleaning and gap-filling step at hour $t$ only uses data from hour $t$ or earlier. Nothing from the future is ever touched.
- **Daily-Cycle Gap Filling**: We wrote an imputer that learns the normal temperature for each month and hour of the day from the training set, and blends that with how warm or cold the weather was just before the sensor went down.
- **Fair Comparison on Real Data**: We ran both the manual baselines and our automated pipeline on the exact same XGBoost model and the exact same 4 time-ordered test splits across 45,253 hours of real weather data.
- **Complete CI/CD Automation**: We connected the repo to GitHub Actions so every code push or new data batch runs linter checks, 11 unit and leakage tests, and an accuracy gate that fails the build if our pipeline's error goes above the manual baseline.

---

## 2. Literature Review and Research Gap
Before building our pipeline, we reviewed several papers that we also discussed in our project proposal and defense slides:

| Paper | What They Did | Dataset Used | What We Took From It | Gap We Address |
| :--- | :--- | :--- | :--- | :--- |
| **Murray et al. (ELATE, 2025)** [2] | Used LLMs to write feature engineering code for time series. | Financial and industrial time-series benchmarks. | Using walk-forward validation with XGBoost to measure real gains. | Needs heavy LLM API calls and token costs; overkill for a fast, local weather pipeline. |
| **Malarkkan et al. (CAFE, 2026)** [3] | Used Reinforcement Learning to pick features that survive distribution shifts. | Standard tabular ML benchmarks. | Checking how features hold up when data patterns shift over time. | Very slow RL training loops; not designed for hourly weather station gaps. |
| **Guillaume-Castel et al. (2026)** [4] | Checked if neural networks actually learn physical storm patterns using LRP. | ERA5 spatial climate grids. | Strict chronological train/test splits, fitting stats on train only, and persistence baselines. | Built for massive 2D satellite/reanalysis grids rather than single-station CSV logs. |
| **Kanter & Veeramachaneni (DFS, 2015)** [5] | Created Deep Feature Synthesis to auto-generate features across tables. | Multi-table relational datasets (KDD Cup). | Rule-based generation of rolling stats and lags without human coding. | Creates way too many duplicate, highly correlated features that need pruning. |
| **Noboa et al. (2024)** [6] | Tested different feature groups with Random Forest for flood prediction. | Rain, elevation, and river data in Curitiba. | Testing feature groups one by one (ablation) to see which group actually helps. | Features were still prepared manually for one specific flood study. |
| **Banerjee et al. (2018)** [7] | Automated time and frequency feature extraction for sensor signals. | Bearing vibration and blood pressure signals. | Showing that automation saves months of manual work while keeping accuracy high. | Tested on mechanical and medical sensors, not weather time series. |
| **WeatherPipe (Our Work)** | **Automated causal weather cleaning, feature pipeline, and CI/CD gate.** | **Kaggle Hourly Weather Data (45,253 rows).** | **Combines group ablation, strict time splits, physical weather features, and pruning.** | **A lightweight, tested CI/CD pipeline built specifically for station time series.** |

---

## 3. How Our Pipeline Works (Methodology)

### 3.1 Dataset and Initial Data Audit (EDA)
We used the *Historical Hourly Weather Data (2012–2017)* dataset from Kaggle, which has 45,253 hourly records across 36 cities in separate CSV files (temperature, humidity, pressure, wind speed, wind direction, and weather description). Before writing the pipeline, we wrote an EDA script (`src/weatherpipe/eda.py`) to check what was actually wrong with the raw files:
- Temperature was stored in Kelvin (around $285\text{ K}$), so we converted it to Celsius ($T_C = T_K - 273.15$).
- Every city had missing hours (for example, San Francisco had around $1.77\%$ missing temperature rows, including an initial block of missing values at the start of the recording).
- Pressure had occasional drop-outs down to $0\text{ hPa}$ or below $500\text{ hPa}$, which is physically impossible at ground level and happens when a barometer resets.

### 3.2 Step 1: Time Alignment and Cleaning
In `preprocessing.py`, our first function (`align_time_series`) parses the timestamps, rounds them to the nearest hour, and averages any duplicate entries that landed in the same hour. Then it builds a complete hourly calendar from the first timestamp to the last and reindexes the table.

Next, `clean_physical_ranges` replaces known error codes (`-999`, `9999`, and pressures $\le 500\text{ hPa}$) with `NaN`, removes impossible values, and runs a spike filter that marks single-hour hardware jumps ($>10^\circ\text{C}$ spike that immediately returns to normal the next hour) as `NaN`.

### 3.3 Step 2: Causal Gap Filling (Imputation)
To fill missing values without peeking into the future, we built `CausalDiurnalImputer`. When we fit the imputer on the training split, it calculates the average value $\mu(m, h)$ for every month $m \in \{1..12\}$ and hour of the day $h \in \{0..23\}$. When filling a missing value at time $t$ (which is $k$ hours after the last known reading $x_{t-k}$), we take the normal seasonal average for that hour $\mu(m_t, h_t)$ and add the recent weather difference, decayed by $\alpha = 0.90$ per hour:
$$\hat{x}_t = \mu(m_t, h_t) + \alpha^k \cdot \left(x_{t-k} - \mu(m_{t-k}, h_{t-k})\right)$$

### 3.4 Step 3: Circular Wind and Automated Features
For wind direction $\theta \in [0^\circ, 360^\circ)$, we convert the angle into sine and cosine before doing anything else. Then `FeatureEngineer` generates four groups of features using only past data ($\le t$):
1. **Lag Features**: Past values at $1, 2, 3, 6, 12,$ and $24$ hours ago.
2. **Rolling Window Stats**: Backward-looking 3-hour and 24-hour rolling mean and standard deviation.
3. **Cyclical Time Features**: Sine and cosine of the hour of day and day of year.
4. **Physical Weather Interactions**: Dew Point ($T_{\text{dew}}$) using the Magnus-Tetens formula, dew point spread ($T - T_{\text{dew}}$), and 3-hour pressure change ($\Delta P_{3h} = P_t - P_{t-3}$).

Finally, our pipeline checks the correlation matrix on the training set and drops any feature with $>0.98$ correlation with another feature.

---

## 4. Experiments and Results

### 4.1 RQ1: Does the Automated Pipeline Beat Manual Preprocessing?
We evaluated 6-hour ahead temperature forecasting ($T_{t+6}$) across 4 time-ordered expanding folds using the exact same XGBoost Regressor:

| Method | RMSE (°C) | MAE (°C) | $R^2$ | Gain vs. Analyst |
| :--- | :---: | :---: | :---: | :---: |
| Persistence (Sanity Check) | 5.3442 | 4.2236 | 0.0961 | — |
| Manual Baseline A (Naive) | 4.5095 | 3.4551 | 0.3564 | — |
| Manual Baseline B (Analyst) | 2.1136 | 1.5719 | 0.8586 | — |
| **WeatherPipe (Ours)** | **1.7776** | **1.3212** | **0.9000** | **+15.90%** |

#### Fold-by-Fold Breakdown
| Fold | Train / Test Rows | Naive (°C) | Analyst (°C) | Ours (°C) | Improvement |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 20,363 / 6,222 | 4.2792 | 1.9547 | **1.7038** | +12.84% |
| Fold 2 | 26,585 / 6,222 | 4.4824 | 2.0900 | **1.7250** | +17.46% |
| Fold 3 | 32,807 / 6,222 | 4.3498 | 2.0965 | **1.7346** | +17.26% |
| Fold 4 | 39,029 / 6,224 | 4.6466 | 3.2158 | **1.9593** | +39.07% |

### 4.2 Feature Group Ablation Study
| Feature Group | Features | RMSE (°C) | MAE (°C) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| Physical Interactions Only | 3 | 5.1360 | 3.9410 | 0.0144 |
| Rolling Stats Only | 6 | 3.9593 | 3.0952 | 0.4137 |
| Cyclical Time Only | 6 | 3.4482 | 2.5460 | 0.5553 |
| Lag Features Only | 17 | 3.0838 | 2.2301 | 0.6451 |
| **All Groups Combined (Ours)** | **37** | **1.9502** | **1.4630** | **0.8581** |

### 4.3 RQ2: Missing Data Imputation Benchmark
| Missing Data Pattern | Mean Fill | Forward Fill | Linear Interp.* | Ours (Causal) |
| :--- | :---: | :---: | :---: | :---: |
| Random 10% Missing | 6.2102 (-1.82) | 1.5004 (+0.04) | 0.6377 (-0.02)* | **0.9931 (-0.06)** |
| 6-Hour Station Outage | 5.8019 (-2.04) | 3.6669 (-0.78) | 1.3913 (-0.12)* | **1.8000 (-0.38)** |
| 12-Hour Station Outage | 5.8076 (-2.18) | 5.1487 (-0.92) | 3.0625 (-0.52)* | **2.4253 (-0.54)** |
| 24-Hour Station Outage | 5.4459 (-1.87) | 4.8859 (-0.61) | 4.8688 (-0.36)* | **2.4928 (-0.40)** |

*\*Note: Linear interpolation uses future data ($t+k$) and cannot be used in live forecasting.*

---

## 5. CI/CD Automation and Continuous Pipeline
1. **Automated Testing and Leakage Check (`ci.yml`)**: Runs `ruff` and `pytest` on 11 unit tests, including `test_leakage.py` which injects a $55^\circ\text{C}$ heatwave into future timestamps and proves past features remain identical (`assert_frame_equal`).
2. **Automated Accuracy Gate (`accuracy_gate.py`)**: Fails the build if our automated pipeline's RMSE ever gets worse than the manual baselines.
3. **Weekly Data Replay and Drift Check (`data-pipeline.yml` & `replay.py`)**: Feeds weekly batches of 168 hours, runs Kolmogorov-Smirnov (KS) drift tests, and quarantines corrupted sensor batches automatically.

---

## 6. Conclusion
In this project, we showed that replacing manual, notebook-based weather preprocessing with an automated, tested CI/CD pipeline improves both forecast accuracy and reliability. Across 45,253 hourly records, our **WeatherPipe** system achieved a 6-hour temperature forecast RMSE of **1.7776°C** ($R^2 = 0.9000$), beating a careful manual analyst baseline by **15.90%** and winning all 4 time-ordered cross-validation folds.

---

## References
1. National Centers for Environmental Information (NCEI), "Global Historical Climatology Network (GHCN) Daily and Hourly In Situ Archive," NOAA, 2025.
2. A. Murray, D. Dervovic, and M. Cashmore, "ELATE: LLM-Assisted Automated Feature Engineering for Time-Series Forecasting," *arXiv preprint arXiv:2508.14667*, Aug. 2025.
3. A. V. Malarkkan, W. Ying, and Y. Fu, "CAFE: Causal Automated Feature Engineering for Robust Machine Learning Under Distribution Shifts," *arXiv preprint arXiv:2602.16435*, Feb. 2026.
4. C. Guillaume-Castel, C. Li, and S. Sobolowski, "Deep Learning Approaches for Identifying Synoptic Meteorological Drivers: Verification and Interpretability of Atmospheric Extratropical Cyclones," *EGUsphere [Preprint]*, DOI: `10.5194/egusphere-2026-3362`, Jun. 2026.
5. M. J. Kanter and K. Veeramachaneni, "Deep Feature Synthesis: Towards Automating Data Science Endeavors," in *Proc. IEEE DSAA*, 2015, pp. 1–10.
6. J. P. Noboa, R. V. Silva, C. G. H. Bastos, and M. V. Lamar, "Feature Engineering and Group Selection for Urban Flood Prediction under Changing Climatic Regimes," *Water Resour. Manage.*, vol. 38, no. 12, pp. 4501–4519, Oct. 2024.
7. M. Banerjee, R. Pal, and S. Ray, "Automated Feature Extraction and Selection in Time-Domain Physiological and Mechanical Sensor Signals," *IEEE Trans. Cybern.*, vol. 48, no. 8, pp. 2410–2421, Aug. 2018.
