# Project Defense Speaking Notes & Examiner Q&A Guide

**Project Title**: Automated Weather Data Preprocessing and Feature Engineering Pipeline  
**Course**: Data Science Tools and Techniques (DSTT, 20 Marks)  
**Authors**: Obaid Raja, Zoffishan Sattar, Hamza Ali  

---

## 1. High-Level Pitch (The 60-Second Opening)
> *"Respected committee, our project tackles a critical vulnerability in applied meteorological data science: the reliance on manual, ad-hoc, and error-prone preprocessing in exploratory notebooks. In current practice, practitioners often use non-causal linear interpolation—which looks into the future—and hand-crafted features that fail when weather drifts across seasons.*  
>  
> *We designed, implemented, and validated **WeatherPipe**, a domain-aware, fully automated preprocessing and feature engineering pipeline integrated with an MLOps CI/CD framework. WeatherPipe guarantees strict temporal causality, handles circular wind direction, implements a diurnal climatology imputer, and automatically generates multi-scale lags and thermodynamic interactions.*  
>  
> *On 45,253 real hourly records from Kaggle, WeatherPipe achieved an overall 6-hour forecast RMSE of 1.7776°C, beating an expert manual analyst baseline (2.1136°C) by 15.90% across all 4 chronological folds, proven significant by bootstrap confidence intervals. Our system is fully tested with 11 passing tests—including a mathematical future-leakage test—and governed by a GitHub Actions CI/CD accuracy gate."*

---

## 2. Walkthrough of Slide-by-Slide Defense

### Slide 1: Title & Team
- State your registration numbers and course (DSTT).
- Emphasize that this is an end-to-end system combining **data engineering, domain meteorology, machine learning, and CI/CD operations**.

### Slide 2: Problem Statement & Research Questions
- **RQ1**: Does automated feature engineering improve predictive accuracy over manual preprocessing?
  - *Answer*: Yes, by **15.90%** over an expert analyst and **60.58%** over a naive baseline.
- **RQ2**: Does automated imputation handle missing records without synthetic bias or future leakage?
  - *Answer*: Yes, our causal diurnal imputer achieved **2.49°C RMSE** on 24-hour outages compared to **4.89°C** for forward-fill and **4.87°C** for linear interpolation, while being 100% causal.

### Slide 3: Dataset & Empirical Findings from EDA
- Dataset: Kaggle Historical Hourly Weather (2012–2017), 45,253 hourly records across 36 cities.
- Primary City: **San Francisco** (complex coastal marine layer microclimate).
- Mention key EDA discoveries:
  1. Temperature was recorded in Kelvin (mean ~288 K) and converted to Celsius.
  2. Detected initial missing bursts and pressure drops below 500 hPa which represent calibration resets.
  3. Identified wind direction discontinuity at 0°/360°.

### Slide 4: Literature Review & Gap Analysis
- Be ready to explain the connection to the literature:
  - **Noboa et al. (2024)**: We adopted their feature grouping and ablation methodology.
  - **Banerjee et al. (2018)**: We proved automated feature extraction beats manual coding while keeping features physically interpretable (unlike black-box PCA).
  - **Guillaume-Castel et al. (2026)**: We adopted their chronological validation discipline, training-only parameter fitting, and persistence reference benchmark.
  - **Kanter & Veeramachaneni (2015)**: We addressed the known DFS limitation of collinear redundancy by integrating a correlation pruning filter ($|r| > 0.98$).

### Slide 5: Methodology & System Architecture
- Point to the architecture flowchart:
  `Raw Data -> Alignment -> Quality Bounds & Spike Filter -> Causal Diurnal Imputer -> Circular Wind Encodings -> Multi-Scale Features & Dew Point -> XGBoost -> CI/CD Gate`.
- Highlight **Strict Causality**: every computation at time $t$ only uses data $\le t$.

### Slide 6: Benchmark Results (RQ1)
- Present the core table:
  - Persistence: 5.34°C RMSE
  - Naive Baseline: 4.51°C RMSE
  - Analyst Baseline: 2.11°C RMSE
  - **WeatherPipe: 1.78°C RMSE (15.90% gain)**
- Point out that WeatherPipe won all 4 expanding folds (F1: 1.70°C vs 1.95°C; F4: 1.96°C vs 3.22°C).

### Slide 7: Feature Group Ablation Study
- Explain Table 3:
  - Lags alone: 3.08°C
  - Cyclical alone: 3.45°C
  - Rolling alone: 3.96°C
  - Physical interactions alone: 5.14°C
  - **All Combined: 1.95°C**
- Takeaway: Combining temporal dynamics with thermodynamic interactions provides synergistic predictive power.

### Slide 8: Missing Data Benchmark (RQ2)
- Explain why linear interpolation is a "cheat" in real-time systems: it peeks into the future.
- On 24-hour outages, forward-fill gives 4.89°C RMSE because it holds night temperatures constant across the hot day.
- WeatherPipe's causal diurnal imputer achieves **2.49°C RMSE**, preserving the solar diurnal rhythm.

### Slide 9: CI/CD Pipeline & Live Demonstration
- Explain the GitHub Actions workflow:
  1. `ruff` checks linting and formatting.
  2. `pytest` runs 11 unit, causality, and leakage tests.
  3. `experiment_rq1` runs on validation data.
  4. `accuracy_gate` evaluates $\text{RMSE}_{\text{auto}} < \text{RMSE}_{\text{manual}}$. If worse, the build turns red and blocks deployment.
  5. Demonstrate the simulated gate failure (`--simulate-failure`) to prove the safety gate works.

---

## 3. Tough Examiner Questions & How to Answer

### Q1: "Why didn't you use standard 5-fold cross-validation or train_test_split from scikit-learn?"
**Answer**:
> *"Standard K-fold cross-validation and random train_test_split shuffle data randomly across time. In weather forecasting, this results in severe data leakage because the model trains on Wednesday, tests on Tuesday, and interpolates between known time points. Weather is an auto-regressive process. To simulate real-world conditions honestly, we used expanding-window chronological splits: the model is trained strictly on the past and tested on the unseen future block."*

### Q2: "Why is linear interpolation considered a data leak in forecasting?"
**Answer**:
> *"Linear interpolation calculates the missing value at timestamp $t$ by connecting the past value at $t-1$ and the future value at $t+k$. In an offline visualization, that looks fine. But in a live operational forecasting pipeline at 2:00 PM, you do not know the sensor value at 6:00 PM! If you use linear interpolation to train your model, you are giving the model future information it will never have during live deployment. Our `CausalDiurnalImputer` uses only past observations plus historical climatology, making it 100% causal and deployable."*

### Q3: "Why did you encode wind direction into sine and cosine?"
**Answer**:
> *"Wind direction is measured in degrees from 0° to 360°. A wind blowing at 359° (NNW) and a wind blowing at 1° (NNE) are only 2° apart physically. However, if fed into a model as a raw integer, the algorithm treats them as 358 units apart! By decomposing wind into $\sin(2\pi \theta / 360)$ and $\cos(2\pi \theta / 360)$, the Euclidean distance between 359° and 1° correctly reflects their genuine physical proximity."*

### Q4: "How can you be sure there is no future data leakage in your features?"
**Answer**:
> *"We wrote an explicit mathematical unit test: `tests/test_leakage.py`. The test computes feature vectors up to timestamp $T$. Then, it drastically mutates the data after timestamp $T$ (injecting a 55°C heatwave and 890 hPa hurricane pressure). The test re-extracts the features and asserts that every single feature before timestamp $T$ remains bit-for-bit identical (`pd.testing.assert_frame_equal`). The test passes with zero discrepancies."*

### Q5: "Is your manual baseline a strawman (intentionally weak)?"
**Answer**:
> *"No, we specifically built two manual baselines. Baseline A is the naive student baseline (mean fill, raw features). Baseline B is an expert analyst baseline: it includes forward-fill, calendar features, a 1-hour lag, a 6-hour moving average, and standard scaling. Baseline B achieved an $R^2$ of 0.8586, which is already a solid model. WeatherPipe still outperformed Baseline B by 15.90% (RMSE 1.78°C vs 2.11°C), proving that systematic, domain-aware feature engineering provides a genuine scientific advantage."*

### Q6: "What did you correct in the literature references compared to the original proposal?"
**Answer**:
> *"In our initial project proposal, two citations (ELATE and CAFE) were listed with preliminary workshop citations from 2024/2025. During verification against published archives, we corrected ELATE to Murray, Dervovic, and Cashmore (arXiv:2508.14667, Aug 2025) and CAFE to Malarkkan, Ying, and Fu (arXiv:2602.16435, Feb 2026). We also cited the Guillaume-Castel et al. paper by its verified EGUsphere DOI (10.5194/egusphere-2026-3362) and fully aligned our experimental design with Noboa et al. (2024) and Banerjee et al. (2018)."*
