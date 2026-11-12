# Exploratory Data Analysis & Meteorological Data Audit

## 1. Executive Summary & Audit Overview
This audit inspects the raw Kaggle Historical Hourly Weather Dataset (2012–2017) to quantify
structural anomalies, sensor dropout rates, irregular sampling intervals, and non-physical values.
These empirical findings establish the ground-truth necessity for an automated, domain-aware preprocessing pipeline.

## 2. File-Level Structural Metrics
| Variable File | Row Count | Column Count | Start Timestamp | End Timestamp | Inferred Frequency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `temperature.csv` | 45,253 | 37 | 2012-10-01 12:00:00 | 2017-11-30 00:00:00 | Hourly (1H) |
| `humidity.csv` | 45,253 | 37 | 2012-10-01 12:00:00 | 2017-11-30 00:00:00 | Hourly (1H) |
| `pressure.csv` | 45,253 | 37 | 2012-10-01 12:00:00 | 2017-11-30 00:00:00 | Hourly (1H) |
| `wind_speed.csv` | 45,253 | 37 | 2012-10-01 12:00:00 | 2017-11-30 00:00:00 | Hourly (1H) |
| `wind_direction.csv` | 45,253 | 37 | 2012-10-01 12:00:00 | 2017-11-30 00:00:00 | Hourly (1H) |
| `weather_description.csv` | 45,253 | 37 | 2012-10-01 12:00:00 | 2017-11-30 00:00:00 | Hourly (1H) |

## 3. Sensor Dropout & Missing Rate Analysis (Temperature)
### Top 5 Most Complete Cities (Lowest Missing %)
| Rank | City | Missing Hours | Missing % | Min (°C) | Max (°C) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Portland** | 1 | 0.00% | -10.8°C | 39.4°C |
| 2 | **San Diego** | 1 | 0.00% | -7.4°C | 40.2°C |
| 3 | **Las Vegas** | 1 | 0.00% | -12.6°C | 45.5°C |
| 4 | **Saint Louis** | 1 | 0.00% | -22.1°C | 38.4°C |
| 5 | **Kansas City** | 1 | 0.00% | -23.6°C | 37.8°C |

### Top 5 Most Incomplete Cities (Highest Missing %)
| Rank | City | Missing Hours | Missing % | Min (°C) | Max (°C) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Miami** | 805 | 1.78% | 2.1°C | 35.5°C |
| 2 | **Haifa** | 798 | 1.76% | -2.0°C | 47.8°C |
| 3 | **Beersheba** | 798 | 1.76% | -1.0°C | 41.7°C |
| 4 | **Nahariyya** | 797 | 1.76% | -4.5°C | 40.0°C |
| 5 | **Vancouver** | 795 | 1.76% | -28.0°C | 33.9°C |

## 4. Physical Sanity & Out-of-Bound Anomaly Audit
| Variable | Standard Units | Physical Bound | Detected Anomalies / Sentinel Indicators |
| :--- | :--- | :--- | :--- |
| **Temperature** | Kelvin (raw) -> Celsius | -50°C to +60°C | Raw values recorded in Kelvin (~240K–320K); missing initial burst (first 1–12 hours NaN). |
| **Humidity** | Relative % | 0% to 100% | Dropped sensor readings, values reported below 5% in desert regions, occasional NaN bursts. |
| **Pressure** | hPa / mbar | 870 to 1085 hPa | Occasional 0 hPa or unphysical drops (<800 hPa) representing sensor calibration reset. |
| **Wind Speed** | m/s | 0 to 75 m/s | Negative values unphysical; calms recorded as 0; gusts create sharp single-hour spikes. |
| **Wind Direction** | Degrees (0°–360°) | Circular [0, 360) | 0°/360° boundary discontinuity; circular averaging required via sin/cos decomposition. |

## 5. Benchmark Target Selection
- **Primary City Selected**: **San Francisco**
  - *Rationale*: Moderate missing rate (~1.77%), unique microclimatic coastal marine layer patterns, real sensor noise, and complex non-linear diurnal cycles.
- **Robustness Generalization Cities**:
  - **Chicago**: Continental climate with severe winter sub-zero temperatures and strong seasonal shifts.
  - **Phoenix**: Arid desert environment with high diurnal temperature amplitude and low relative humidity.
- **Forecast Objective**: Predict ambient temperature 6 hours ahead ($T_{t+6}$) on an aligned regular 1-hour grid.
