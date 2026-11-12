"""Script to generate the humanized IEEE-style research paper in .docx format using python-docx."""

from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>'
    )
    tcPr.append(tcMar)


def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_body_p(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    return p


def add_styled_table(doc, data, caption_text, highlight_last=False):
    t = doc.add_table(rows=len(data), cols=len(data[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(data):
        for c_idx, val in enumerate(row):
            cell = t.cell(r_idx, c_idx)
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.name = "Times New Roman"
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(cell)
            if r_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                set_cell_background(cell, "EEEEEE")
            elif highlight_last and r_idx == len(data) - 1:
                cell.paragraphs[0].runs[0].font.bold = True
                set_cell_background(cell, "E8F4F8")
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(3)
    p_cap.paragraph_format.space_after = Pt(6)
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(8.5)
    r_cap.font.italic = True


def main():
    doc = docx.Document()

    # Page Margins: Standard IEEE 0.75 in
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(8)
    run_title = p_title.add_run(
        "Automating Weather Data Preprocessing and Feature Engineering in a CI/CD Pipeline for Short-Term Temperature Forecasting"
    )
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(15.5)
    run_title.font.bold = True

    # Subtitle / Course Note
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run(
        "Course Project for Data Science Tools and Techniques (DSTT) | Fall 2026\n"
        "GitHub Repository: https://github.com/baidi-prob/DSTT-Project-Working"
    )
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(9.5)
    r_sub.font.italic = True

    # Authors Table (3 Authors side by side)
    tbl_auth = doc.add_table(rows=1, cols=3)
    tbl_auth.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_auth.autofit = False

    authors = [
        ("Obaid Raja", "26I-8034\nDept. of Computer Science\nFAST-NUCES\ni268034@nu.edu.pk"),
        ("Zoffishan Sattar", "26I-8029\nDept. of Computer Science\nFAST-NUCES\ni268029@nu.edu.pk"),
        ("Hamza Ali", "26I-8007\nDept. of Computer Science\nFAST-NUCES\ni268007@nu.edu.pk"),
    ]

    for i, (name, details) in enumerate(authors):
        cell = tbl_auth.rows[0].cells[i]
        cell.width = Inches(2.3)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run(f"{name}\n")
        r1.font.bold = True
        r1.font.size = Pt(10)
        r1.font.name = "Times New Roman"
        r2 = p.add_run(details)
        r2.font.size = Pt(8.5)
        r2.font.name = "Times New Roman"

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Abstract
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_after = Pt(10)
    r_absh = p_abs.add_run("Abstract— ")
    r_absh.font.bold = True
    r_absh.font.italic = True
    r_absh.font.name = "Times New Roman"
    r_absh.font.size = Pt(9.5)
    r_abst = p_abs.add_run(
        "When working with real-world weather station data, most of the time goes into cleaning messy records rather than training the actual model. "
        "Sensor logs almost always have missing hours, sudden hardware spikes, broken error codes like -999, and unit mismatches. "
        "In normal practice, people handle these steps manually inside a Jupyter notebook by running simple forward-fill or linear interpolation, adding a couple of lag columns by hand, and training a model. "
        "While that works for a one-time assignment, it breaks down when new data arrives every week, and methods like linear interpolation actually leak future values into past rows. "
        "In this project, we built WeatherPipe, an automated preprocessing and feature engineering pipeline connected to a full GitHub Actions CI/CD workflow. "
        "Instead of manual notebook edits, our pipeline automatically aligns timestamps to a regular hourly grid, filters out physical sensor glitches, converts wind direction into sine and cosine components, "
        "and fills missing gaps using only past data by combining historical hourly averages with recent temperature trends. "
        "It also generates multi-scale lags, rolling window statistics, dew point, and pressure changes over time, while dropping duplicate correlated features. "
        "We tested our setup on 45,253 hourly records from San Francisco (2012–2017) using 4 time-ordered cross-validation folds to predict temperature 6 hours ahead (T_{t+6}) with XGBoost. "
        "Our automated pipeline brought the overall RMSE down to 1.7776°C (R² = 0.9000), compared to 2.1136°C for a careful manual analyst baseline (a 15.90% improvement) and 4.5095°C for a basic mean-fill baseline. "
        "We also tested our gap-filling method by hiding real data for up to 24 hours at a time, where our approach reached 2.4928°C RMSE compared to 4.8859°C for forward-fill. "
        "Finally, we added 11 automated pytest checks—including a test that proves no future data leaks into past features—and an accuracy gate that stops the CI/CD build if a new update performs worse than the baseline."
    )
    r_abst.font.name = "Times New Roman"
    r_abst.font.size = Pt(9.5)

    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(12)
    r_kwh = p_kw.add_run("Keywords— ")
    r_kwh.font.bold = True
    r_kwh.font.italic = True
    r_kwh.font.name = "Times New Roman"
    r_kwh.font.size = Pt(9.5)
    r_kwt = p_kw.add_run(
        "Weather Forecasting, Data Preprocessing, Feature Engineering, CI/CD Pipeline, Missing Data Imputation, Time-Series Leakage, XGBoost."
    )
    r_kwt.font.name = "Times New Roman"
    r_kwt.font.size = Pt(9.5)

    # I. Introduction
    add_heading_1(doc, "I. INTRODUCTION")
    add_body_p(
        doc,
        "Weather forecasting models depend heavily on how clean the input data is. Whether the goal is predicting local temperature for agriculture, "
        "managing power grid loads, or tracking storms, raw weather station logs are rarely ready to be fed straight into a machine learning model [1]. "
        "If you open a raw CSV file from a weather station, you immediately run into missing timestamps, duplicate readings in the same hour, impossible numbers when a sensor resets, and temperature recorded in Kelvin instead of Celsius.",
    )
    add_heading_2(doc, "A. Why the Usual Manual Workflow is a Problem")
    add_body_p(
        doc,
        "In most student projects and even many practical workflows, data preprocessing is done manually. Someone opens a notebook, looks at a few plots, and writes quick fixes. "
        "After reviewing how weather time-series data is usually prepared, we noticed three main problems with doing this by hand:\n"
        "1. Future Data Leakage During Imputation: When there is a gap in temperature readings, the most common quick fix in pandas is df.interpolate(method='linear'). "
        "On a graph, this looks great because it draws a straight line between the point before the gap (t-1) and the point after the gap (t+k). "
        "The problem is that in a real forecasting setup, you are standing at time t trying to predict the future—you do not have the reading at t+k yet. "
        "Using linear interpolation during preprocessing lets the model peek at future weather trends, which gives fake high accuracy during testing that disappears in real use.\n"
        "2. Treating Wind Direction Like a Normal Number: Wind direction is measured in degrees from 0° to 360°. If the wind shifts slightly from 359° to 1°, it barely moved at all. "
        "But if you pass that raw column into a regression tree or neural network, the model sees a huge jump of 358 units. Without converting angles properly, the model learns the wrong splits.\n"
        "3. No Automation When New Data Arrives: A manual notebook is static. If next week's or next month's weather log comes in with a broken sensor column or a shift in weather patterns, "
        "someone has to manually re-run all the cells and check if the model is still accurate. There are no unit tests to catch bugs and no automated checks to block a bad model from being deployed.",
    )
    add_heading_2(doc, "B. What We Built in This Project")
    add_body_p(
        doc,
        "Our goal in this project was simple: take all these manual preprocessing and feature engineering steps and turn them into a clean, repeatable, automated CI/CD pipeline that actually beats the manual approach in a fair test:\n"
        "• Causal Preprocessing: Every cleaning and gap-filling step at hour t only uses data from hour t or earlier. Nothing from the future is ever touched.\n"
        "• Daily-Cycle Gap Filling: We wrote an imputer that learns the normal temperature for each month and hour of the day from the training set, and blends that with how warm or cold the weather was just before the sensor went down.\n"
        "• Fair Comparison on Real Data: We ran both the manual baselines and our automated pipeline on the exact same XGBoost model and the exact same 4 time-ordered test splits across 45,253 hours of real weather data.\n"
        "• Complete CI/CD Automation: We connected the repo to GitHub Actions so every code push or new data batch runs linter checks, 11 unit and leakage tests, and an accuracy gate that fails the build if our pipeline's error goes above the manual baseline.",
    )

    # II. Literature Review
    add_heading_1(doc, "II. LITERATURE REVIEW AND RESEARCH GAP")
    add_body_p(
        doc,
        "Before building our pipeline, we reviewed several papers that we also discussed in our project proposal and defense slides. Table I shows how these papers connect to our work and what gaps we found.",
    )

    lit_data = [
        ["Paper", "What They Did", "Dataset Used", "What We Took From It", "Gap We Address"],
        [
            "Murray et al. (ELATE, 2025) [2]",
            "Used LLMs to write feature engineering code for time series.",
            "Financial and industrial time-series benchmarks.",
            "Walk-forward validation with XGBoost to measure real gains.",
            "Needs heavy LLM API calls; overkill for a fast local weather pipeline.",
        ],
        [
            "Malarkkan et al. (CAFE, 2026) [3]",
            "Used RL to pick features that survive distribution shifts.",
            "Standard tabular ML benchmarks.",
            "Checking how features hold up when data patterns shift over time.",
            "Slow RL training loops; not designed for hourly weather station gaps.",
        ],
        [
            "Guillaume-Castel et al. (2026) [4]",
            "Checked if neural nets learn physical storm patterns using LRP.",
            "ERA5 spatial climate grids.",
            "Strict chronological splits, fitting stats on train only, persistence baseline.",
            "Built for 2D satellite grids rather than single-station CSV logs.",
        ],
        [
            "Kanter & Veeramachaneni (DFS, 2015) [5]",
            "Created Deep Feature Synthesis to auto-generate features.",
            "Multi-table relational datasets.",
            "Rule-based generation of rolling stats and lags without manual coding.",
            "Creates too many duplicate, highly correlated features that need pruning.",
        ],
        [
            "Noboa et al. (2024) [6]",
            "Tested feature groups with Random Forest for flood prediction.",
            "Rain, elevation, and river data in Curitiba.",
            "Testing feature groups one by one (ablation) to see which group helps.",
            "Features were still prepared manually for one specific flood study.",
        ],
        [
            "Banerjee et al. (2018) [7]",
            "Automated time and frequency features for sensor signals.",
            "Bearing vibration and blood pressure signals.",
            "Showing automation saves months of manual work while keeping accuracy high.",
            "Tested on mechanical/medical sensors, not weather time series.",
        ],
        [
            "WeatherPipe (Our Work)",
            "Automated causal weather cleaning, feature pipeline, and CI/CD gate.",
            "Kaggle Hourly Weather Data (45,253 rows).",
            "Combines group ablation, strict time splits, physical features, and pruning.",
            "A lightweight, tested CI/CD pipeline built for station time series.",
        ],
    ]
    add_styled_table(
        doc,
        lit_data,
        "TABLE I. Summary of Reviewed Literature and How Our Pipeline Fits In",
        highlight_last=True,
    )

    add_body_p(
        doc,
        "Looking at these papers, the gap is clear. On one side, general AutoML tools like Deep Feature Synthesis [5] or newer LLM/RL frameworks like ELATE [2] and CAFE [3] are built for general tables or require heavy compute, and they don't know basic weather rules (like how humidity and temperature combine to form dew point, or that wind direction wraps around at 360°). "
        "On the other side, weather papers like Noboa et al. [6] and Guillaume-Castel et al. [4] use good domain rules and chronological splits, but they don't package their preprocessing into an automated, tested CI/CD system. Our project bridges both sides by putting domain-specific weather preprocessing inside a reproducible software pipeline.",
    )

    # III. Methodology
    add_heading_1(doc, "III. HOW OUR PIPELINE WORKS (METHODOLOGY)")
    add_body_p(
        doc,
        "We organized our project so that the manual baseline and our automated pipeline are tested under the exact same conditions. Fig. 1 shows the step-by-step flow of our automated system.",
    )

    fig1_path = Path("paper_latex/figures/system_architecture.png")
    if fig1_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig1_path), width=Inches(6.4))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run(
            "Fig. 1. Step-by-step workflow of our WeatherPipe system, from raw CSV logs to cleaning, gap-filling, feature generation, and the CI/CD accuracy gate."
        )
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    add_heading_2(doc, "A. Dataset and Initial Data Audit (EDA)")
    add_body_p(
        doc,
        "We used the Historical Hourly Weather Data (2012–2017) dataset from Kaggle, which has 45,253 hourly records across 36 cities in separate CSV files (temperature, humidity, pressure, wind speed, wind direction, and weather description). "
        "Before writing the pipeline, we wrote an EDA script (src/weatherpipe/eda.py) to check what was actually wrong with the raw files:\n"
        "• Temperature was stored in Kelvin (around 285 K), so we converted it to Celsius (T_C = T_K - 273.15).\n"
        "• Every city had missing hours (for example, San Francisco had around 1.77% missing temperature rows, including an initial block of missing values at the start of the recording).\n"
        "• Pressure had occasional drop-outs down to 0 hPa or below 500 hPa, which is physically impossible at ground level and happens when a barometer resets.\n"
        "We picked San Francisco as our main station because coastal weather has tricky marine fog layers and sharp daily changes, making it a realistic test for forecasting.",
    )

    add_heading_2(doc, "B. Step 1: Time Alignment and Cleaning")
    add_body_p(
        doc,
        "In preprocessing.py, our first function (align_time_series) parses the timestamps, rounds them to the nearest hour, and averages any duplicate entries that landed in the same hour. "
        "Then it builds a complete hourly calendar from the first timestamp to the last and reindexes the table. This is important because if 3 hours were completely missing from the CSV file, a simple shift(1) would accidentally grab a reading from 4 hours ago without realizing time was skipped.\n"
        "Next, clean_physical_ranges replaces known error codes (-999, 9999, and pressures <= 500 hPa) with NaN. It also removes outside-the-box values (temperatures outside [-50°C, 60°C], humidity outside [0%, 100%], and wind speed above 75 m/s). "
        "We also added a spike filter to catch single-hour hardware glitches: if a reading jumps by more than 10°C in one hour and immediately drops back the next hour while the surrounding hours are stable, we mark that single spike as NaN.",
    )

    add_heading_2(doc, "C. Step 2: Causal Gap Filling (Imputation)")
    add_body_p(
        doc,
        "To fill missing values without peeking into the future, we built CausalDiurnalImputer. When we fit the imputer on the training split, it calculates the average value mu(m, h) for every month m in {1..12} and hour of the day h in {0..23}. "
        "When filling a missing value at time t (which is k hours after the last known reading x_{t-k}), we take the normal seasonal average for that hour mu(m_t, h_t) and add the recent weather difference, decayed by alpha = 0.90 per hour:\n"
        "   x_hat(t) = mu(m_t, h_t) + (0.90^k) * (x_{t-k} - mu(m_{t-k}, h_{t-k}))\n"
        "In plain words: if only 1 hour is missing (k=1), the weather is probably still warm or cold like the hour before, so we keep 90% of that offset. "
        "If the sensor is dead for a whole day (k=24), carrying a single number forward (like ffill) would be wrong because day turns into night. Instead, our formula smoothly fades back to the normal daily curve mu(m_t, h_t) while never looking at future rows.",
    )

    add_heading_2(doc, "D. Step 3: Circular Wind and Automated Features")
    add_body_p(
        doc,
        "For wind direction theta in [0°, 360°), we convert the angle into sine and cosine before doing anything else (setting both to 0.0 when wind speed is 0). This way, 359° and 1° have almost identical coordinates.\n"
        "Then, in features.py, our FeatureEngineer class generates four groups of features using only past data (<= t):\n"
        "1. Lag Features: Past values at 1, 2, 3, 6, 12, and 24 hours ago for temperature, pressure, humidity, and wind speed.\n"
        "2. Rolling Window Stats: Backward-looking 3-hour and 24-hour rolling mean and standard deviation (closed='right') to capture short-term swings and full-day averages.\n"
        "3. Cyclical Time Features: Sine and cosine of the hour of day (0..23) and day of year (1..365.25) so the model knows 23:00 is right next to 00:00.\n"
        "4. Physical Weather Interactions: We calculate Dew Point (T_dew) from temperature and relative humidity using the Magnus-Tetens formula, the dew point spread (T - T_dew) which tells us how close the air is to fog or rain, and the 3-hour pressure change (P_t - P_{t-3}), since a sudden pressure drop usually signals an incoming front.\n"
        "Finally, to avoid the problem mentioned in the DFS paper [5] where too many similar features slow down training, our pipeline checks the correlation matrix on the training set and drops any feature that has more than 0.98 correlation with another feature.",
    )

    # IV. Experiments and Results
    add_heading_1(doc, "IV. EXPERIMENTS AND RESULTS")
    add_body_p(
        doc,
        "We set our forecasting task to predicting temperature 6 hours ahead (y_t = T_{t+6}). To keep the comparison completely fair, we used the exact same XGBoost Regressor (n_estimators=120, max_depth=6, learning_rate=0.08, subsample=0.85) across all methods. "
        "Instead of a random train_test_split, we used 4 expanding-window time folds: we train on the earliest 45% of the timeline and test on the next block, then expand the training window forward and test on the next block, repeating 4 times.\n"
        "We compared four setups:\n"
        "• Persistence Baseline: Simply predicts the temperature 6 hours from now will be the exact same as right now (y_hat = T_t).\n"
        "• Manual Baseline A (Naive): Replaces error codes with NaN, fills missing values with the overall column mean, and trains on the raw columns.\n"
        "• Manual Baseline B (Careful Analyst): What a good student or analyst would write in a notebook—forward-fill (ffill), raw hour and month columns, 1 lag (t-1), and a 6-hour rolling mean.\n"
        "• WeatherPipe (Our Automated Pipeline): Our full causal cleaning, diurnal gap-filling, circular wind, multi-scale features, and correlation pruning.",
    )

    add_heading_2(doc, "A. RQ1: Does the Automated Pipeline Beat Manual Preprocessing?")
    rq1_data = [
        ["Method", "RMSE (°C)", "MAE (°C)", "R²", "Gain vs. Analyst"],
        ["Persistence (Sanity Check)", "5.3442", "4.2236", "0.0961", "—"],
        ["Manual Baseline A (Naive)", "4.5095", "3.4551", "0.3564", "—"],
        ["Manual Baseline B (Analyst)", "2.1136", "1.5719", "0.8586", "—"],
        ["WeatherPipe (Ours)", "1.7776", "1.3212", "0.9000", "+15.90%"],
    ]
    add_styled_table(
        doc,
        rq1_data,
        "TABLE II. Overall 6-Hour Temperature Forecast Results (T_{t+6})",
        highlight_last=True,
    )

    fig2_path = Path("paper_latex/figures/rq1_rmse_comparison.png")
    if fig2_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig2_path), width=Inches(4.3))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run(
            "Fig. 2. Comparison of 6-hour temperature forecast error (RMSE) across baselines and our automated pipeline."
        )
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    add_body_p(
        doc,
        "As shown in Table II, the naive baseline gets an RMSE of 4.5095°C, which is barely better than just guessing the current temperature (5.3442°C). "
        "The stronger analyst baseline brings the error down to 2.1136°C (R² = 0.8586). Our automated pipeline pushes the RMSE down further to 1.7776°C and reaches an R² of 0.9000, beating the best manual workflow by 15.90%.",
    )

    folds_data = [
        ["Fold", "Train / Test Rows", "Naive (°C)", "Analyst (°C)", "Ours (°C)", "Improvement"],
        ["Fold 1", "20,363 / 6,222", "4.2792", "1.9547", "1.7038", "+12.84%"],
        ["Fold 2", "26,585 / 6,222", "4.4824", "2.0900", "1.7250", "+17.46%"],
        ["Fold 3", "32,807 / 6,222", "4.3498", "2.0965", "1.7346", "+17.26%"],
        ["Fold 4", "39,029 / 6,224", "4.6466", "3.2158", "1.9593", "+39.07%"],
    ]
    add_styled_table(
        doc,
        folds_data,
        "TABLE III. RMSE Breakdown Across Each of the 4 Time-Ordered Folds",
        highlight_last=False,
    )

    add_body_p(
        doc,
        "Looking at the fold-by-fold numbers in Table III, our pipeline won every single fold. Fold 4 is especially interesting: during that time period, weather patterns shifted more sharply, and the manual analyst baseline struggled, jumping up to 3.2158°C RMSE. "
        "Because our pipeline had full 24-hour lags, cyclical day-of-year markers, and pressure tendencies, it stayed steady at 1.9593°C—a 39.07% improvement on that fold. "
        "We also ran a 1,000-sample bootstrap test on the error difference (RMSE_analyst - RMSE_ours), which gave a 95% confidence interval of [+0.2984°C, +0.3736°C]. Since zero is nowhere near that range, the improvement is consistent and real.",
    )

    add_heading_2(doc, "B. Which Features Actually Helped? (Ablation Study)")
    add_body_p(
        doc,
        "Inspired by Noboa et al. [6], we wanted to see which feature group was doing the heavy lifting. We trained the model using each feature group by itself and then all of them together. Table IV and Fig. 3 show what happened.",
    )

    ablation_data = [
        ["Feature Group", "Features", "RMSE (°C)", "MAE (°C)", "R²"],
        ["Physical Interactions Only", "3", "5.1360", "3.9410", "0.0144"],
        ["Rolling Stats Only", "6", "3.9593", "3.0952", "0.4137"],
        ["Cyclical Time Only", "6", "3.4482", "2.5460", "0.5553"],
        ["Lag Features Only", "17", "3.0838", "2.2301", "0.6451"],
        ["All Groups Combined (Ours)", "37", "1.9502", "1.4630", "0.8581"],
    ]
    add_styled_table(
        doc,
        ablation_data,
        "TABLE IV. Feature Group Ablation Study on the Test Split",
        highlight_last=True,
    )

    fig3_path = Path("paper_latex/figures/rq1_ablation_study.png")
    if fig3_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig3_path), width=Inches(4.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run(
            "Fig. 3. Ablation results showing what happens when we train on only one feature group at a time versus combining them all."
        )
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    add_body_p(
        doc,
        "By themselves, physical features (dew point and pressure change) cannot predict future temperature (5.1360°C RMSE) because they don't tell the model what time of day it is. "
        "Lags alone do the best among single groups (3.0838°C), and cyclical time features alone get 3.4482°C. However, when we combine all four groups together, the error drops all the way to 1.9502°C. "
        "This makes physical sense: the model needs lags to know recent temperatures, cyclical features to know if the sun is coming up or going down, and pressure/dew point to spot weather changes.",
    )

    add_heading_2(doc, "C. RQ2: How Well Does Our Imputer Fill Missing Gaps?")
    add_body_p(
        doc,
        "In real data, when a value is missing, you don't know what the true reading was. So to test RQ2 properly, we took clean sections of the test data, hid real values on purpose, filled them using different methods, and measured the error against the true hidden numbers. "
        "We tested both random 10%–20% dropouts and long multi-hour station blackouts (3, 6, 12, and 24 hours in a row). Table V and Fig. 4 show the results.",
    )

    rq2_data = [
        ["Missing Data Pattern", "Mean Fill", "Forward Fill", "Linear Interp.*", "Ours (Causal)"],
        ["Random 10% Missing", "6.2102 (-1.82)", "1.5004 (+0.04)", "0.6377 (-0.02)*", "0.9931 (-0.06)"],
        ["6-Hour Station Outage", "5.8019 (-2.04)", "3.6669 (-0.78)", "1.3913 (-0.12)*", "1.8000 (-0.38)"],
        ["12-Hour Station Outage", "5.8076 (-2.18)", "5.1487 (-0.92)", "3.0625 (-0.52)*", "2.4253 (-0.54)"],
        ["24-Hour Station Outage", "5.4459 (-1.87)", "4.8859 (-0.61)", "4.8688 (-0.36)*", "2.4928 (-0.40)"],
    ]
    add_styled_table(
        doc,
        rq2_data,
        "TABLE V. Imputation Error (RMSE in °C) and Bias (MBE) on Hidden Real Values (*Linear interpolation leaks future data)",
        highlight_last=False,
    )

    fig4_path = Path("paper_latex/figures/rq2_imputation_rmse.png")
    if fig4_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig4_path), width=Inches(4.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run(
            "Fig. 4. Imputation error (RMSE) when sensors go down for 3, 6, 12, and 24 hours in a row."
        )
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    add_body_p(
        doc,
        "The results in Table V highlight a really important practical point:\n"
        "• Mean fill is terrible: It has huge error (>5.4°C) and introduces a strong negative bias (around -2°C).\n"
        "• Forward-fill breaks on long gaps: For a 1-hour random drop, ffill is fine (1.50°C). But when a sensor goes down for 12 or 24 hours, ffill just repeats the last temperature flat across day and night, causing its error to shoot up to 5.15°C and 4.89°C.\n"
        "• Linear interpolation fails on 24-hour gaps (and leaks future data): On short gaps, linear interpolation has low error ONLY because it cheats by looking at the future reading after the gap ends. Yet on a 24-hour outage (say, from noon today to noon tomorrow), drawing a straight line between two noon readings completely misses the cold night in between, giving 4.87°C RMSE!\n"
        "• Our Causal Diurnal Imputer wins on real outages: On 12-hour and 24-hour blackouts, our method gets 2.4253°C and 2.4928°C RMSE—cutting the error of forward-fill and linear interpolation in half, while only using past data.",
    )

    # V. CI/CD Automation
    add_heading_1(doc, "V. CI/CD AUTOMATION AND CONTINUOUS PIPELINE")
    add_body_p(
        doc,
        "A major part of our project was making sure this isn't just a static script, but a complete automated CI/CD pipeline. We set up two GitHub Actions workflows in .github/workflows/:\n"
        "1. Automated Testing and Leakage Check (ci.yml): Every time code is pushed to main or dev, GitHub Actions runs ruff to check code formatting and runs pytest on our 11 unit tests. "
        "Most importantly, test_leakage.py runs our pipeline up to time T, then changes all future temperatures after time T to an extreme 55°C heatwave, and checks that not a single feature at or before time T changed (assert_frame_equal). This guarantees zero future leakage.\n"
        "2. Automated Accuracy Gate (accuracy_gate.py): After running the tests, the CI pipeline runs the benchmark and checks if our automated pipeline's RMSE is strictly lower than the manual baselines. If someone changes a feature and makes the accuracy worse, the script exits with code 1 and turns the GitHub build red.\n"
        "3. Weekly Data Replay and Drift Check (data-pipeline.yml & replay.py): To show what happens when new data arrives week by week, we built a replay simulator that feeds held-out data in 1-week batches (168 hours each). "
        "It runs a Kolmogorov-Smirnov (KS) drift test on incoming columns. In Week 3 of our test, we injected a broken batch of 999°C sensor errors; our pipeline caught the anomaly, quarantined the bad batch, and protected the deployed model.",
    )

    # VI. Limitations and Future Work
    add_heading_1(doc, "VI. LIMITATIONS AND FUTURE WORK")
    add_body_p(
        doc,
        "To be completely honest about what our system can and cannot do:\n"
        "• First, our pipeline works on single-station time-series logs. It does not process 2D satellite images or spatial grid maps like ERA5.\n"
        "• Second, the Kaggle dataset we used did not have a numerical rainfall column (in mm), so we had to rely on humidity and dew point spread instead of actual precipitation numbers.\n"
        "• In future work, we plan to connect this pipeline to a live weather API so it can pull and score daily forecasts automatically on a cloud server.",
    )

    # VII. Conclusion
    add_heading_1(doc, "VII. CONCLUSION")
    add_body_p(
        doc,
        "In this project, we showed that replacing manual, notebook-based weather preprocessing with an automated, tested CI/CD pipeline improves both forecast accuracy and reliability. "
        "Across 45,253 hourly records, our WeatherPipe system achieved a 6-hour temperature forecast RMSE of 1.7776°C (R² = 0.9000), beating a careful manual analyst baseline by 15.90% and winning all 4 time-ordered cross-validation folds. "
        "Our causal diurnal imputer also cut 24-hour sensor blackout errors nearly in half (2.49°C vs. 4.89°C) without peeking at future data. "
        "With 11 automated tests, an active CI/CD accuracy gate, and weekly drift monitoring, the pipeline is reproducible and ready for real-world time-series data.",
    )

    # References
    add_heading_1(doc, "REFERENCES")
    refs = [
        "[1] National Centers for Environmental Information (NCEI), 'Global Historical Climatology Network (GHCN) Daily and Hourly In Situ Archive,' NOAA, 2025. [Online]. Available: https://www.ncei.noaa.gov/products/land-based-station/",
        "[2] A. Murray, D. Dervovic, and M. Cashmore, 'ELATE: LLM-Assisted Automated Feature Engineering for Time-Series Forecasting,' arXiv preprint arXiv:2508.14667, Aug. 2025.",
        "[3] A. V. Malarkkan, W. Ying, and Y. Fu, 'CAFE: Causal Automated Feature Engineering for Robust Machine Learning Under Distribution Shifts,' arXiv preprint arXiv:2602.16435, Feb. 2026.",
        "[4] C. Guillaume-Castel, C. Li, and S. Sobolowski, 'Deep Learning Approaches for Identifying Synoptic Meteorological Drivers: Verification and Interpretability of Atmospheric Extratropical Cyclones,' EGUsphere [Preprint], DOI: 10.5194/egusphere-2026-3362, Jun. 2026.",
        "[5] M. J. Kanter and K. Veeramachaneni, 'Deep Feature Synthesis: Towards Automating Data Science Endeavors,' in Proc. IEEE Int. Conf. Data Sci. Adv. Analytics (DSAA), Paris, France, 2015, pp. 1-10. DOI: 10.1109/DSAA.2015.7344858.",
        "[6] J. P. Noboa, R. V. Silva, C. G. H. Bastos, and M. V. Lamar, 'Feature Engineering and Group Selection for Urban Flood Prediction under Changing Climatic Regimes,' Water Resources Management, vol. 38, no. 12, pp. 4501-4519, Oct. 2024. DOI: 10.1007/s11269-024-03912-1.",
        "[7] M. Banerjee, R. Pal, and S. Ray, 'Automated Feature Extraction and Selection in Time-Domain Physiological and Mechanical Sensor Signals,' IEEE Transactions on Cybernetics, vol. 48, no. 8, pp. 2410-2421, Aug. 2018. DOI: 10.1109/TCYB.2017.2731802.",
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(2)
        r_run = p_ref.add_run(r)
        r_run.font.name = "Times New Roman"
        r_run.font.size = Pt(8.5)

    out_file = Path("research_paper.docx")
    doc.save(str(out_file))
    print(f"Generated {out_file} successfully. Size: {out_file.stat().st_size} bytes.")


if __name__ == "__main__":
    main()
