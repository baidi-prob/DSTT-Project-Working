"""Script to generate IEEE-style research paper in .docx format using python-docx."""

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


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>'
    )
    tcPr.append(tcMar)


def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
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
    run_title = p_title.add_run("WeatherPipe: An Automated, Causal Preprocessing and Feature Engineering Pipeline for Robust Meteorological Time-Series Forecasting")
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(16)
    run_title.font.bold = True

    # Subtitle / Course Note
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("Course Project for Data Science Tools and Techniques (DSTT) | Fall 2026\nGitHub Implementation Repository: https://github.com/baidi-prob/DSTT-Project-Working")
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

    # Abstract Box
    p_abs = doc.add_paragraph()
    p_abs.paragraph_format.space_after = Pt(12)
    r_absh = p_abs.add_run("Abstract— ")
    r_absh.font.bold = True
    r_absh.font.italic = True
    r_absh.font.name = "Times New Roman"
    r_absh.font.size = Pt(9.5)
    r_abst = p_abs.add_run(
        "Meteorological time-series forecasting is critical for agricultural scheduling, municipal hazard management, and renewable energy dispatch. "
        "However, raw environmental sensor telemetry is frequently corrupted by irregular sampling intervals, transmission dropout bursts, instrument recalibration resets, and severe diurnal non-linearities. "
        "Standard data science practice relies heavily on manual, ad-hoc exploratory notebook scripts, utilizing bidirectional linear interpolation and heuristic feature engineering. "
        "This manual paradigm introduces severe future data leakage, lacks reproducible testing, and collapses under production distribution shifts. "
        "This paper introduces WeatherPipe, a domain-aware, strictly causal automated preprocessing and feature engineering pipeline integrated with an MLOps Continuous Integration and Continuous Deployment (CI/CD) framework. "
        "WeatherPipe incorporates: (1) automated temporal sequence alignment and deduplication to a regular hourly grid; (2) thermodynamic boundary enforcement and isolated instrument spike filtering; "
        "(3) a domain-aware causal diurnal imputer combining seasonal diurnal climatology with decaying anomaly persistence without peeking into future data; (4) circular wind direction decomposition (sin/cos); "
        "(5) multi-scale causal lags, backward rolling statistics, cyclical encodings, and thermodynamic interaction metrics (Magnus-Tetens dew point and pressure tendency); and (6) automated collinearity pruning. "
        "Evaluated on 45,253 real-world hourly observations across four expanding temporal folds for 6-hour ahead temperature forecasting (T_{t+6}), WeatherPipe achieves an overall RMSE of 1.7776°C (R² = 0.9000), representing a 15.90% error reduction over an expert manual analyst baseline (2.1136°C) and a 60.58% reduction over a naive baseline (4.5095°C). "
        "A feature ablation study validates that combining multi-scale temporal dynamics with thermodynamic interactions provides synergistic predictive power. "
        "In controlled ground-truth masking benchmarks, WeatherPipe's causal imputer reduces 24-hour blackout error to 2.4928°C RMSE compared to 4.8859°C for forward-fill and 4.8688°C for non-causal linear interpolation. "
        "The system is validated by 11 automated unit and mathematical leakage tests, and governed by a production GitHub Actions CI/CD accuracy gate."
    )
    r_abst.font.name = "Times New Roman"
    r_abst.font.size = Pt(9.5)

    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_after = Pt(14)
    r_kwh = p_kw.add_run("Keywords— ")
    r_kwh.font.bold = True
    r_kwh.font.italic = True
    r_kwh.font.name = "Times New Roman"
    r_kwh.font.size = Pt(9.5)
    r_kwt = p_kw.add_run("Automated Feature Engineering, Meteorological Time-Series, Causal Imputation, CI/CD, MLOps, Data Leakage Prevention, XGBoost.")
    r_kwt.font.name = "Times New Roman"
    r_kwt.font.size = Pt(9.5)

    # I. Introduction
    add_heading_1(doc, "I. INTRODUCTION")
    add_body_p(doc,
        "Surface weather observations collected via environmental sensor networks serve as the primary operational inputs for short-term atmospheric forecasting, "
        "agricultural planning, renewable energy load balancing, and municipal hazard prevention. In applied settings, model predictive accuracy is strictly bounded by "
        "the structural fidelity, quality, and temporal causality of upstream data engineering."
    )
    add_heading_2(doc, "A. The Fragility of Manual Preprocessing")
    add_body_p(doc,
        "Despite significant advances in model architectures, data preprocessing in applied meteorological projects remains predominantly manual, subjective, and ad-hoc. "
        "Practitioners routinely rely on interactive notebooks, introducing three fatal vulnerabilities:\n"
        "1. Future Data Leakage via Bidirectional Interpolation: Linear interpolation connects observations at t-1 and t+k to impute t. While visually smooth, it peeks into the future, which is physically impossible in live operational forecasting.\n"
        "2. Euclidean Discontinuities in Circular Variables: Wind direction is recorded in degrees [0°, 360°). Treating wind linearly treats 359° and 1° as 358 units apart rather than 2° apart.\n"
        "3. Absence of Automated Verification: Ad-hoc notebooks lack regression testing, unit assertions, and performance gates, leading to silent model degradation in production."
    )
    add_heading_2(doc, "B. Research Contributions")
    add_body_p(doc,
        "To resolve these bottlenecks, we architect, implement, and validate WeatherPipe. The system guarantees strict temporal causality (observations <= t only), "
        "introduces a domain-aware diurnal imputer, conducts 4-fold expanding temporal cross-validation on 45,253 records, and integrates a production GitHub Actions CI/CD accuracy gate."
    )

    # II. Related Work
    add_heading_1(doc, "II. RELATED WORK & THEORETICAL POSITIONING")
    add_body_p(doc,
        "The design of WeatherPipe is grounded in foundational literature across automated feature engineering and climate modeling:\n"
        "• Murray et al. (ELATE, 2025) explored LLM-assisted code generation for time-series features but suffer from high computational token latency and non-deterministic execution.\n"
        "• Malarkkan et al. (CAFE, 2026) utilized reinforcement learning for causal feature generation under distribution shifts.\n"
        "• Guillaume-Castel, Li, and Sobolowski (2026) emphasized strict chronological train/test splitting, training-only parameter fitting, and reference persistence baselines in atmospheric modeling.\n"
        "• Kanter & Veeramachaneni (DFS, 2015) introduced Deep Feature Synthesis for automated relational features, though combinatorial expansion causes collinearity.\n"
        "• Noboa et al. (2024) demonstrated that isolating feature groups (precipitation, elevation, hydrometric gauges) reveals drivers of model generalization. We adopt their structured ablation methodology.\n"
        "• Banerjee et al. (2018) showed automated feature extraction matches months of manual engineering in days while retaining interpretability compared to black-box PCA."
    )

    # III. System Architecture
    add_heading_1(doc, "III. SYSTEM ARCHITECTURE & METHODOLOGY")
    add_body_p(doc,
        "WeatherPipe executes a sequential, deterministic pipeline designed for reproducibility, operational safety, and zero future leakage."
    )

    # Embed Fig 1 Architecture
    fig1_path = Path("paper_latex/figures/system_architecture.png")
    if fig1_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(6)
        doc.add_picture(str(fig1_path), width=Inches(6.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Fig. 1. End-to-End System Architecture of WeatherPipe, illustrating causal preprocessing, feature extraction, and CI/CD accuracy gate.")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    add_heading_2(doc, "A. Ingestion, Alignment, and Physical Sanity Boundaries")
    add_body_p(doc,
        "Raw timestamps are rounded to the nearest integer hour (1h), deduplicated via mean pooling, and reindexed against a complete date range. "
        "Sentinel codes (-999, 9999, pressure <= 500 hPa) and values exceeding physical thermodynamic bounds are mapped to NaN. "
        "Isolated instrument spikes (|x_t - x_{t-1}| > 10°C and |x_t - x_{t+1}| > 10°C with context difference < 10°C) are removed."
    )

    add_heading_2(doc, "B. Causal Diurnal Climatological Imputation")
    add_body_p(doc,
        "The CausalDiurnalImputer operates strictly causally in two stages:\n"
        "1. Training: Computes empirical diurnal climatology tables mu(month, hour) strictly on the training partition.\n"
        "2. Inference: Imputes missing values at timestamp t occurring k hours after the last valid reading x_{t-k} via:\n"
        "   x_hat(t) = mu(m_t, h_t) + (alpha^k) * (x_{t-k} - mu(m_{t-k}, h_{t-k}))\n"
        "where alpha = 0.90 is the persistence decay rate. Short dropouts (k=1) retain 90% of local atmospheric inertia, while long outages (k >= 48) relax smoothly to the true diurnal climatological mean without future leakage."
    )

    add_heading_2(doc, "C. Circular Wind Decomposition & Feature Engineering")
    add_body_p(doc,
        "Wind direction is decomposed into sin(2*pi*theta / 360) and cos(2*pi*theta / 360), eliminating the 0°/360° discontinuity. "
        "Multi-scale lags (1, 2, 3, 6, 12, 24h), backward rolling statistics (3h and 24h), cyclical hour and day-of-year markers, "
        "Magnus-Tetens dew point, dew point spread, and 3-hour pressure tendency (Delta P_{3h}) are systematically computed. "
        "A collinearity filter drops redundant features with Pearson |r| > 0.98 on the training split."
    )

    # IV. Experimental Evaluation (RQ1)
    add_heading_1(doc, "IV. EXPERIMENTAL EVALUATION & RESULTS (RQ1)")
    add_body_p(doc,
        "The benchmark was executed on 45,253 hourly records (2012–2017) for San Francisco on a 6-hour ahead temperature forecast task (T_{t+6}) "
        "using XGBoost Regressor (120 estimators, max depth 6, lr 0.08) across 4 expanding temporal folds."
    )

    # Table 1: RQ1 Summary Table
    t1 = doc.add_table(rows=5, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_data = [
        ["Forecasting Method", "Preprocessing & Features", "Overall RMSE (°C)", "Overall MAE (°C)", "Gain vs Analyst (%)"],
        ["Persistence Baseline", "Predict current temperature", "5.3442", "4.2236", "—"],
        ["Manual Baseline A (Naive)", "Global mean fill, raw columns", "4.5095", "3.4551", "—"],
        ["Manual Baseline B (Analyst)", "Forward-fill, calendar, 1 lag, 1 roll", "2.1136", "1.5719", "—"],
        ["WeatherPipe (Proposed)", "Causal imputer, multi-scale features", "1.7776", "1.3212", "+15.90%"],
    ]
    for r_idx, row in enumerate(t1_data):
        for c_idx, val in enumerate(row):
            cell = t1.cell(r_idx, c_idx)
            cell.paragraphs[0].text = val
            cell.paragraphs[0].runs[0].font.name = "Times New Roman"
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_margins(cell)
            if r_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True
                set_cell_background(cell, "EEEEEE")
            elif r_idx == 4:
                cell.paragraphs[0].runs[0].font.bold = True
                set_cell_background(cell, "E8F4F8")

    p_t1 = doc.add_paragraph()
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_t1.add_run("TABLE I. Overall 6-Hour Forecast Performance Comparison across 4 Expanding Folds")
    r_t1.font.name = "Times New Roman"
    r_t1.font.size = Pt(8.5)
    r_t1.font.italic = True

    add_body_p(doc,
        "WeatherPipe achieves an overall RMSE of 1.7776°C (R² = 0.9000), outperforming the expert analyst baseline (2.1136°C) by 15.90% and the naive baseline by 60.58%. "
        "A 1,000-iteration bootstrap confidence interval for RMSE improvement is [+0.2984°C, +0.3736°C] (p < 0.001), confirming statistical significance."
    )

    # Embed Fig 2 RQ1 RMSE
    fig2_path = Path("paper_latex/figures/rq1_rmse_comparison.png")
    if fig2_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(6)
        doc.add_picture(str(fig2_path), width=Inches(4.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Fig. 2. 6-Hour forecast RMSE across baselines and the automated pipeline.")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    # V. Feature Ablation Study
    add_heading_1(doc, "V. FEATURE FAMILY ABLATION STUDY")
    add_body_p(doc,
        "Following Noboa et al. (2024), we isolated feature families to assess predictive contributions. "
        "While multi-scale lags provide the strongest individual signal (3.0838°C), combining lags, rolling statistics, cyclical encodings, and thermodynamic interactions achieves the dominant RMSE of 1.9502°C."
    )

    # Embed Fig 3 Ablation
    fig3_path = Path("paper_latex/figures/rq1_ablation_study.png")
    if fig3_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(6)
        doc.add_picture(str(fig3_path), width=Inches(5.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Fig. 3. Feature group ablation study showing predictive RMSE across isolated feature families versus WeatherPipe.")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    # VI. Imputation Benchmark (RQ2)
    add_heading_1(doc, "VI. MISSING DATA IMPUTATION BENCHMARK (RQ2)")
    add_body_p(doc,
        "Known ground-truth values were artificially masked under 10%–20% random dropouts and 3h–24h burst blackouts. "
        "On 24-hour sensor outages, forward-fill gives 4.8859°C RMSE and linear interpolation gives 4.8688°C RMSE. "
        "WeatherPipe's causal diurnal imputer achieves 2.4928°C RMSE (a 49.0% improvement over forward-fill) while being strictly causal and deployable on live data."
    )

    # Embed Fig 4 RQ2 Imputation
    fig4_path = Path("paper_latex/figures/rq2_imputation_rmse.png")
    if fig4_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(6)
        doc.add_picture(str(fig4_path), width=Inches(5.0))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Fig. 4. Imputation fidelity benchmark under burst sensor outages across different gap durations.")
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    # VII. Production CI/CD
    add_heading_1(doc, "VII. PRODUCTION CI/CD & CONTINUOUS MLOPS MONITORING")
    add_body_p(doc,
        "WeatherPipe includes: (1) 11 automated unit and leakage tests, notably test_leakage.py which mutates future data (55°C heatwave) and verifies past features remain bit-for-bit identical; "
        "(2) an automated CI/CD accuracy gate asserting RMSE_auto < RMSE_manual; and (3) streaming weekly replay with Kolmogorov-Smirnov drift detection, successfully quarantining corrupted batches."
    )

    # VIII. Threats to Validity
    add_heading_1(doc, "VIII. THREATS TO VALIDITY & LIMITATIONS")
    add_body_p(doc,
        "1. Point-Station Observations: The pipeline targets localized surface station telemetry rather than 2D spatial reanalysis grids (e.g. ERA5).\n"
        "2. Precipitation Proxy: Precipitation depth (mm) was absent in the raw Kaggle dataset, necessitating reliance on atmospheric proxy variables (humidity, pressure tendency, dew point spread)."
    )

    # IX. Conclusion
    add_heading_1(doc, "IX. CONCLUSION")
    add_body_p(doc,
        "WeatherPipe demonstrates that replacing manual, ad-hoc weather preprocessing with an automated, domain-aware, CI/CD-driven pipeline yields a 15.90% forecast error reduction over expert manual practice, "
        "cuts 24-hour sensor outage imputation error to 2.4928°C without data leakage, and provides robust operational deployment through automated safety gates."
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

