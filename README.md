# Weather Forecasting
Predicting daily rainfall totals using historical weather records, engineered temporal features, and gradient boosting models.

This project uses data provided within the SSF SSDS UNS framework. Because the dataset is used for academic and competition-related purposes, the raw files cannot be publicly displayed or redistributed in this repository.

---

## Project Overview

This repository presents a structured machine learning workflow for forecasting daily rainfall across multiple monitoring locations. The dataset consists of raw weather logs gathered from several independent sources, including measurements of precipitation, temperature, and other atmospheric variables. Since these inputs vary in format and structure, the project begins by consolidating them into a unified dataset and preparing them for time-series modeling. The forecasting task requires capturing seasonal behavior, short-term fluctuations, and environmental interactions, and the workflow is designed to reveal these dynamics through careful preprocessing, feature engineering, and model evaluation.

---

## Workflow Summary

### Data Cleaning

The workflow starts by loading a collection of raw CSV files and transforming them into a coherent time-series dataset. Column names are standardized, timestamps are parsed into a consistent format, and missing or irregular values are examined closely. Because rainfall data can be sensitive to measurement errors and discontinuities, this stage includes checks for outliers and structural inconsistencies across locations. The various datasets are merged using dates as the central key, resulting in a clean dataset that aligns the available weather indicators across time.

### Exploratory Data Analysis

The exploratory phase investigates the behavior of rainfall and its relationship with other meteorological variables. Visual and statistical summaries highlight seasonal cycles, monthly variation, and differences across geographical locations. Correlations between temperature, humidity, and precipitation help identify which features are most informative for forecasting. These insights provide a foundation for the feature engineering stage by clarifying which aspects of the weather data contribute most strongly to rainfall patterns.

### Feature Engineering

The project develops a set of time-based features that capture both short-term and long-term weather dynamics. These include calendar attributes such as the month, day, and weekday, as well as lagged rainfall values and rolling window statistics computed from several weather variables. These engineered features provide the model with richer temporal structure, allowing it to recognize trends, seasonality, and recent weather conditions. By explicitly encoding temporal behavior, the dataset becomes more suitable for learning predictive patterns.

### Modeling

Forecasting is performed using LightGBM, a gradient boosting model well-suited for tabular time-series data. The modeling process incorporates time-series cross-validation to preserve chronological order and evaluate the model fairly. Hyperparameters are tuned using Optuna, ensuring that the configuration balances accuracy and generalization. Feature importance analysis is used to understand which weather indicators influence predictions most strongly, providing interpretability and guiding future refinement of the feature set.

### Evaluation and Output

Model predictions are assessed using regression metrics such as Mean Squared Error (MSE), along with visual comparisons between predicted and observed rainfall values. Additional checks ensure that the model produces outputs consistent with physical expectations, such as maintaining realistic temporal patterns. Final predictions are formatted into output files suitable for downstream analysis or submission to evaluation systems used in the SSF SSDS UNS workflow. This ensures that the forecasting pipeline is practical, consistent, and ready for real-world use.

---

## Purpose of the Repository

This repository serves as a complete applied machine learning workflow for daily rainfall forecasting. It demonstrates how raw weather logs can be cleaned, enriched with informative features, and modeled using modern gradient boosting techniques. Through structured processing steps and careful handling of time-series characteristics, the project showcases an effective approach for turning meteorological measurements into actionable forecasts. Developed within the context of the SSF SSDS UNS event—Sebelas Maret Statistics Fair and Sebelas Maret Statistics Data Science—it reflects the kind of analytical work encouraged in applied statistics and data science competitions.

---

This project was created by **Muhammad Rafli Azrarsyah**, a third-year Actuarial Science student at Universitas Gadjah Mada, who enjoys exploring data and uncovering insights through applied modeling and statistical analysis.


```text
rainfall-chart-extraction-forecasting
├── data
│   ├── raw
│   │   ├── Train
│   │   │   ├── Admiralty
│   │   │   ├── Ang_Mo_Kio
│   │   │   ├── Bukit_Panjang
│   │   │   └── ...
│   │   │       ├── Data_Gabungan_Lainnya_YYYY.csv
│   │   │       └── Plot_Daily_Rainfall_Total_mm_YYYY.png
│   │   └── Test
│   │       ├── Admiralty
│   │       ├── Ang_Mo_Kio
│   │       ├── Bukit_Panjang
│   │       └── ...
│   │           ├── Data_Gabungan_Lainnya_YYYY.csv
│   │           └── Plot_Daily_Rainfall_Total_mm_YYYY.png
│   ├── process
│   │   ├── extract
│   │   │   └── *.csv
│   │   └── merge
│   │       ├── train
│   │       │   └── *.csv
│   │       └── test
│   │           └── *.csv
│   └── clean
│       └── *.csv
├── notebooks
│   ├── 00_merge.ipynb
│   ├── 01_extract.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_train.ipynb
├── reports
│   ├── executive_summary.md
│   ├── final_report.pdf
│   └── figures
│       └── *.png
├── src
│   ├── cleaning.py
│   ├── exploration.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── __init__.py
├── requirements.txt
└── README.md
```