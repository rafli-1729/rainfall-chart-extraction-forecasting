# Total Daily Rainfall Prediction in Singapore

## Project Overview

This project implements an **end-to-end machine learning pipeline** for **forecasting total daily rainfall across cities in Singapore**.

Unlike typical tabular machine learning problems, the raw data provided in this project is **not in a tabular format**. Instead, the dataset consists of **rainfall charts and related files**, which require an additional **data extraction stage** before they can be used for machine learning. As a result, this project covers the full workflow from **data extraction** to **model deployment**.

The primary goal of this project is to **predict total daily rainfall** based on:
- **Location**
- **Date**

with the constraint that **future date predictions are limited by external weather API availability**.

To achieve this, the project includes:
- Automated extraction of rainfall information into machine-learning–friendly tabular data
- Feature engineering and dataset construction
- Model training and evaluation
- A deployment-ready inference pipeline exposed via an API

This structure allows the model to be reused, retrained, and deployed in real-world scenarios rather than being confined to exploratory notebooks.

## Project Structures

```text
rainfall-chart-extraction-forecasting/
│
├── config/
│   ├── config.toml
│   └── config.dev.toml
│
├── data/
│   ├── raw/
│   ├── process/
│   └── clean/
│
├── src/
│   ├── locations.csv
│   ├── app_service.py
│   ├── config.py
│   ├── dataset.py
│   ├── external.py
│   ├── extraction.py
│   ├── features.py
│   ├── model.py
│   ├── observed.py
│   ├── pipeline.py
│   └── schema.py
│
├── api/
│   ├── dependencies.py
│   └── main.py
│
├── ui/
│   └── app.py
│
├── scripts/
│   ├── extract.py
│   ├── build_dataset.py
│   └── train.py
│
├── notebooks/
│   ├── analysis.ipynb
│   └── preprocessing.ipynb
│
├── models/
│   ├── xgb_model.pkl
│   └── feature_order.json
│
├── reports/
│
├── .gitattributes
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```