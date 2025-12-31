from pathlib import Path
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

import streamlit as st
import requests
import json
import duckdb
import re

from datetime import timedelta

import pandas as pd
from streamlit.components.v1 import html

from utils import (
    today_sg,
    render_templates,
    load_styles,
    evaluate_prediction,
    rainfall_intensity,
    forecast_insight_text,
    scenario_insight_text,
    load_html,
    build_line_chart_payload,
    render_component,
    kpi_value,
    format_baseline
)

from src.config import config

assets_dir = config.paths.assets

@st.cache_data()
def load_sql(path):
    with open(path) as f:
        content = f.read()

    blocks = re.split(r"-- name:\s*", content)[1:]
    queries = {}

    for block in blocks:
        name, sql = block.split("\n", 1)
        queries[name.strip()] = sql.strip()

    return queries

# =============================== CONFIG ===============================

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Rainfall Forecasting",
    layout="wide"
)

load_styles(config.paths.styles)

# =============================== HEADER ===============================

st.title("Rainfall Forecasting Dashboard")

render_templates(config.paths.templates/'divider.html')

# ============================== DASHBOARD =============================

con = duckdb.connect(config.paths.database/"rainfall.duckdb")

global_sql = load_sql(config.paths.database/"global_queries.sql")
city_sql = load_sql(config.paths.database/"city_queries.sql")
year_sql = load_sql(config.paths.database/"year_queries.sql")

mae, mse, rmse = con.execute(global_sql['metrics']).fetchone()
r2, = con.execute(global_sql['r2']).fetchone()
epsilon = 5 # mm
zero_rain, = con.execute(global_sql['zero-rain'], [epsilon]).fetchone()

render_component(
    html_path=Path("app/assets/kpi/global.html"),
    css_path=Path("app/assets/kpi/global.css"),
    js_path=Path("app/assets/kpi/script.js"),
    height=110,
    zero_rain_false_rate=zero_rain,
    epsilon=epsilon,
    r2=r2,
    mae=mae,
    rmse=rmse
)

raw_locations = con.execute(global_sql["available_locations"]).fetchall()
available_locations = [
    loc[0] for loc in raw_locations
]

num_observed, = con.execute(global_sql['shape']).fetchone()

col1, col2 = st.columns(2)
with col1:
    selected_location = st.selectbox(
        "Location",
        available_locations,
        key="location"
    )

years = con.execute(
    global_sql["available_years_by_location"],
    [selected_location]
).fetchall()

available_years = [y[0] for y in years]
if not available_years:
    with col2:
        selected_year = st.selectbox(
            "Year",
            available_years,
            key="year"
        )

    st.warning(f"No data available for {selected_location}")
    st.stop()

prev_year = st.session_state.get("year")

if prev_year in available_years:
    default_year = prev_year
else:
    default_year = available_years[-1]

with col2:
    selected_year = st.selectbox(
        "Year",
        available_years,
        index=available_years.index(default_year),
        key="year"
    )

# =========================== DATABASE QUERYING ===========================

available_years = con.execute(global_sql["available_years"]).fetchall()

plot_df = con.execute(
    global_sql['melt_df'],
    [str(selected_location),
     int(selected_year)]
).df()

(
    year_mae, year_rmse,
    year_bias, year_corr,
    year_false_rain,
    year_extreme
) = con.execute(
    year_sql['metrics'],
    [str(selected_location),
    int(selected_year), epsilon]
).fetchone()


(
    _,
    baseline_mae,
    mae_impr_pct,
    _,
    baseline_extreme,
    extreme_impr_pct
) = con.execute(
    year_sql["baseline_comparison"],
    [selected_location, selected_year]
).fetchone()


(
    city_mae, city_rmse,
    city_bias, city_corr,
    city_false_rain,
    city_extreme
) = con.execute(
    city_sql['metrics'],
    [str(selected_location), epsilon]
).fetchone()

con.close()

# ================================= CHART & KPIs =================================

render_component(
    html_path=Path("app/assets/kpi/city.html"),
    css_path=Path("app/assets/kpi/city.css"),
    js_path=Path("app/assets/kpi/script.js"),
    location= selected_location,
    height=45,
    city_mae=city_mae,
    city_bias=city_bias,
    city_false_rain=city_false_rain
)

year_mae = kpi_value(year_mae, "mm")
year_bias = kpi_value(year_bias, "mm")
year_extreme = kpi_value(year_extreme, "mm")
year_false_rain = kpi_value(year_false_rain, "%")


year_mae_baseline_pct, year_mae_baseline_class = (
    format_baseline(mae_impr_pct)
)

year_extreme_baseline_pct, year_extreme_baseline_class = (
    format_baseline(extreme_impr_pct)
)

render_component(
    html_path=Path("app/assets/kpi/year.html"),
    css_path=Path("app/assets/kpi/year.css"),
    js_path=Path("app/assets/kpi/script.js"),
    height=250,

    location=selected_location,
    year=selected_year,

    year_mae_num=year_mae["num"],
    year_mae_text=year_mae["text"],
    year_mae_unit=year_mae["unit"],

    year_bias_num=year_bias["num"],
    year_bias_text=year_bias["text"],
    year_bias_unit=year_bias["unit"],

    year_extreme_num=year_extreme["num"],
    year_extreme_text=year_extreme["text"],
    year_extreme_unit=year_extreme["unit"],

    year_false_rain_num=year_false_rain["num"],
    year_false_rain_text=year_false_rain["text"],
    year_false_rain_unit=year_false_rain["unit"],

    year_mae_baseline_pct=year_mae_baseline_pct,
    year_mae_baseline_class=year_mae_baseline_class,

    year_extreme_baseline_pct=year_extreme_baseline_pct,
    year_extreme_baseline_class=year_extreme_baseline_class
)

payload = build_line_chart_payload(plot_df)

render_component(
    html_path=assets_dir / "chart/index.html",
    css_path=assets_dir / "chart/style.css",
    js_path=assets_dir / "chart/script.js",
    height=300,
    PAYLOAD_JSON=json.dumps(payload),
    TITLE="Recent Rainfall Trend",
    SUBTITLE=f"{selected_location} • {selected_year}"
)

html(load_html(config.paths.templates/'divider.html'), height=10)

# =============================== TRY THE MODEL ===============================

st.subheader("Try the Model")
st.caption("Test the rainfall prediction model using different modes and inputs.")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    location = st.selectbox("Location", config.features.locations)

with col2:
    date = st.date_input("Date")

with col3:
    mode = st.selectbox(
        "Select Mode",
        ["Evaluation", "Forecast", "Random Scenario"],
    )

date_str = date.strftime("%Y-%m-%d")
date_formatted = date.strftime("%d %b %Y")

# ================================== FORECAST TAB ==================================

if mode == 'Random Scenario':
    col1, col2, col3 = st.columns(3)

    with col1:
        r30 = st.number_input("Highest 30-min Rainfall (mm)", 0.0, 200.0, 1.0, step=0.5)
        min_temp = st.number_input("Min Temperature (°C)", 0.0, 40.0, 24.0)

    with col2:
        r60 = st.number_input("Highest 60-min Rainfall (mm)", 0.0, 300.0, 1.0, step=0.5)
        max_temp = st.number_input("Max Temperature (°C)", 0.0, 40.0, 30.0)

    with col3:
        r120 = st.number_input("Highest 120-min Rainfall (mm)", 0.0, 500.0, 1.0, step=0.5)
        mean_temp = st.number_input("Mean Temperature (°C)", 0.0, 40.0, 27.0)

    col1, col2 = st.columns(2)
    with col1:
        mean_wind = st.number_input("Mean Wind Speed (km/h)", 0.0, 50.0, 8.0)
    with col2:
        max_wind = st.number_input("Max Wind Speed (km/h)", 0.0, 100.0, 15.0)

    if st.button("Predict", width=1100, type='primary'):
        payload = {
            "features": {
                "location": location,
                "date": date_str,
                "mean_temperature_c": mean_temp,
                "maximum_temperature_c": max_temp,
                "minimum_temperature_c": min_temp,
                "mean_wind_speed_kmh": mean_wind,
                "max_wind_speed_kmh": max_wind,
                "highest_30_min_rainfall_mm": r30,
                "highest_60_min_rainfall_mm": r60,
                "highest_120_min_rainfall_mm": r120,
            }
        }

        result_slot = st.empty()
        with result_slot:
            render_templates(config.paths.templates/"loading.html")

        res = requests.post(f"{API_BASE}/random", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = scenario_insight_text(rain_mm, level)

        scenario_inputs = {
            "Feature Source": data["meta"]["feature_source"],
        }

        with result_slot:
            render_templates(
                config.paths.templates/"random_result.html",
                rainfall_mm=f"{rain_mm:.1f}",
                intensity_level=level,
                intensity_label=label,
                location=data["input"]["location"]
            )

# ================================= FORECAST MODE =================================

if mode == 'Forecast':

    is_valid = True
    error_msg = None

    today = today_sg()
    max_forecast = today + timedelta(days=14)

    if date < today:
        is_valid = False
        error_msg = "Forecast date cannot be in the past."

    elif date > max_forecast:
        is_valid = False
        error_msg = (
            f"Forecast only available up to {max_forecast}"
        )

    if not is_valid:
        st.warning(error_msg)

    if st.button("Forecast", disabled=not is_valid, width=1100, type='primary'):
        payload = {
            "location": location,
            "date": date_str
        }

        result_slot = st.empty()
        with result_slot:
            render_templates(config.paths.templates/"loading.html")

        res = requests.post(f"{API_BASE}/forecast", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = forecast_insight_text(rain_mm, level)

        with result_slot:
            render_templates(
                config.paths.templates/"forecast_result.html",
                rainfall_mm=f"{rain_mm:.1f}",
                intensity_level=level,
                intensity_label=label,
                insight_text=insight,
                location=data["input"]["location"],
                date=date_formatted
            )


# ================================= EVALUATION MODE =================================

if mode == 'Evaluation':
    is_valid = True
    error_msg = None

    today = today_sg()
    selected_date = date

    if selected_date >= today:
        is_valid = False
        error_msg = (
            "Evaluation requires observed data. "
            "Please select a date before today (SGT)."
        )

    if not is_valid:
        st.warning(error_msg)

    if st.button("Evaluate", disabled=not is_valid, width=1100, type='primary'):
        payload = {
            "location": location,
            "date": selected_date.strftime("%Y-%m-%d")
        }

        result_slot = st.empty()
        with result_slot:
            render_templates(config.paths.templates/"loading.html")

        res = requests.post(f"{API_BASE}/evaluate", json=payload)
        data = res.json()

        mode = data["mode"]
        location = data["input"]["location"]
        date = data["input"]["date"]
        pred_mm = data["prediction"]["daily_rainfall_mm"]
        obs_mm = data["comparison"]["observed_daily_rainfall_mm"]
        error_mm = data["comparison"]["error_mm"]

        with result_slot:
            if data["comparison"] is None:
                render_templates(
                    config.paths.templates/"evaluation_unavailable.html",
                    location=location,
                    date=date
                )
            else:
                eval_result = evaluate_prediction(pred_mm, obs_mm)
                render_templates(
                    config.paths.templates/"evaluate_result.html",
                    rainfall_mm=f"{eval_result['predicted_mm']:.1f}",
                    obs_rainfall_mm=f"{eval_result['observed_mm']:.1f}",
                    error_pct=f"{eval_result['relative_error_pct']:.1f}",
                    error_level=eval_result["severity"],
                    insight_text=eval_result["insight_text"],
                    location=location,
                    date=date_formatted
                )

render_templates("app/assets/footer.html")