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

from render import (
    load_styles,
    render_component,
    load_html,
)

from utils import (
    today_sg,
    evaluate_prediction,
    rainfall_intensity,
    forecast_insight_text,
    scenario_insight_text,
    build_line_chart_payload,
    build_error_chart_payload,
    build_timeseries_payload,
    kpi_value,
    format_baseline,
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

st.markdown(load_html(config.paths.templates/'divider.html'), unsafe_allow_html=True)

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

weekly_df = con.execute(
    year_sql["weekly_timeseries"],
    [selected_location, selected_year]
).df()

daily_df = con.execute(
    year_sql["daily_timeseries"],
    [selected_location, selected_year]
).df()

weekly_df["date"] = pd.to_datetime(weekly_df["date"])
daily_df["date"] = pd.to_datetime(daily_df["date"])

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

error_df = con.execute(
    year_sql['plot_error'],
    [str(selected_location),
    int(selected_year)]
).df()


error_distribution = con.execute(
    year_sql["error_distribution"],
    [str(selected_location),
    int(selected_year)]
).df()


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
    height=240,

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

if not daily_df["cv_predicted_mm"].isna().all():
    payload_ts = build_timeseries_payload(
        weekly_df,
        series_map={
            "Observed": "observed_mm",
            "Extracted": "extracted_mm",
            "Predicted": "cv_predicted_mm",
        }
    )

    calibration_df = daily_df[["extracted_mm", "cv_predicted_mm"]].dropna()

    payload_calibration = {
        "x": calibration_df["extracted_mm"].astype(float).tolist(),
        "y": calibration_df["cv_predicted_mm"].astype(float).tolist()
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        render_component(
            html_path=assets_dir / "chart/line.html",
            css_path=assets_dir / "chart/style.css",
            js_path=assets_dir / "chart/line.js",
            height=320,
            PAYLOAD_JSON=json.dumps(payload_ts),
            TITLE="Average Weekly Rainfall Trend",
            SUBTITLE=f"{selected_location} • {selected_year}"
        )

    with col2:
        render_component(
            html_path=assets_dir / "chart/calibration.html",
            css_path=assets_dir / "chart/style.css",
            js_path=assets_dir / "chart/calibration.js",
            height=320,
            PAYLOAD_JSON=json.dumps(payload_calibration),
            TITLE="Predicted vs Extracted",
            SUBTITLE=""
        )

    payload_error = {
        "dates": weekly_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "bias": (weekly_df["observed_mm"] - weekly_df["extracted_mm"]).tolist()
    }

    render_component(
        html_path=assets_dir / "chart/error.html",
        css_path=assets_dir / "chart/style.css",
        js_path=assets_dir / "chart/error.js",
        height=320,
        PAYLOAD_JSON=json.dumps(payload_error),
        TITLE="Average Weekly Bias",
        SUBTITLE=f"{selected_location} • {selected_year}"
    )

    p90 = weekly_df["extracted_mm"].quantile(0.9)

    extreme_df = daily_df[daily_df["extracted_mm"] >= p90]
    payload_extreme = build_timeseries_payload(
        extreme_df,
        series_map={
            "Extreme Error": "abs_error"
        }
    )

    payload_dist = {
        "values": daily_df["abs_error"]
            .dropna()
            .astype(float)
            .tolist()
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        render_component(
            html_path=assets_dir / "chart/error.html",
            css_path=assets_dir / "chart/style.css",
            js_path=assets_dir / "chart/extreme.js",
            height=320,
            PAYLOAD_JSON=json.dumps(payload_extreme),
            TITLE="Daily Extreme Error",
            SUBTITLE=f"{selected_location} • {selected_year}"
        )

    with col2:
        render_component(
            html_path=assets_dir / "chart/error_dist.html",
            css_path=assets_dir / "chart/style.css",
            js_path=assets_dir / "chart/error_dist.js",
            height=320,
            PAYLOAD_JSON=json.dumps(payload_dist),
            TITLE="Daily Error Distribution",
            SUBTITLE=f"{selected_location} • {selected_year}"
        )
else:
    payload_ts = build_timeseries_payload(
        weekly_df,
        series_map={
            "Observed": "observed_mm",
            "Extracted": "extracted_mm",
            "Predicted": "cv_predicted_mm",
        }
    )
    render_component(
        html_path=assets_dir / "chart/line.html",
        css_path=assets_dir / "chart/style.css",
        js_path=assets_dir / "chart/line.js",
        height=320,
        PAYLOAD_JSON=json.dumps(payload_ts),
        TITLE="Average Weekly Rainfall Trend",
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
            st.markdown(load_html(config.paths.templates/'loading.html'), unsafe_allow_html=True)

        res = requests.post(f"{API_BASE}/random", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = scenario_insight_text(rain_mm, level)

        scenario_inputs = {
            "Feature Source": data["meta"]["feature_source"],
        }

        with result_slot:
            st.markdown(load_html(
                config.paths.templates/"random_result.html",
                rainfall_mm=f"{rain_mm:.1f}",
                intensity_level=level,
                intensity_label=label,
                location=data["input"]["location"]
            ), unsafe_allow_html=True)

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
            st.markdown(load_html(config.paths.templates/'loading.html'), unsafe_allow_html=True)

        res = requests.post(f"{API_BASE}/forecast", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = forecast_insight_text(rain_mm, level)

        with result_slot:
            st.markdown(load_html(
                config.paths.templates/"forecast_result.html",
                rainfall_mm=f"{rain_mm:.1f}",
                intensity_level=level,
                intensity_label=label,
                insight_text=insight,
                location=data["input"]["location"],
                date=date_formatted
            ), unsafe_allow_html=True)


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
            st.markdown(load_html(config.paths.templates/'loading.html'), unsafe_allow_html=True)

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
                st.markdown(load_html(
                    config.paths.templates/"evaluation_unavailable.html",
                    location=location,
                    date=date
                ), unsafe_allow_html=True)
            else:
                eval_result = evaluate_prediction(pred_mm, obs_mm)
                st.markdown(load_html(
                    config.paths.templates/"evaluate_result.html",
                    rainfall_mm=f"{eval_result['predicted_mm']:.1f}",
                    obs_rainfall_mm=f"{eval_result['observed_mm']:.1f}",
                    error_pct=f"{eval_result['relative_error_pct']:.1f}",
                    error_level=eval_result["severity"],
                    insight_text=eval_result["insight_text"],
                    location=location,
                    date=date_formatted
                ), unsafe_allow_html=True)

st.markdown(load_html(assets_dir/"footer.html"), unsafe_allow_html=True)