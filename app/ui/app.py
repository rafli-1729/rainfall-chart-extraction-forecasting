from pathlib import Path
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))

import streamlit as st
import requests
import json
import uuid
import duckdb
import re

from datetime import timedelta, datetime
import pandas as pd
import altair as alt
from streamlit.components.v1 import html

from utils import (
    today_sg,
    render_template,
    load_css,
    evaluate_prediction,
    rainfall_intensity,
    forecast_insight_text,
    scenario_insight_text,
    render_vega_chart,
    render_kpi_html
)

from src.config import config

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
    page_title="Rainfall Forecasting App",
    layout="wide",
)

load_css(config.paths.styles)

# =============================== HEADER ===============================

st.title("🌧️ Rainfall Forecasting Dashboard")

render_template(config.paths.templates/'divider.html')

# ============================== DASHBOARD =============================

con = duckdb.connect(config.paths.database/"rainfall.duckdb")
sql = load_sql(config.paths.database/"queries.sql")

raw_available_locations = con.execute(sql["available_locations"]).fetchall()
available_locations = [
    loc[0].replace("_", " ")
    for loc in raw_available_locations
]
available_years = con.execute(sql["available_years"]).fetchall()
num_observed = con.execute(sql['shape']).df().loc[0, "n"]

available_years = [y[0] for y in available_years]

render_kpi_html(
    "app/assets/kpi.html",
    locations_covered=len(available_locations),
    evaluation_records=num_observed,
    mae=6.02,
    rmse=12.45
)

render_template(config.paths.templates/'divider.html')

st.subheader("Recent Rainfall Trend")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    selected_location = st.selectbox(
        "Location", available_locations, key=1
        )

with col2:
    selected_year = st.selectbox(
        "Year", available_years, index=len(available_years) - 1
    )

with col3:
    selected_types = st.multiselect(
        "Series",
        options=["Observed", "Predicted", "Extracted"],
        default=["Observed", "Predicted"]
    )

if len(selected_types) == 1:
    st.caption("Tip: Select multiple series to compare.")

# =========================== DATABASE QUERYING ===========================

sql = load_sql(config.paths.database/"queries.sql")

available_years = con.execute(sql["available_years"]).fetchall()
plot_df = con.execute(
    sql['melt_df'],
    [str(selected_location.replace(" ", "_")),
     int(selected_year), str(selected_types)]
).df()

con.close()

# ================================= CHART =================================

hover = alt.selection_point(
    fields=["date"],
    nearest=True,
    on="pointerover",
    clear="pointerout"
)

base = (
    alt.Chart(plot_df)
    .mark_line(interpolate="monotone", strokeWidth=1.75)
    .encode(
        x=alt.X("date:T", axis=alt.Axis(title=None)),
        y=alt.Y("Rainfall:Q", title="Rainfall (mm)"),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(
                domain=["Observed", "Predicted", "Extracted"],
                range=["#0f172a", "#60a5fa", "#b91c1c"]
            ),
            legend=alt.Legend(
                orient="bottom",
                title=None,
                values=selected_types
            )
        ),
        strokeDash=alt.condition(
            alt.datum.Type == "Predicted",
            alt.value([4, 4]),
            alt.value([1, 0])
        )
    )
)

points = (
    alt.Chart(plot_df)
    .mark_point(
        size=20,
        filled=True
    )
    .encode(
        x="date:T",
        y="Rainfall:Q",
        color=alt.Color("Type:N", legend=None),
        opacity=alt.condition(
            hover,
            alt.value(1),
            alt.value(0)
        )
    )
    .add_params(hover)
)

rule = (
    alt.Chart(plot_df)
    .mark_rule(color="#94a3b8", strokeWidth=0)
    .encode(
        x="date:T",
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Type:N", title=""),
            alt.Tooltip("Rainfall:Q", title="Rainfall (mm)", format=".2f")
        ]
    )
    .transform_filter(hover)
)

chart = (
    alt.layer(
        base,
        points,
        rule
    )
    .properties(height=300)
    .configure(background='transparent')
    .configure_axis(
        gridColor="#e5e7eb",
        tickColor="#94a3b8",
        labelColor="#475569",
        titleColor="#475569"
    )
    .configure_legend(
        orient="bottom",
        labelColor="#475569",
        titleColor="#475569"
    )
)

insight_html = " "
if "Observed" in selected_types and "Predicted" in selected_types:
    obs = plot_df[plot_df["Type"] == "Observed"]
    pred = plot_df[plot_df["Type"] == "Predicted"]

    wide = (
        plot_df
        .pivot(index="date", columns="Type", values="Rainfall")
        .reset_index()
    )
    obs_mm = wide['Observed']
    pred_mm = wide['Predicted']

    wide["error"] = pred_mm - obs_mm
    err = wide["Predicted"] - wide["Observed"]

    mean_err = err.mean()
    std_err = err.std()
    mae = err.abs().mean()

    under_pct = (err < 0).mean() * 100
    over_pct = (err > 0).mean() * 100

    p90_obs = obs_mm.quantile(0.9)
    extreme_err = err[obs_mm >= p90_obs].mean()

    if not obs.empty and not pred.empty:
        bias = "underestimation" if mean_err < 0 else "overestimation"

        insight_html = f"""
        The model exhibits a <b>systematic {bias} bias</b>
        (mean error: <b>{mean_err:.2f} mm</b>), with an average absolute deviation
        of <b>{mae:.2f} mm</b>. Error variability (<b>{std_err:.2f} mm</b>)
        indicates degraded performance during high-rainfall periods,
        particularly beyond the <b>90th percentile</b>.
        """

spec = chart.to_dict()
spec["width"] = "container"
spec["autosize"] = {
    "type": "fit",
    "contains": "padding"
}

spec_json = json.dumps(spec)
chart_id = f"chart_{uuid.uuid4().hex}"

render_vega_chart(chart, insight_html=insight_html)

render_template(config.paths.templates/'divider.html')

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
    st.subheader("Scenario Inputs")

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
            render_template(config.paths.templates/"loading.html")

        res = requests.post(f"{API_BASE}/random", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = scenario_insight_text(rain_mm, level)

        scenario_inputs = {
            "Feature Source": data["meta"]["feature_source"],
        }

        with result_slot:
            render_template(
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
            render_template(config.paths.templates/"loading.html")

        res = requests.post(f"{API_BASE}/forecast", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = forecast_insight_text(rain_mm, level)

        with result_slot:
            render_template(
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
            render_template(config.paths.templates/"loading.html")

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
                render_template(
                    config.paths.templates/"evaluation_unavailable.html",
                    location=location,
                    date=date
                )
            else:
                eval_result = evaluate_prediction(pred_mm, obs_mm)
                render_template(
                    config.paths.templates/"evaluate_result.html",
                    rainfall_mm=f"{eval_result['predicted_mm']:.1f}",
                    obs_rainfall_mm=f"{eval_result['observed_mm']:.1f}",
                    error_pct=f"{eval_result['relative_error_pct']:.1f}",
                    error_level=eval_result["severity"],
                    insight_text=eval_result["insight_text"],
                    location=location,
                    date=date_formatted
                )

render_template("app/assets/footer.html")