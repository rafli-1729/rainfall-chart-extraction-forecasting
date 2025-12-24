# ui/app.py

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def today_sg():
    return datetime.now(ZoneInfo("Asia/Singapore")).date()

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Rainfall Forecasting App",
    layout="centered"
)

st.title("🌧️ Rainfall Forecasting & Evaluation")

# ======================================================
# MODE SELECTION
# ======================================================

mode = st.selectbox(
    "Select Mode",
    ["Random (Scenario)", "Forecast", "Evaluation"]
)

st.divider()

# ======================================================
# COMMON INPUTS
# ======================================================

location = st.text_input("Location", value="Admiralty")
date = st.date_input("Date")

date_str = date.strftime("%Y-%m-%d")

# ======================================================
# RANDOM MODE
# ======================================================

if mode == "Random (Scenario)":
    st.subheader("Scenario Inputs")

    col1, col2 = st.columns(2)

    with col1:
        mean_temp = st.number_input("Mean Temperature (°C)", 0.0, 40.0, 27.0)
        min_temp = st.number_input("Min Temperature (°C)", 0.0, 40.0, 24.0)
        mean_wind = st.number_input("Mean Wind Speed (km/h)", 0.0, 50.0, 8.0)

    with col2:
        max_temp = st.number_input("Max Temperature (°C)", 0.0, 40.0, 30.0)
        max_wind = st.number_input("Max Wind Speed (km/h)", 0.0, 100.0, 15.0)

    r30 = st.number_input("Highest 30-min Rainfall (mm)", 0.0, 200.0, 10.0)
    r60 = st.number_input("Highest 60-min Rainfall (mm)", 0.0, 300.0, 18.0)
    r120 = st.number_input("Highest 120-min Rainfall (mm)", 0.0, 500.0, 25.0)

    if st.button("Run Scenario"):
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

        res = requests.post(f"{API_BASE}/random", json=payload)
        st.write(f"Status code: {res.status_code}")

        try:
            st.json(res.json())
        except Exception:
            st.error("API did not return valid JSON")
            st.subheader("Raw response:")
            st.code(res.text)


# ======================================================
# FORECAST MODE
# ======================================================

elif mode == "Forecast":

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

    if st.button("Run Forecast"):
        payload = {
            "location": location,
            "date": date_str
        }
        res = requests.post(f"{API_BASE}/forecast", json=payload)
        st.write(f"Status code: {res.status_code}")

        try:
            st.json(res.json())
        except Exception:
            st.error("API did not return valid JSON")
            st.subheader("Raw response:")
            st.code(res.text)


# ======================================================
# EVALUATION MODE
# ======================================================

elif mode == "Evaluation":

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

    if st.button("Run Evaluation", disabled=not is_valid):
        payload = {
            "location": location,
            "date": selected_date.strftime("%Y-%m-%d")
        }

        res = requests.post(f"{API_BASE}/evaluate", json=payload)

        st.write(f"Status code: {res.status_code}")
        try:
            st.json(res.json())
        except Exception:
            st.error("API did not return valid JSON")
            st.code(res.text)