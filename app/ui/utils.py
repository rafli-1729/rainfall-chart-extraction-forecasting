from warnings import filterwarnings
filterwarnings("ignore")

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import re
import base64

import sys
from pathlib import Path
from streamlit.components.v1 import html

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def today_sg():
    return datetime.now(ZoneInfo("Asia/Singapore")).date()

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def render_template(text: str, **kwargs) -> str:
    for key, value in kwargs.items():
        if value == "\n":
            continue

        text = re.sub(
            rf"\{{\{{\s*{key}\s*\}}\}}",
            str(value),
            text
        )

    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def load_css(path: Path, **kwargs) -> str:
    css = path.read_text(encoding="utf-8")
    css = render_template(css, **kwargs)
    return f"<style>{css}</style>"


def load_js(path: Path, **kwargs) -> str:
    js = path.read_text(encoding="utf-8")
    js = render_template(js, **kwargs)
    return f"<script>{js}</script>"


def load_html(path: Path, **kwargs) -> str:
    html = path.read_text(encoding="utf-8")
    html = render_template(html, **kwargs)
    return html


def render_component(
    *,
    html_path: Path,
    css_path: Path | None = None,
    js_path: Path | None = None,
    height: int = 300,
    **kwargs
):
    parts = []

    if css_path:
        parts.append(load_css(css_path, **kwargs))

    parts.append(load_html(html_path, **kwargs))

    if js_path:
        parts.append(load_js(js_path, **kwargs))

    doc = "\n".join(parts)
    html(doc, height=height, scrolling=False)


def render_templates(path: str, **kwargs):
    html = Path(path).read_text()
    for key, value in kwargs.items():
        if value == "\n":
            continue

        html = re.sub(
            rf"\{{\{{\s*{key}\s*\}}\}}",
            str(value),
            html
        )
        html = html.replace(f"{{{{ {key} }}}}", str(value))

    html = re.sub(r"\n\s*\n+", "\n", html)

    st.markdown(html, unsafe_allow_html=True)


def kpi_value(value, unit):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {
            "num": None,
            "text": "Not Available",
            "unit": ""
        }
    return {
        "num": float(value),
        "text": f"{value:.2f}",
        "unit": unit
    }


def format_baseline(improvement_pct):
    if improvement_pct is None:
        return "", ""

    val = float(improvement_pct)
    if not np.isfinite(val):
        return "", ""

    text = f"{val:+.0f}%"
    css_class = "negative" if val < 0 else ""

    return text, css_class


def build_line_chart_payload(plot_df: pd.DataFrame) -> dict:
    df = plot_df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

    dates = df["date"].dt.strftime("%Y-%m-%d").unique().tolist()

    series = {}
    for t, g in df.groupby("type"):
        tmp = (
            g.assign(Date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
             .set_index("date")
             .reindex(dates)
        )
        series[t] = [
            None if pd.isna(v) else float(v)
            for v in tmp["rainfall"].tolist()
        ]

    return {"dates": dates, "series": series}


def load_styles(folder: str):
    css = ""

    for css_file in sorted(Path(folder).glob("*.css")):
        css += css_file.read_text() + "\n"

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def evaluate_prediction(pred, obs, eps=1.0):
    diff = pred - obs
    abs_error = abs(diff)
    rel_error_pct = abs_error / max(obs, eps) * 100

    if diff > 0:
        direction = "overprediction"
        direction_text = "Predicted Rainfall is higher than observed."
    elif diff < 0:
        direction = "underprediction"
        direction_text = "Predicted Rainfall is lower than observed."
    else:
        direction = "exact"
        direction_text = "Prediction closely matches observed Rainfall."

    if obs <= 5:
        regime = "drizzle"
        regime_text = (
            "Rainfall is within the drizzle regime, where small absolute differences "
            "are common and generally low-impact."
        )
    elif obs <= 30:
        regime = "moderate"
        regime_text = (
            "Rainfall falls within a moderate range, where prediction accuracy "
            "can influence short-term planning."
        )
    else:
        regime = "heavy"
        regime_text = (
            "Rainfall is in the heavy regime, where prediction errors may have "
            "significant real-world impact."
        )

    if regime == "drizzle":
        if abs_error <= 2:
            severity = "low"
            severity_text = "Deviation is minimal and unlikely to have practical impact."
        else:
            severity = "moderate"
            severity_text = (
                "Relative deviation appears large, but absolute Rainfall remains low."
            )

    else:
        if rel_error_pct <= 20:
            severity = "low"
            severity_text = "Prediction is highly reliable for this Rainfall level."
        elif rel_error_pct <= 50:
            severity = "moderate"
            severity_text = (
                "Prediction shows noticeable deviation that may affect planning decisions."
            )
        else:
            severity = "high"
            severity_text = (
                "Prediction deviates substantially from observed Rainfall."
            )

    insight_text = (
        f"{severity_text} {direction_text} {regime_text}"
    )

    return {
        "predicted_mm": pred,
        "observed_mm": obs,
        "absolute_error_mm": abs_error,
        "relative_error_pct": round(rel_error_pct, 1),

        "direction": direction,
        "regime": regime,
        "severity": severity,

        "insight_text": insight_text
    }

def rainfall_intensity(rain_mm: float):
    x = float(rain_mm)
    if x <= 1.0:
        return "drizzle", "No rain / Drizzle"
    if x <= 10.0:
        return "light", "Light rain"
    if x <= 30.0:
        return "moderate", "Moderate rain"
    return "heavy", "Heavy rain"


def forecast_insight_text(rain_mm: float, intensity_level: str) -> str:
    if intensity_level == "drizzle":
        return "Minimal Rainfall is expected. No significant impact is anticipated under typical conditions."
    if intensity_level == "light":
        return "Light Rainfall is expected. Conditions may feel damp, but impacts are generally limited."
    if intensity_level == "moderate":
        return "Moderate Rainfall is expected. Consider short-term planning impacts, especially around commute hours."
    return "Heavy Rainfall is expected. Be alert for localized flooding risk and changing conditions."


def scenario_insight_text(rain_mm: float, intensity_level: str) -> str:
    if intensity_level == "drizzle":
        return "Under the specified inputs, the model indicates minimal Rainfall. This reflects a low-rain response scenario."
    if intensity_level == "light":
        return "Under the specified inputs, the model predicts light Rainfall. This scenario suggests mild rain conditions."
    if intensity_level == "moderate":
        return "Under the specified inputs, the model predicts moderate Rainfall. This scenario can affect short-term activity planning."
    return "Under the specified inputs, the model predicts heavy Rainfall. This scenario indicates elevated rain intensity."


def render_evaluation_page():
    st.header("Evaluation results")
    st.caption("Observed vs predicted Rainfall")

    # SEMI DASHBOARD
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", "1.24 mm")

    with col2:
        st.metric("RMSE", "2.01 mm")

    with col3:
        st.metric("Bias", "-0.32 mm")

    st.divider()

    st.subheader("Prediction vs Observation")
    st.line_chart(eval_df)