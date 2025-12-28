from warnings import filterwarnings
filterwarnings('ignore')

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

import re
import base64

import json
import uuid

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

def render_template(path: str, **kwargs):
    html = Path(path).read_text()
    for key, value in kwargs.items():
        if value == '\n':
            continue

        html = re.sub(
            rf"\{{\{{\s*{key}\s*\}}\}}",
            str(value),
            html
        )
        html = html.replace(f"{{{{ {key} }}}}", str(value))

    html = re.sub(r"\n\s*\n+", "\n", html)

    st.markdown(html, unsafe_allow_html=True)


def render_kpi_html(path: str, height: int = 220, **kwargs):
    template = Path(path).read_text()

    for key, value in kwargs.items():
        template = re.sub(
            rf"\{{\{{\s*{key}\s*\}}\}}",
            str(value),
            template
        )

    html(
        template,
        height=height,
        scrolling=False
    )


def render_vega_chart(chart, insight_html: str, height=460):
    spec = chart.to_dict()
    spec["width"] = "container"
    spec["autosize"] = {"type": "fit", "contains": "padding"}

    chart_id = f"chart_{uuid.uuid4().hex}"

    html(
        f"""
        <div class="chart-js-card">
            <div id="{chart_id}" style="width: 100%;"></div>

            <div class="chart-insight">
                {insight_html}
            </div>
        </div>

        <style>
        .chart-js-card {{
            font-family: -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, Helvetica,
                 Arial, sans-serif;
            background: rgba(238,242,255,1);
            border-radius: 24px;
            padding: 24px 28px 15px 28px;
            width: 95%;
        }}

        .chart-insight {{
            margin-top: 14px;
            padding-top: 12px;
            border-top: 1px dashed rgba(15,23,42,0.15);
            font-size: 14px;
            color: #334155;
            line-height: 1.55;
        }}

        .chart-insight b {{
            color: #0f172a;
            font-weight: 600;
        }}
        </style>

        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>

        <script>
        vegaEmbed("#{chart_id}", {json.dumps(spec)}, {{
            actions: false,
            renderer: "svg"
        }});
        </script>
        """,
        height=height,
        scrolling=False
    )


def load_css(folder: str):
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
        direction_text = "Predicted rainfall is higher than observed."
    elif diff < 0:
        direction = "underprediction"
        direction_text = "Predicted rainfall is lower than observed."
    else:
        direction = "exact"
        direction_text = "Prediction closely matches observed rainfall."

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
                "Relative deviation appears large, but absolute rainfall remains low."
            )

    else:
        # moderate & heavy rain
        if rel_error_pct <= 20:
            severity = "low"
            severity_text = "Prediction is highly reliable for this rainfall level."
        elif rel_error_pct <= 50:
            severity = "moderate"
            severity_text = (
                "Prediction shows noticeable deviation that may affect planning decisions."
            )
        else:
            severity = "high"
            severity_text = (
                "Prediction deviates substantially from observed rainfall."
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
        return "Minimal rainfall is expected. No significant impact is anticipated under typical conditions."
    if intensity_level == "light":
        return "Light rainfall is expected. Conditions may feel damp, but impacts are generally limited."
    if intensity_level == "moderate":
        return "Moderate rainfall is expected. Consider short-term planning impacts, especially around commute hours."
    return "Heavy rainfall is expected. Be alert for localized flooding risk and changing conditions."


def scenario_insight_text(rain_mm: float, intensity_level: str) -> str:
    if intensity_level == "drizzle":
        return "Under the specified inputs, the model indicates minimal rainfall. This reflects a low-rain response scenario."
    if intensity_level == "light":
        return "Under the specified inputs, the model predicts light rainfall. This scenario suggests mild rain conditions."
    if intensity_level == "moderate":
        return "Under the specified inputs, the model predicts moderate rainfall. This scenario can affect short-term activity planning."
    return "Under the specified inputs, the model predicts heavy rainfall. This scenario indicates elevated rain intensity."


def render_evaluation_page():
    st.header("Evaluation results")
    st.caption("Observed vs predicted rainfall")

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