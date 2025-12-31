import re
from pathlib import Path
import streamlit as st
from streamlit.components.v1 import html

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


def load_styles(folder: str):
    css = ""

    for css_file in sorted(Path(folder).glob("*.css")):
        css += css_file.read_text() + "\n"

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)