import pandas as pd
import numpy as np

def map_by_single_anchor(
    dots,
    anchor_dot,
    anchor_date,
    df,
):
    df["Date"] = pd.to_datetime(df["Date"])

    series = pd.Series(
        np.nan,
        index=df["Date"],
        name="y_px"
    )

    dot_idx = dots.index(anchor_dot)
    date_idx = df.index[df["Date"] == anchor_date][0]

    series.iloc[date_idx] = anchor_dot[1]

    di, dj = dot_idx + 1, date_idx + 1
    while di < len(dots) and dj < len(series):
        series.iloc[dj] = dots[di][1]
        di += 1
        dj += 1

    di, dj = dot_idx - 1, date_idx - 1
    while di >= 0 and dj >= 0:
        series.iloc[dj] = dots[di][1]
        di -= 1
        dj -= 1

    return series


def y_px_to_rainfall(series_px, y_to_value):
    series_mm = series_px.copy()

    for idx, y in series_mm.items():
        if pd.isna(y):
            series_mm.loc[idx] = np.nan
        else:
            series_mm.loc[idx] = float(y_to_value(y))

    series_mm.name = "daily_rainfall_total_mm"
    return series_mm


def build_y_pixel_to_value(labels: dict):
    pairs = sorted(
        [(coord[1], val) for val, coord in labels.items()],
        key=lambda t: t[0]
    )
    y_pixels = np.array([p[0] for p in pairs], dtype=float)
    values   = np.array([p[1] for p in pairs], dtype=float)

    def y_to_value(y_pixel: float):
        return float(np.interp(y_pixel, y_pixels, values))

    return y_to_value