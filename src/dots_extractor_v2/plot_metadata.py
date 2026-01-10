import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pytesseract

import re
from collections import Counter

from src.dots_extractor_v2.utils import is_year_start

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def is_year_end(ts):
    return ts.month == 12 and ts.day == 31


def find_data_boundaries(
    blue_mask: np.ndarray,
    vertical_grid: list[int],
    xticks: list[tuple],
    delta: int,
) -> dict:
    if len(vertical_grid) < 4:
        raise ValueError("vertical_grid must contain at least 4 entries (border, timestamps, border)")

    contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No blue plot detected")

    x_coords = np.concatenate([cnt[:, 0, 0] for cnt in contours])
    blue_min_x = int(x_coords.min())
    blue_max_x = int(x_coords.max())

    border_left = int(vertical_grid[0])
    first_ts = int(vertical_grid[1])
    last_ts = int(vertical_grid[-2])
    border_right = int(vertical_grid[-1])

    if blue_min_x >= first_ts - delta:
        data_start = first_ts
    else:
        data_start = border_left

    first_xtick_x, _, _ = xticks[0]
    last_xtick_x, last_xtick_ts, _ = xticks[-1]

    if is_year_end(last_xtick_ts):
        data_end = last_ts

    elif is_year_start(last_xtick_ts):
        if blue_max_x <= last_ts + delta:
            data_end = last_ts
        else:
            data_end = border_right
    else:
        if blue_max_x <= last_ts:
            data_end = last_ts
        else:
            data_end = border_right

    if border_left == data_start:
        data_start += 2

    return {
        "data_start": int(data_start),
        "data_end": int(data_end)
    }


def perform_ocr(
    image: np.ndarray,
    punctuation: bool = True,
    roi_x0: int = 0,
    roi_x1: int = 1500,
    roi_y0: int = 0,
    roi_y1: int = 700,
    show: bool = True
):
    roi = image[roi_y0:roi_y1, roi_x0:roi_x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    data = pytesseract.image_to_data(
        bin_img,
        lang="eng",
        config=(
            f"--psm 6 "
            f"-c tessedit_char_whitelist=0123456789{'-' if punctuation else ''}"
        ),
        output_type=pytesseract.Output.DICT
    )

    det = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt:
            continue

        x_roi = data["left"][i]
        y_roi = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        x = x_roi + roi_x0
        y = y_roi + roi_y0

        det.append((txt, x, y, w, h))

    if show:
        overlay = image.copy()

        for txt, x, y, w, h in det:
            cv2.rectangle(
                overlay,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                1
            )
            cv2.putText(
                overlay,
                txt,
                (x, max(10, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axs[1].set_title("ROI")
        axs[1].axis("off")

        axs[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[2].set_title("OCR")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

    return det


def extract_xticks_from_ocr(ocr_results, roi_x0=0):
    XTICK_PATTERN = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?")
    ticks = []

    for txt, x, y, w, h in ocr_results:
        m = XTICK_PATTERN.search(txt)
        if not m:
            continue

        year, month, day = m.groups()

        try:
            if day is None:
                dt = pd.to_datetime(f"{year}-{month}")
            else:
                dt = pd.to_datetime(f"{year}-{month}-{day}")
        except Exception:
            continue

        x_center = roi_x0 + x + w/2

        ticks.append((x_center, dt, txt.replace(",", "")))

    if len(ticks) < 2:
        raise RuntimeError("Not enough valid xticks detected")

    return ticks


def compute_pixel_time_slope(ticks):
    xs = np.array([x for x, _, _ in ticks])
    ts = np.array([t.value for _, t, _ in ticks])

    a, b = np.polyfit(xs, ts, 1)
    return a, b


def estimate_pixel_spacing(ticks):
    xs = np.array([x for x, _, _ in ticks])
    dx = np.diff(xs)
    return np.median(dx)


def extract_yticks_from_ocr(
    ocr_results,
    *,
    top_border_y=82,
    bottom_border_y=618.0,
    roi_y0=0,
    min_ticks=2,
    x_tol=30,
):
    NUMERIC_PATTERN = re.compile(r"^\d+(\.\d+)?$")

    candidates = []
    for txt, x, y, w, h in ocr_results:
        if not NUMERIC_PATTERN.fullmatch(txt):
            continue

        try:
            value = float(txt)
        except Exception:
            continue

        if (value < 10 and value != 0) or value > 300:
            continue

        candidates.append({
            "value": value,
            "x": x + w/2,
            "y": roi_y0 + y + h/2,
        })

    if len(candidates) < min_ticks:
        raise RuntimeError("Not enough numeric OCR tokens for yticks")

    if top_border_y is not None:
        candidates = [
            c for c in candidates
            if c["y"] >= top_border_y
        ]

    if bottom_border_y is not None:
        candidates = [
            c for c in candidates
            if c["y"] <= bottom_border_y
        ]

    xs = np.array([c["x"] for c in candidates])
    x_med = np.median(xs)

    column = [
        c for c in candidates
        if abs(c["x"] - x_med) <= x_tol
    ]

    if len(column) < min_ticks:
        raise RuntimeError("Failed to isolate Y-axis column")

    column.sort(key=lambda c: c["value"])
    values = np.array([c["value"] for c in column])

    diffs = np.abs(np.diff(values))
    diffs = diffs[diffs > 1e-6]

    step = Counter(np.round(diffs, 3)).most_common(1)[0][0]

    value_set = set(np.round(values, 6))

    def chain_from(start):
        chain = []
        v = start
        while round(v, 6) in value_set:
            chain.append(round(v, 6))
            v += step
        return chain

    chains = [chain_from(v) for v in values]
    main_chain = max(chains, key=len)
    main_chain = set(main_chain)

    extended_chain = set(main_chain)

    for v in values:
        if v in main_chain:
            continue

        d = min(abs(v - c) for c in main_chain)
        if abs(d % step) < 1e-6:
            extended_chain.add(round(v, 6))

    filtered = [
        c for c in column
        if round(c["value"], 6) in extended_chain
    ]

    if len(filtered) < min_ticks:
        raise RuntimeError("Too many yticks rejected")

    filtered.sort(key=lambda c: c["value"])
    collapsed = {}

    for d in filtered:
        v = d["value"]
        if v not in collapsed or d["y"] > collapsed[v]["y"]:
            collapsed[v] = d

    filtered = list(collapsed.values())

    low, high = filtered[0], filtered[-1]

    dy = high["y"] - low["y"]
    dv = high["value"] - low["value"]
    if dv == 0:
        raise ValueError("Can't compute scale (dv=0)")

    pixel_per_unit = abs(dy) / abs(dv)

    if 0.0 not in [m["value"] for m in filtered]:
        zero_y = low["y"] + low["value"] * pixel_per_unit
        filtered.append({
            "value": 0.0,
            "x": low["x"],
            "y": float(zero_y)
        })

    max_label = max(filtered, key=lambda l: l["value"])
    delta_pixel = max_label["y"] - top_border_y
    max_value = max_label["value"] + delta_pixel / pixel_per_unit

    filtered.append({
        "value": float(max_value),
        "x": max_label["x"],
        "y": float(top_border_y)
    })

    return {
        l["value"]: (l["x"], l["y"])
        for l in sorted(filtered, key=lambda l: l["value"])
    }


def cluster_positions(pos, max_gap=0):
    clustered = []
    if not pos:
        return clustered

    cur = [pos[0]]
    for p in pos[1:]:
        if p - cur[-1] <= max_gap:
            cur.append(p)
        else:
            clustered.append(int(np.mean(cur)))
            cur = [p]
    clustered.append(int(np.mean(cur)))
    return clustered


def detect_horizontal_grid(
    image_bgr: np.ndarray,
    right_px: int = 70,
    edge_thresh1: int = 50,
    edge_thresh2: int = 150,
    min_edge_fraction: float = 0.6,
    show: bool = False,
):
    H, W = image_bgr.shape[:2]

    strip = image_bgr[:, W - right_px : W]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)

    ys = []
    for y in range(H):
        if np.count_nonzero(edges[y, :]) >= right_px * min_edge_fraction:
            ys.append(y)

    grid_ys = cluster_positions(ys, max_gap=2)

    if show:
        vis = image_bgr.copy()

        cv2.rectangle(
            vis,
            (W - right_px, 0),
            (W - 1, H - 1),
            (255, 0, 255),
            1,
        )

        for y in grid_ys:
            cv2.line(
                vis,
                (W - right_px, y),
                (W - 1, y),
                (128, 128, 255),
                1,
            )

        plt.figure(figsize=(15, 7))
        plt.title("Horizontal Grid Detection (Right Strip Only)")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return grid_ys


def detect_vertical_grid_band(
    image_bgr: np.ndarray,
    grid_ys: list,
    min_edge_fraction: float = 0.6,
    edge_thresh1: int = 50,
    edge_thresh2: int = 150,
    show: bool = False,
):
    y_low, y_high = grid_ys[-1], grid_ys[-2]
    H, W = image_bgr.shape[:2]

    band = image_bgr[y_high:y_low, :]

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)

    h, w = edges.shape

    xs = []
    for x in range(w):
        if np.count_nonzero(edges[:, x]) >= h * min_edge_fraction:
            xs.append(x)

    grid_xs = cluster_positions(xs, max_gap=2)

    if show:
        vis = image_bgr.copy()

        cv2.rectangle(
            vis,
            (0, y_high),
            (W - 1, y_low),
            (255, 0, 255),
            1,
        )

        for x in grid_xs:
            cv2.line(
                vis,
                (x, y_high),
                (x, y_low),
                (128, 128, 255),
                1,
            )

        plt.figure(figsize=(15, 7))
        plt.title("Vertical Grid Detection (Band Between 2 Horizontal Grids)")
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return grid_xs


def detect_grid(
    image,
    right_px: int = 70,
    show: bool = True
):
    grid_ys = detect_horizontal_grid(image, show=show, right_px=right_px)
    grid_xs = detect_vertical_grid_band(image, grid_ys=grid_ys, show=show)

    return grid_xs, grid_ys


def most_common_x_distances(
    dots,
    top_k: int = 3,
    verbose: bool = False,
):
    xs = sorted(x for x, _ in dots)

    dxs = []
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        if dx > 0:
            dxs.append(dx)

    if not dxs:
        raise ValueError("No valid dx found")

    counter = Counter(dxs)
    base_dx, base_freq = counter.most_common(1)[0]
    candidates = [(base_dx, base_freq)]

    for d in (base_dx - 1, base_dx + 1):
        if d > 0 and d in counter:
            candidates.append((d, counter[d]))

    candidates.sort(key=lambda x: -x[1])
    if verbose:
        print("[DEBUG] dx histogram:", counter.most_common())
        print("[DEBUG] selected candidates:", candidates)

    return candidates[:top_k]