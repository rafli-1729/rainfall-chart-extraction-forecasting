# Ignore warning
from warnings import filterwarnings
filterwarnings('ignore')

# Progress library
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

# Core library
from collections import Counter
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Pathing library
import glob
import os

# Computer vision library
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def log(msg, verbose=True, level="info"):
    if not verbose:
        return

    if tqdm._instances:
        tqdm.write(msg)
    else:
        print(msg)

# =============================== UTILS =============================== #

def count_total_rows(input_root: str) -> int:
    total = 0
    for loc in os.scandir(input_root):
        if not loc.is_dir():
            continue
        for csv_file in glob.glob(f"{loc.path}/*.csv"):
            total += len(pd.read_csv(csv_file, usecols=[0]))

    return total


def extract_blue_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([100, 50, 30]),
        np.array([130, 255, 255])
    )


def detect_plot_top_border(
    gray: np.ndarray,
    plot_x_start: int,
    plot_x_end: int,
    min_row: int = 10,
    occupancy_ratio: float = 0.20,
    binarize_thresh: int = 200
) -> int | None:
    plot_region = gray[:, plot_x_start:plot_x_end]

    _, binary_inv = cv2.threshold(
        plot_region,
        binarize_thresh,
        255,
        cv2.THRESH_BINARY_INV
    )

    row_projection = np.sum(binary_inv, axis=1)

    required_pixels = plot_region.shape[1] * occupancy_ratio * 255

    for y in range(min_row, len(row_projection)):
        if row_projection[y] > required_pixels:
            return y

    return None


def detect_plot_side_border(
    image: np.ndarray,
    min_height_ratio: float = 0.6,
    canny1: int = 50,
    canny2: int = 150,
    hough_thresh: int = 150,
    verbose: bool = False
) -> dict:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, canny1, canny2)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=int(h * min_height_ratio),
        maxLineGap=10
    )

    xs = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]

            if abs(x1 - x2) <= 2:
                height = abs(y2 - y1)
                if height >= h * min_height_ratio:
                    xs.append(x1)

    if not xs:
        if verbose:
            print("[WARN] No vertical plot borders detected")
        return {
            "left": None,
            "right": None,
            "all_detected": []
        }

    xs = sorted(xs)

    return {
        "left": int(xs[0]),
        "right": int(xs[-1]),
        "all_detected": xs
    }


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

    last_xtick_x, last_xtick_ts, _ = xticks[-1]

    if is_year_start(last_xtick_ts):
        if blue_max_x <= last_ts:
            data_end = last_ts
        else:
            data_end = border_right
    else:
        if blue_max_x <= border_right + delta:
            data_end = last_ts
        else:
            data_end = blue_max_x

    return {
        "data_start": int(data_start),
        "data_end": int(data_end)
    }


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