# Ignore warning
from warnings import filterwarnings
filterwarnings('ignore')

# Progress library
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

# Core library
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Pathing library
import glob
import os
from pathlib import Path

# Helper: clean column names from dataset preparation
from src.dataset_builder import clean_column_names

# Computer vision library
import cv2
import pytesseract

# Adjust here
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def log(msg, verbose=True, level="info"):
    if not verbose:
        return

    if tqdm._instances:
        tqdm.write(msg)
    else:
        print(msg)


def count_total_rows(input_root: str) -> int:
    total = 0
    for loc in os.scandir(input_root):
        if not loc.is_dir():
            continue
        for csv_file in glob.glob(f"{loc.path}/*.csv"):
            total += len(pd.read_csv(csv_file, usecols=[0]))

    return total


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

def find_data_boundaries(image: np.ndarray) -> dict:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(
        hsv,
        np.array([100, 50, 30]),
        np.array([130, 255, 255])
    )

    contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No blue plot detected")

    x_coords = np.concatenate([cnt[:, 0, 0] for cnt in contours])

    return {
        "data_start": int(x_coords.min()),
        "data_end": int(x_coords.max())
    }


def inspect_xtick_ocr(
    image: np.ndarray,
    roi_x0: int = 0,
    roi_x1: int = 1500,
    roi_y0: int = 0,
    roi_y1: int = 700,
    show: bool = True
):
    import cv2, pytesseract, matplotlib.pyplot as plt

    roi = image[roi_y0:roi_y1, roi_x0:roi_x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    data = pytesseract.image_to_data(
        bin_img,
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    det = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        if not txt:
            continue
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        det.append((txt, x, y, w, h))

    overlay = roi.copy()
    for txt, x, y, w, h in det:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv2.putText(
            overlay, txt, (x, max(10, y-3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1
        )

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axs[1].set_title("ROI")
        axs[1].axis("off")

        axs[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axs[2].set_title("OCR result")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

    return det

def extract_xticks_from_ocr(ocr_results, roi_x0=0):
    XTICK_PATTERN = re.compile(r"(\d{4})-(\d{2})")
    ticks = []

    for txt, x, y, w, h in ocr_results:
        m = XTICK_PATTERN.search(txt)
        if not m:
            continue

        year, month = m.groups()

        try:
            dt = pd.to_datetime(f"{year}-{month}")
        except Exception:
            continue

        x_center = roi_x0 + x + w / 2
        ticks.append((x_center, dt, txt))

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


def estimate_missing_left_timestamp(
    ticks,
    plot_border_x,
    tolerance_ratio=0.4
):
    a, b = compute_pixel_time_slope(ticks)
    pixel_spacing = estimate_pixel_spacing(ticks)

    first_x, first_t, _ = ticks[0]

    candidate_x = first_x - pixel_spacing
    if candidate_x >= plot_border_x + tolerance_ratio * pixel_spacing:
        candidate_t = pd.to_datetime(int(a * candidate_x + b))
        return {
            "exists": True,
            "x": candidate_x,
            "timestamp": candidate_t,
            "method": "estimated_missing_tick"
        }

    border_t = pd.to_datetime(int(a * plot_border_x + b))
    return {
        "exists": False,
        "x": plot_border_x,
        "timestamp": border_t,
        "method": "border_fallback"
    }


def extract_y_axis_labels(
    image: np.ndarray,
    x_start: int = 30,
    x_frac: float = 0.05,
    y_top_frac: float = 0.10,
    y_bot_frac: float = 0.95,
    verbose: bool = True
) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    x0 = x_start
    x1 = int(w * x_frac)
    y0 = int(h * y_top_frac)
    y1 = int(h * y_bot_frac)

    roi = gray[y0:y1, x0:x1]
    roi_bin = cv2.threshold(
        roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    data = pytesseract.image_to_data(
        roi_bin,
        config="--oem 3 --psm 6 outputbase digits",
        output_type=pytesseract.Output.DICT
    )

    labels = []

    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt.isdigit():
            continue

        val = int(txt)
        if val >= 200:
            continue

        x, y, ww, hh = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        if ww < 6 or hh < 8:
            continue

        cx = x0 + x + ww / 2
        cy = y0 + y + hh / 2

        labels.append({"value": val, "x": cx, "y": cy})

    if len(labels) < 2:
        if verbose:
            print("[DEBUG] ROI coords:", (x0, y0, x1, y1))
            print("[DEBUG] Raw OCR:", [t for t in data["text"] if t])
        raise ValueError("Insufficient Y-axis labels detected")


    labels_by_y = sorted(labels, key=lambda d: d["y"], reverse=True)

    uniq = []
    seen = set()
    for item in labels_by_y:
        if item["value"] in seen:
            continue
        uniq.append(item)
        seen.add(item["value"])
        if len(uniq) >= 2:
            break

    if len(uniq) < 2:
        raise ValueError("Need at least 2 distinct labels")

    low, high = uniq[0], uniq[1]

    dy = high["y"] - low["y"]
    dv = high["value"] - low["value"]
    if dv == 0:
        raise ValueError("Can't compute scale (dv=0)")

    pixel_per_unit = abs(dy) / abs(dv)

    zero_y = low["y"] + low["value"] * pixel_per_unit
    labels.append({
        "value": 0,
        "x": low["x"],
        "y": float(zero_y)
    })

    data_boundaries = find_data_boundaries(image)
    plot_x_start = data_boundaries['data_start']
    plot_x_end = data_boundaries['data_end']
    top_border_y = detect_plot_top_border(
        gray, plot_x_start, plot_x_end
    )

    if top_border_y is not None:
        max_label = max(labels, key=lambda l: l["value"])
        delta_pixel = max_label["y"] - top_border_y
        max_value = max_label["value"] + delta_pixel / pixel_per_unit

        labels.append({
            "value": float(max_value),
            "x": max_label["x"],
            "y": float(top_border_y)
        })

        if verbose:
            print(f"[INFO] Extrapolated max ≈ {max_value:.2f}")

    return {
        l["value"]: (l["x"], l["y"])
        for l in sorted(labels, key=lambda l: l["value"])
    }

def load_and_resize(image_path: str, scale: float = 1.0) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    if scale != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    return image


def extract_blue_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([100, 50, 30]),
        np.array([130, 255, 255])
    )


def build_y_pixel_to_value(labels: dict):
    pairs = sorted(
        [(coord[1], val) for val, coord in labels.items()],
        key=lambda t: t[0]  # sort by y pixel
    )
    y_pixels = np.array([p[0] for p in pairs], dtype=float)
    values   = np.array([p[1] for p in pairs], dtype=float)

    def y_to_value(y_pixel: float):
        return float(np.interp(y_pixel, y_pixels, values))

    return y_to_value


def extract_dot_pixels(
    blue_mask: np.ndarray,
    scale: float = 1.0,
    vertical_kernel_height: int = 6,
    min_area: int = 2,
    debug: bool = False,
    original_image: np.ndarray | None = None
) -> list[tuple[int, int]]:
    vert_h = max(3, int(vertical_kernel_height * scale * scale))
    min_area = max(1, int(min_area * scale * scale))

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vert_h)
    )

    mask_vert = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    dot_mask = cv2.subtract(blue_mask, mask_vert)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        dot_mask, connectivity=8
    )

    dots = []

    for i in range(1, num):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue

        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue

        cx = int(xs.mean())
        cy = int(ys.mean())

        dots.append((cx, cy))

    if debug:
        if original_image is None:
            raise ValueError("original_image must be provided when debug=True")

        vis = original_image.copy()

        # draw detected dot centroids
        for (x, y) in dots:
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")

        axs[1].imshow(blue_mask, cmap="gray")
        axs[1].set_title("Blue Mask")

        axs[2].imshow(dot_mask, cmap="gray")
        axs[2].set_title("Dot Mask (Residual)")

        axs[3].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axs[3].set_title(f"Detected Dots ({len(dots)})")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    return dots


def dots_to_daily_rainfall(
    dots,
    y_to_value,
    boundaries,
    total_days,
    zero_tol: float = 0.5,
):

    rainfall = [0] * total_days
    flags = {
        "multi_dot_nonzero": 0,
        "mixed_zero_nonzero": 0
    }

    x_start = boundaries["data_start"]
    plot_width = boundaries["data_end"] - boundaries["data_start"]

    pixel_per_day = plot_width / total_days
    xs = sorted(x for x, _ in dots)

    gap_factor = 20
    min_gap_px = gap_factor * pixel_per_day

    for x1, x2 in zip(xs[:-1], xs[1:]):
        gap = x2 - x1

        if gap > min_gap_px:
            day_start = int(round((x1 - x_start) / plot_width * (total_days - 1)))
            day_end   = int(round((x2 - x_start) / plot_width * (total_days - 1)))

            for d in range(day_start + 1, day_end):
                if 0 <= d < total_days:
                    rainfall[d] = np.nan

    bucket = [[] for _ in range(total_days)]
    for x, y in dots:
        day = int(round((x - x_start) / plot_width * (total_days - 1)))
        if 0 <= day < total_days:
            bucket[day].append((x, y))

    for day, pts in enumerate(bucket):
        if not pts:
            continue

        values = [float(y_to_value(y)) for _, y in pts]

        nonzero = [v for v in values if v > zero_tol]
        zeros   = [v for v in values if v <= zero_tol]

        if len(nonzero) >= 2:
            flags["multi_dot_nonzero"] += 1

        elif len(nonzero) == 1 and len(zeros) >= 1:
            flags["mixed_zero_nonzero"] += 1

        if nonzero:
            rainfall[day] = max(nonzero)
        else:
            rainfall[day] = 0.0

    return rainfall, flags


def extract_rainfall_from_plot(
    image_path: str,
    total_days: int,
    scale: float = 1.0,
    verbose: bool = False,
    debug: bool = False
) -> list[float]:
    try:
        image      = load_and_resize(image_path, scale)
        gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        side = detect_plot_side_border(image)

        ocr = inspect_xtick_ocr(image, show=False)
        xticks = extract_xticks_from_ocr(ocr)

        result = estimate_missing_left_timestamp(
            xticks,
            plot_border_x=0
        )
        xticks_pixel = [px for px, _, _ in xticks]

        boundaries   = find_data_boundaries(image)

        if boundaries["data_start"] < min(xticks_pixel) - 5:
            boundaries["data_start"] = side['left']

        boundaries["data_start"] = boundaries['data_start']
        boundaries["data_end"] = boundaries['data_end']

        labels = extract_y_axis_labels(
            image=image, verbose=verbose
        )

        y_to_value  = build_y_pixel_to_value(labels)
        blue_mask   = extract_blue_mask(image)

        dots = extract_dot_pixels(
            blue_mask,
            scale=scale,
            vertical_kernel_height=6,
            min_area=2,
            debug=debug,
            original_image=image
        )

        rainfall, _ = dots_to_daily_rainfall(
            dots,
            y_to_value,
            boundaries,
            total_days,
        )

        return rainfall

    except Exception as e:
        print("   ERROR in extract_rainfall_from_plot")
        print("   image_path :", image_path)
        print("   boundaries :", boundaries)
        print("   labels     :", labels)
        raise


def process_all_locations(
    input_root: str,
    output_dir: str,
    verbose=True
):
    os.makedirs(output_dir, exist_ok=True)

    total_rows = count_total_rows(input_root)

    with tqdm(
        total=total_rows,
        desc="Extracting rainfall",
        unit="rows"
    ) as pbar:

        for loc in os.scandir(input_root):
            if not loc.is_dir():
                continue

            csvs = sorted(glob.glob(f"{loc.path}/*.csv"))
            pngs = sorted(glob.glob(f"{loc.path}/*.png"))

            yearly_data = []

            for csv_file, png_file in zip(csvs, pngs):
                df = clean_column_names(pd.read_csv(csv_file))
                df["date"] = pd.to_datetime(df["date"])

                rainfall = extract_rainfall_from_plot(
                    png_file, len(df), verbose
                )

                df["daily_rainfall_total_mm"] = rainfall
                yearly_data.append(df[["date", "daily_rainfall_total_mm"]])

                pbar.update(len(df))

            if yearly_data:
                final_df = pd.concat(yearly_data, ignore_index=True)
                final_df.to_csv(
                    os.path.join(output_dir, f"{loc.name}.csv"),
                    index=False
                )