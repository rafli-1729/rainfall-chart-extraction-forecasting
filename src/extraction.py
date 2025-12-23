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

# Pathing library
import glob
import os
from pathlib import Path

# Helper: clean column names from dataset preparation
from src.dataset import clean_column_names

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
    """
    Detects the Y pixel position of the top plot border using
    horizontal projection of inverted binary image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image.
    plot_x_start : int
        Left boundary of plot region.
    plot_x_end : int
        Right boundary of plot region.
    min_row : int, optional
        Rows above this index are ignored to avoid title/labels.
    occupancy_ratio : float, optional
        Minimum ratio of dark pixels required to detect border.
    binarize_thresh : int, optional
        Threshold value for binary inversion.

    Returns
    -------
    int | None
        Y pixel of detected top border, or None if not found.
    """

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
        "data_start": int(x_coords.min())+1,
        "data_end": int(x_coords.max())-1
    }

def extract_y_axis_labels(
    image: np.ndarray,
    plot_x_start: int,
    plot_x_end: int,
    verbose: bool = True
) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # === ROI: match with your best debug settings ===
    left_pad   = 75
    right_pad  = 30
    y_top_frac = 0.0
    y_bot_frac = 0.90

    y0 = int(h * y_top_frac)
    y1 = int(h * y_bot_frac)
    x1 = max(1, plot_x_start - right_pad)
    x0 = max(0, x1 - left_pad)

    roi = gray[y0:y1, x0:x1]

    # binarize (same as debug)
    roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    data = pytesseract.image_to_data(
        roi_bin,
        config="--oem 3 --psm 6 outputbase digits",
        output_type=pytesseract.Output.DICT
    )

    labels = []
    roi_w = roi.shape[1]
    x_cut = int(0.5 * roi_w)  # avoid axis line area (dynamic, not hardcoded 100)

    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt.isdigit():
            continue

        val = int(txt)
        if val >= 200:
            continue

        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        # Skip boxes too far right inside ROI (often axis/grid noise)
        if x + ww > x_cut:
            continue

        # Optional: skip tiny garbage boxes
        if ww < 6 or hh < 8:
            continue

        # convert ROI coords -> global coords
        cx = x0 + x + ww / 2
        cy = y0 + y + hh / 2

        labels.append({"value": val, "x": cx, "y": cy})

    if len(labels) < 2:
        if verbose:
            detected = [t.strip() for t in data["text"] if (t or "").strip()]
            print("[DEBUG] ROI coords:", (x0, y0, x1, y1))
            print("[DEBUG] Raw detected texts (first 30):", detected[:30])
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
        raise ValueError("Need at least 2 distinct labels to estimate zero")

    low, high = uniq[0], uniq[1]  # low: bottom tick (largest y)
    dy = high["y"] - low["y"]
    dv = high["value"] - low["value"]

    if dv == 0:
        raise ValueError("Can't compute scale (dv=0)")

    pixel_per_unit = abs(dy) / abs(dv)

    # Predict y for 0 (same concept as before, but based on bottom tick)
    zero_y = low["y"] + low["value"] * pixel_per_unit
    labels.append({"value": 0, "x": low["x"], "y": float(zero_y)})

    # Extrapolate max using top border (keep your logic)
    top_border_y = detect_plot_top_border(gray, plot_x_start, plot_x_end)
    if top_border_y is not None:
        max_label = max(labels, key=lambda l: l["value"])
        delta_pixel = max_label["y"] - top_border_y
        max_value = max_label["value"] + delta_pixel / pixel_per_unit

        labels.append({"value": float(max_value), "x": max_label["x"], "y": float(top_border_y)})

        if verbose:
            print(f"[INFO] Extrapolated max value ≈ {max_value:.2f}")

    return {l["value"]: (l["x"], l["y"]) for l in sorted(labels, key=lambda l: l["value"])}

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
    # labels: {value: (x, y)}
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
    """
    Extract dot centroids from a blue mask using residual-based morphology.

    Parameters
    ----------
    blue_mask : np.ndarray
        Binary mask of blue pixels.
    scale : float
        Image scale factor.
    vertical_kernel_height : int
        Height of vertical structuring element for line removal.
    min_area : int
        Minimum connected component area to be considered a dot.
    debug : bool
        If True, visualize intermediate masks and detected dots.
    original_image : np.ndarray | None
        Original image for overlay visualization (required if debug=True).

    Returns
    -------
    list[tuple[int, int]]
        List of (x, y) dot centroids.
    """

    vert_h = max(3, int(vertical_kernel_height * scale * scale))
    min_area = max(1, int(min_area * scale * scale))

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vert_h)
    )

    # Remove vertical line components
    mask_vert = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # Residual mask (dots)
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

    # cari gap antar dots
    for x1, x2 in zip(xs[:-1], xs[1:]):
        gap = x2 - x1

        if gap > min_gap_px:
            # ini gap mencurigakan → map ke day range
            day_start = int(round((x1 - x_start) / plot_width * (total_days - 1)))
            day_end   = int(round((x2 - x_start) / plot_width * (total_days - 1)))

            for d in range(day_start + 1, day_end):
                if 0 <= d < total_days:
                    rainfall[d] = np.nan

    # bucket dots by day
    bucket = [[] for _ in range(total_days)]
    for x, y in dots:
        day = int(round((x - x_start) / plot_width * (total_days - 1)))
        if 0 <= day < total_days:
            bucket[day].append((x, y))

    # resolve per day
    for day, pts in enumerate(bucket):
        if not pts:
            continue

        values = [float(y_to_value(y)) for _, y in pts]

        # classify values
        nonzero = [v for v in values if v > zero_tol]
        zeros   = [v for v in values if v <= zero_tol]

        # FLAG LOGIC (your rule)
        if len(nonzero) >= 2:
            flags["multi_dot_nonzero"] += 1

        elif len(nonzero) == 1 and len(zeros) >= 1:
            flags["mixed_zero_nonzero"] += 1

        # final value = max (domain-correct)
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

        boundaries = find_data_boundaries(image)
        plot_x_start = boundaries["data_start"]
        plot_x_end   = boundaries["data_end"]

        labels = extract_y_axis_labels(
            image=image,
            plot_x_start=plot_x_start,
            plot_x_end=plot_x_end,
            verbose=verbose
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
        print("❌ ERROR in extract_rainfall_from_plot")
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