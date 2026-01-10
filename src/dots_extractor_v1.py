# Ignore warning
from warnings import filterwarnings
filterwarnings('ignore')

# Progress library
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

# Core library
import numpy as np
import matplotlib.pyplot as plt
from src.plot_label_extractor import (
    extract_blue_mask,
    find_data_boundaries,
    extract_y_axis_labels,
    build_y_pixel_to_value,
    # count_total_rows,
    # detect_plot_top_border,
    # detect_plot_side_border,
    # perform_ocr,
    # extract_xticks_from_ocr,
    # compute_pixel_time_slope,
    # estimate_pixel_spacing
)


# Computer vision library
import cv2
import pytesseract

import numpy as np

# Adjust here
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def log(msg, verbose=True, level="info"):
    if not verbose:
        return

    if tqdm._instances:
        tqdm.write(msg)
    else:
        print(msg)


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

    # detect horizontal
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    mask_horiz = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, horiz_kernel)

    # detect vertical
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    mask_vert = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, vert_kernel)

    # remove horizontal, keep vertical
    dot_mask = cv2.subtract(blue_mask, mask_vert)
    dot_mask = cv2.subtract(dot_mask, mask_horiz)

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
        image = cv2.imread(image_path)
        boundaries   = find_data_boundaries(image)

        labels = extract_y_axis_labels(
            image=image, verbose=verbose
        )

        y_to_value  = build_y_pixel_to_value(labels)

        blue_mask = extract_blue_mask(image)

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