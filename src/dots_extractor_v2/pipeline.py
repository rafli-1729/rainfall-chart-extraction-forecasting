import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from src.dots_extractor_v2.utils import (
    extract_blue_mask,
    snap_values,
    extend_xticks,
    expected_n_in_plot,
    estimate_plot_time_window_from_xticks
)

from src.dots_extractor_v2.plot_metadata import (
    detect_grid,
    perform_ocr,
    extract_yticks_from_ocr,
    extract_xticks_from_ocr,
    most_common_x_distances,
    find_data_boundaries,
)

from src.dots_extractor_v2.dot_detection import (
    detect_dot_mask,
    extract_dots,
    find_dots_near_xticks,
    pick_best_anchor_from_matches,
    detect_missing_ranges,
)

from src.dots_extractor_v2.missing_solver import (
    assign_missing_with_constraint_ratio,
    insert_missing_dots_from_result,
)

from src.dots_extractor_v2.mapper import (
    map_by_single_anchor,
    build_y_pixel_to_value,
    y_px_to_rainfall,
)


import traceback
from pathlib import Path

def extract_rainfall_from_plot(
    png: str | Path,
    csv: str | Path,
    dot_color: tuple[int] = (31, 119, 180),

    color_tolerance: int = 4,
    label_px_tolerance: int = 2,
    label_ocr_pad: int = 5,
    ticks_px_tolerance: int = 2,
    vertical_right: int = 100,

    show: bool = False,
    verbose: bool = False,
):
    png = Path(png)
    csv = Path(csv)

    image = None
    debug_dir = png.parent / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- load image & mask ---
        image = cv2.imread(str(png))
        if image is None:
            raise RuntimeError(f"Failed to read image: {png}")

        blue = extract_blue_mask(image)

        # --- load & normalize csv ---
        df = pd.read_csv(csv)
        df["Date"] = pd.to_datetime(df["Date"])

        df = (
            df
            .set_index("Date")
            .reindex(
                pd.date_range(
                    start=df["Date"].min(),
                    end=df["Date"].max(),
                    freq="D",
                )
            )
            .rename_axis("Date")
            .reset_index()
        )

        n_days = len(df)

        # --- detect grids ---
        vertical_grid, horizontal_grid = detect_grid(
            image,
            right_px=vertical_right,
            show=show,
        )

        # --- detect dots ---
        mask, _ = detect_dot_mask(
            image_bgr=image,
            target_rgb=dot_color,
            tolerance=color_tolerance,
        )

        dots = extract_dots(mask)

        # --- OCR y labels ---
        label_ocr = perform_ocr(
            image,
            punctuation=False,
            roi_x0=27,
            roi_x1=vertical_grid[0] + label_ocr_pad,
            show=show,
        )

        labels = extract_yticks_from_ocr(label_ocr)

        snapped_items = snap_values(
            labels.items(),
            horizontal_grid,
            get_value=lambda item: item[1][1],
            set_value=lambda item, y: (item[0], (item[1][0], y)),
            tol_px=label_px_tolerance,
        )

        snapped_labels = dict(snapped_items)

        # --- OCR x ticks ---
        date_ocr = perform_ocr(
            image,
            punctuation=True,
            roi_y0=horizontal_grid[-1],
            show=show,
        )

        xticks = extract_xticks_from_ocr(date_ocr)

        snapped_xticks = snap_values(
            xticks,
            vertical_grid,
            get_value=lambda item: item[0],
            set_value=lambda item, x: (x, item[1], item[2]),
            tol_px=ticks_px_tolerance,
        )

        # --- estimate plot boundaries ---
        top_dx = most_common_x_distances(
            dots,
            verbose=verbose,
        )

        candidates = sorted(delta for delta, _ in top_dx)

        boundaries = find_data_boundaries(
            blue_mask=blue,
            vertical_grid=vertical_grid,
            xticks=snapped_xticks,
            delta=max(candidates),
        )

        x0 = boundaries["data_start"]
        x1 = boundaries["data_end"]

        # --- extend xticks & estimate time window ---
        xticks_full = extend_xticks(snapped_xticks, boundaries)

        t0, t1 = estimate_plot_time_window_from_xticks(
            xticks=xticks_full,
            data_start_px=x0,
            data_end_px=x1,
        )

        expected_plot_n = expected_n_in_plot(
            df["Date"],
            t0,
            t1,
        )

        extracted_dot_n = len(dots)
        n_missing = expected_plot_n - extracted_dot_n

        # --- anchor matching ---
        max_dx = max(1, int(np.floor(365 / n_days)))

        matches = find_dots_near_xticks(
            xticks=snapped_xticks,
            dots=dots,
            max_dx=max_dx,
            candidates=candidates
        )

        anchor_dot, anchor_date = pick_best_anchor_from_matches(matches)

        # --- missing detection & solving ---
        if n_missing != 0:
            missing_ranges = detect_missing_ranges(
                dots=dots,
                boundaries=boundaries,
                xticks=xticks_full,
                candidates=candidates,
            )

            missing = assign_missing_with_constraint_ratio(
                pairs=missing_ranges,
                n_missing_total=n_missing,
                candidates=candidates,
                dx_hist=dict(top_dx),
                verbose=verbose,
            )

            all_dots = insert_missing_dots_from_result(
                dots=dots,
                missing_ranges=missing_ranges,
                result=missing,
                candidates=candidates,
                boundaries=boundaries,
            )
        else:
            all_dots = sorted(dots, key=lambda x: x[0])

        # --- mapping ---
        mapped_x = map_by_single_anchor(
            dots=all_dots,
            anchor_dot=anchor_dot,
            anchor_date=anchor_date,
            df=df,
        )

        y_to_value = build_y_pixel_to_value(snapped_labels)
        rainfall = y_px_to_rainfall(
            series_px=mapped_x,
            y_to_value=y_to_value,
        )

        if verbose:
            print(f"[OK] {png.name} → {len(rainfall)} days extracted")

        return rainfall

    except Exception as e:
        stem = png.stem

        debug_img_path = debug_dir / f"{stem}_FAILED.png"
        debug_log_path = debug_dir / f"{stem}_FAILED.txt"

        if image is not None:
            cv2.imwrite(str(debug_img_path), image)

        with open(debug_log_path, "w") as f:
            f.write(f"image_path: {png}\n")
            f.write(f"csv_path  : {csv}\n\n")
            f.write(str(e) + "\n\n")
            f.write(traceback.format_exc())

        if verbose or show:
            print(f"[FAIL] {png.name}")
            print(f"PNG : {debug_img_path}")
            print(f"LOG : {debug_log_path}")
            print(traceback.format_exc())

        return pd.Series(dtype=float)