import cv2

import pandas as pd
import numpy as np

from src.dots_extractor_v2.utils import is_year_start

def detect_dot_mask(
    image_bgr,
    target_rgb=(31, 119, 180),
    tolerance=5,
    roi=None,
):
    H, W, _ = image_bgr.shape
    target_bgr = np.array(target_rgb[::-1], dtype=np.int16)

    if roi is None:
        x0, x1 = 0, W
        y0, y1 = 0, H
    else:
        x0, x1, y0, y1 = roi
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(W, int(x1))
        y1 = min(H, int(y1))

    roi_img = image_bgr[y0:y1, x0:x1].astype(np.int16)
    diff = np.abs(roi_img - target_bgr)

    color_mask = np.all(diff <= tolerance, axis=2).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)

    dot_roi = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_HITMISS,
        kernel
    )

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = dot_roi * 255

    return mask, (x0, x1)


def extract_dots(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dots = []

    for cid in range(1, num):
        x, y, w, h, area = stats[cid]

        ys, xs = np.where(labels == cid)

        if w >= 2*h:
            x_mid = np.median(xs)

            left = xs <= x_mid
            right = xs > x_mid

            if left.any() and right.any():
                lx = int(np.mean(xs[left]))
                ly = int(np.mean(ys[left]))

                rx = int(np.mean(xs[right]))+1
                ry = int(np.mean(ys[right]))
                dots.append((lx, ly))
                dots.append((rx, ry))
            else:
                dots.append((int(xs.mean()), int(ys.mean())))
            continue

        dots.append((int(xs.mean()), int(ys.mean())))

    return dots


def find_dots_near_xticks(
    xticks,
    dots,
    candidates,
    max_dx=1.0,
):
    results = []

    dots = list(dots)

    def find_best_dot(x_target):
        best = None
        best_dx = None
        for x, y in dots:
            dx = abs(x - x_target)
            if dx <= max_dx:
                if best_dx is None or dx < best_dx:
                    best_dx = dx
                    best = (x, y)
        return best

    for x_tick, date, label in xticks:
        dot = find_best_dot(x_tick)
        used_date = date

        if dot is None:
            for dx_step in candidates:
                # coba kiri
                d = find_best_dot(x_tick - dx_step)
                if d is not None:
                    dot = d
                    used_date = date - pd.Timedelta(days=1)
                    break

                # coba kanan
                d = find_best_dot(x_tick + dx_step)
                if d is not None:
                    dot = d
                    used_date = date + pd.Timedelta(days=1)
                    break

        results.append({
            "x_tick": x_tick,
            "date": used_date,
            "label": label,
            "dot": dot,
        })

    return results


def pick_best_anchor_from_matches(matches):
    best = None

    for m in matches:
        dot = m["dot"]
        if dot is None:
            continue

        x_dot, _ = dot
        x_tick = m["x_tick"]
        dx = abs(x_dot - x_tick)

        if dx == 0:
            return dot, m["date"]

        if best is None or dx < best[0]:
            best = (dx, dot, m["date"])

    if best is None:
        return None, None

    _, dot, date = best

    return dot, date


def detect_missing_ranges(
    dots,
    boundaries,
    xticks,
    candidates,
):
    xs = sorted(x for x, _ in dots)

    x0 = boundaries["data_start"]
    x1 = boundaries["data_end"]

    min_dx = min(candidates)
    max_dx = max(candidates)

    missing_ranges = []

    first_dot_x = xs[0]

    if first_dot_x > x0:
        missing_ranges.append(
            (x0 - max_dx, first_dot_x)
        )

    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]

        if dx > max_dx + 1:
            missing_ranges.append(
                (xs[i], xs[i + 1])
            )

    last_dot_x = xs[-1]
    last_xtick_x, last_xtick_ts, _ = xticks[-1]

    if x1 > last_dot_x:
        if not is_year_start(last_xtick_ts):
            missing_ranges.append(
                (last_dot_x, x1 + max_dx)
            )
        else:
            if x1 - last_dot_x >= max_dx + 1:
                missing_ranges.append(
                    (last_dot_x, x1)
                )
            else:
                pass

    return missing_ranges