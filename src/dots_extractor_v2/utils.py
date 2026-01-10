import cv2

import pandas as pd
import numpy as np


def is_year_start(ts):
    return ts.month == 1 and ts.day == 1


def extract_blue_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.array([100, 50, 30]),
        np.array([130, 255, 255])
    )


def snap_values(
    items,
    ground_truth,
    get_value,
    set_value,
    tol_px=2,
):
    unused_gt = set(ground_truth)
    snapped = []

    for item in items:
        val = get_value(item)

        if not unused_gt:
            snapped.append(item)
            continue

        nearest = min(unused_gt, key=lambda g: abs(g - val))
        dist = abs(nearest - val)

        if dist <= tol_px:
            snapped.append(set_value(item, nearest))
            unused_gt.remove(nearest)
        else:
            snapped.append(item)

    return snapped


def extend_xticks(
    xticks,
    boundaries,
):
    assert len(xticks) >= 2, "Need at least 2 xticks to infer spacing"

    data_start = boundaries["data_start"]
    data_end = boundaries["data_end"]

    xticks = sorted(xticks, key=lambda x: x[0])

    xs = [x for x, _, _ in xticks]
    ts = [t for _, t, _ in xticks]

    dxs = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    dts = [ts[i] - ts[i - 1] for i in range(1, len(ts))]

    dx = float(np.median(dxs))
    dt = pd.to_timedelta(
        np.median([dt.value for dt in dts]),
        unit="ns",
    )

    new_ticks = []

    x0, ts0, _ = xticks[0]
    cur_x = x0
    cur_ts = ts0

    while cur_x - dx >= data_start:
        cur_x -= dx
        cur_ts -= dt
        new_ticks.append(
            (cur_x, cur_ts, cur_ts.strftime("%Y-%m"))
        )

    new_ticks = new_ticks[::-1]

    xN, tsN, _ = xticks[-1]
    cur_x = xN
    cur_ts = tsN

    right_ticks = []

    while cur_x + dx <= data_end:
        cur_x += dx
        cur_ts += dt
        right_ticks.append(
            (cur_x, cur_ts, cur_ts.strftime("%Y-%m"))
        )

    return new_ticks + xticks + right_ticks


def round_day_with_cutoff(ts, cutoff_hour=23, cutoff_minute=15):
    ts = pd.Timestamp(ts)
    shift = pd.Timedelta(hours=24 - cutoff_hour, minutes=-cutoff_minute)
    return (ts + shift).floor("D")


def estimate_plot_time_window_from_xticks(
    xticks,
    data_start_px,
    data_end_px,
    cutoff_hour=16,
    cutoff_minute=00,
):
    assert len(xticks) >= 2

    xticks = sorted(xticks, key=lambda t: t[0])

    xs = np.array([x for x, _, _ in xticks], float)
    ts_ns = np.array([ts.value for _, ts, _ in xticks], np.int64)

    a, b = np.polyfit(xs, ts_ns, 1)

    t0_raw = pd.Timestamp(int(a * data_start_px + b))
    t1_raw = pd.Timestamp(int(a * data_end_px   + b))

    shift = pd.Timedelta(hours=24 - cutoff_hour, minutes=-cutoff_minute)

    t0 = (t0_raw + shift).floor("D")
    t1 = (t1_raw + shift).floor("D")

    if is_year_start(t1):
        t1 = t1 - pd.Timedelta(days=1)

    return min(t0, t1), max(t0, t1)


def expected_n_in_plot(csv_dates, t0, t1):
    csv_dates = pd.to_datetime(csv_dates).sort_values()

    t0 = pd.Timestamp(t0).normalize()
    t1 = pd.Timestamp(t1).normalize()

    window = t1 - t0
    n_expected = window.days + 1
    return n_expected