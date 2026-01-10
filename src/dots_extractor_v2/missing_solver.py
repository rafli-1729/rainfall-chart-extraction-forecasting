from collections import Counter
import numpy as np
import math

INF = 1e18

def valid_extras_for_dx(dx, remaining, candidates):
    c1, c2 = candidates
    d = c2 - c1
    valid = set()

    for extra in range(0, remaining + 1):
        n_intervals = extra + 2

        if d == 0:
            if dx == n_intervals * c1:
                valid.add(extra)
            continue

        num = dx - n_intervals * c1
        if num < 0 or num % d != 0:
            continue

        b = num // d
        if 0 <= b <= n_intervals:
            valid.add(extra)

    return valid


def _solve_ab_for_dx(dx, n_intervals, candidates):
    c1, c2 = candidates
    d = c2 - c1
    if d == 0:
        if dx == n_intervals * c1:
            return (n_intervals, 0)
        return None

    num = dx - n_intervals * c1
    if num < 0 or num % d != 0:
        return None

    b = num // d
    if not (0 <= b <= n_intervals):
        return None

    a = n_intervals - b
    return (a, b)


def _ratio_cost(a, b, r_target):
    if b == 0:
        return abs((1e9) - r_target)

    r = a / b
    return abs(r - r_target)


def assign_missing_with_constraint_ratio(
    pairs,
    n_missing_total,
    candidates,
    dx_hist=None,
    target_ratio=None,
    verbose=False,
):
    G = len(pairs)
    if n_missing_total < G:
        raise RuntimeError("Total missing < number of gaps")
    remaining = n_missing_total - G

    c1, c2 = sorted(candidates)
    candidates = (c1, c2)

    if target_ratio is None:
        if dx_hist is None:
            r_target = 1.0
        else:
            count_c1 = dx_hist.get(c1, 0)
            count_c2 = dx_hist.get(c2, 0)
            if count_c2 == 0:
                r_target = 1e9
            else:
                r_target = count_c1 / count_c2
    else:
        r_target = float(target_ratio)

    if verbose:
        print(f"[RATIO] target a/b ~ {r_target:.6f} (a=min_dx intervals, b=max_dx intervals)")

    gap_opts = []
    for (x_left, x_right) in pairs:
        dx = abs(x_right - x_left)
        feas = valid_extras_for_dx(dx, remaining, candidates)
        if not feas:
            raise RuntimeError(f"No feasible extra for dx={dx}")

        opts = []
        for extra in sorted(feas):
            n_intervals = extra + 2
            ab = _solve_ab_for_dx(dx, n_intervals, candidates)
            if ab is None:
                continue
            a, b = ab
            cost = _ratio_cost(a, b, r_target)
            opts.append((extra, cost, a, b))

        if not opts:
            raise RuntimeError(f"Feasible extras exist but no (a,b) solutions for dx={dx}")

        gap_opts.append(opts)

    dp = [(INF, None, None, None, None)] * (remaining + 1)
    dp[0] = (0.0, None, None, None, None)

    parents = []
    for gi, opts in enumerate(gap_opts):
        new_dp = [(INF, None, None, None, None)] * (remaining + 1)
        for s in range(remaining + 1):
            base_cost, *_ = dp[s]
            if base_cost >= INF:
                continue
            for extra, cost, a, b in opts:
                ns = s + extra
                if ns > remaining:
                    continue
                cand_cost = base_cost + cost
                if cand_cost < new_dp[ns][0]:
                    new_dp[ns] = (cand_cost, s, extra, a, b)
        dp = new_dp
        parents.append(dp)

    final_cost, prev_s, extra, a, b = dp[remaining]
    if final_cost >= INF:
        raise RuntimeError("No DP solution meets global extra constraint")

    chosen = [None] * G
    s = remaining
    for gi in range(G - 1, -1, -1):
        cost, ps, ex, aa, bb = parents[gi][s]
        if ps is None and gi != 0:
            raise RuntimeError("DP reconstruction failed")
        chosen[gi] = (ex, aa, bb)
        s = ps if ps is not None else 0

    result = []
    final_missing = []
    for (x_left, x_right), (extra, a, b) in zip(pairs, chosen):
        n_missing = 1 + extra
        final_missing.append(n_missing)

        n_intervals = n_missing + 1
        if a + b != n_intervals:
            raise RuntimeError(f"(a+b) mismatch for gap {x_left}-{x_right}")

        result.append((n_missing, (a, b)))

    if verbose:
        print("[RESULT] Missing per gap:")
        for i, m in enumerate(final_missing):
            print(f"  gap {i:02d}: missing={m}, (a,b)={result[i][1]}")
        print(f"[CHECK] Σ missing = {sum(final_missing)} (expected {n_missing_total})")
        print(f"[COST] total ratio deviation cost = {final_cost:.6f}")

    return result


def insert_missing_dots_from_result(
    dots,
    missing_ranges,
    result,
    candidates,
    boundaries,
):
    dots_sorted = sorted(dots, key=lambda d: d[0])
    final_dots = []

    min_dx, max_dx = candidates
    data_start = boundaries["data_start"]

    dot_idx = 0

    for i, ((x_left, x_right), (n_missing, (cnt_min, cnt_max))) in enumerate(
        zip(missing_ranges, result)
    ):
        while dot_idx < len(dots_sorted) and dots_sorted[dot_idx][0] <= x_left:
            final_dots.append(dots_sorted[dot_idx])
            dot_idx += 1

        is_leading = (i == 0 and dot_idx == 0)
        if is_leading:
            cur_x = data_start
            final_dots.append((cur_x, np.nan))
            placed = 1
            n_missing = n_missing - 1
        else:
            cur_x = x_left
            placed = 0

        segments = [min_dx] * cnt_min + [max_dx] * cnt_max
        for d in segments[placed:placed + n_missing]:
            cur_x += d
            final_dots.append((cur_x, np.nan))

    while dot_idx < len(dots_sorted):
        final_dots.append(dots_sorted[dot_idx])
        dot_idx += 1

    return sorted(final_dots, key = lambda x:x[0])