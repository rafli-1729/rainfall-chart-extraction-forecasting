-- name: metrics
WITH base AS (
    SELECT
        extracted_mm,
        cv_predicted_mm,
        quantile(extracted_mm, 0.9) OVER () AS p90
    FROM daily_rainfall_v2
    WHERE location = ?
      AND EXTRACT(YEAR FROM date) = ?
      AND extracted_mm IS NOT NULL
      AND cv_predicted_mm IS NOT NULL
)
SELECT
    AVG(ABS(cv_predicted_mm - extracted_mm))                      AS mae,
    SQRT(AVG(POWER(cv_predicted_mm - extracted_mm, 2)))           AS rmse,
    AVG(cv_predicted_mm - extracted_mm)                           AS bias,
    CORR(extracted_mm, cv_predicted_mm)                           AS corr,

    100.0 * SUM(
        CASE
            WHEN extracted_mm = 0
             AND cv_predicted_mm > ?
            THEN 1 ELSE 0
        END
    ) / NULLIF(
        SUM(CASE WHEN extracted_mm = 0 THEN 1 ELSE 0 END),
        0
    )                                                             AS false_rain_rate,

    AVG(
        CASE
            WHEN extracted_mm >= p90
            THEN ABS(cv_predicted_mm - extracted_mm)
        END
    )                                                             AS extreme_mae
FROM base;

-- name: baseline_comparison
WITH base AS (
    SELECT
        extracted_mm,
        obs_predicted_mm,
        cv_predicted_mm,
        quantile(extracted_mm, 0.9) OVER () AS p90
    FROM daily_rainfall_v2
    WHERE location = ?
      AND EXTRACT(YEAR FROM date) = ?
      AND cv_predicted_mm IS NOT NULL
      AND extracted_mm IS NOT NULL
      AND obs_predicted_mm IS NOT NULL
),
metrics AS (
    SELECT
        AVG(ABS(cv_predicted_mm - extracted_mm)) AS year_mae,
        AVG(ABS(cv_predicted_mm - obs_predicted_mm))  AS baseline_mae,

        AVG(
            CASE
                WHEN extracted_mm >= p90
                THEN ABS(cv_predicted_mm - extracted_mm)
            END
        ) AS year_extreme_mae,

        AVG(
            CASE
                WHEN extracted_mm >= p90
                THEN ABS(cv_predicted_mm - obs_predicted_mm)
            END
        ) AS baseline_extreme_mae
    FROM base
)
SELECT
    year_mae,
    baseline_mae,
    100.0 * (baseline_mae - year_mae) / baseline_mae
        AS mae_improvement_pct,

    year_extreme_mae,
    baseline_extreme_mae,
    100.0 * (baseline_extreme_mae - year_extreme_mae)
          / baseline_extreme_mae
        AS extreme_improvement_pct
FROM metrics;


-- name: plot_error
SELECT
    date,
    ABS(cv_predicted_mm - extracted_mm) AS abs_error
FROM weekly_rainfall_v2
WHERE location = ?
  AND EXTRACT(YEAR FROM date) = ?
  AND cv_predicted_mm IS NOT NULL
  AND extracted_mm IS NOT NULL
ORDER BY date;


-- name: error_distribution
SELECT
    ABS(cv_predicted_mm - extracted_mm) AS abs_error
FROM daily_rainfall_v2
WHERE location = ?
  AND EXTRACT(YEAR FROM date) = ?
  AND cv_predicted_mm IS NOT NULL
  AND extracted_mm IS NOT NULL;


-- name: extreme_day_error
WITH base AS (
    SELECT
        date,
        extracted_mm,
        cv_predicted_mm,
        ABS(cv_predicted_mm - extracted_mm) AS abs_error
    FROM daily_rainfall_v2
    WHERE location = ?
      AND EXTRACT(YEAR FROM date) = ?
      AND cv_predicted_mm IS NOT NULL
      AND extracted_mm IS NOT NULL
),
threshold AS (
    SELECT quantile(extracted_mm, 0.9) AS p90
    FROM base
)
SELECT
    date,
    extracted_mm,
    abs_error
FROM base, threshold
WHERE extracted_mm >= p90
ORDER BY date;


-- name: weekly_timeseries
SELECT
    date,
    observed_mm,
    extracted_mm,
    cv_predicted_mm,
    ABS(observed_mm - extracted_mm) AS abs_error
FROM weekly_rainfall_v2
WHERE location = ?
  AND EXTRACT(YEAR FROM date) = ?
ORDER BY date;


-- name: daily_timeseries
SELECT
    date,
    observed_mm,
    extracted_mm,
    cv_predicted_mm,
    ABS(observed_mm - extracted_mm) AS abs_error
FROM daily_rainfall_v2
WHERE location = ?
  AND EXTRACT(YEAR FROM date) = ?
ORDER BY date;
