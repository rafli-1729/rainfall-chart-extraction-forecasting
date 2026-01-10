-- name: melt_df
WITH base AS (
    SELECT *
    FROM weekly_rainfall_v2
    WHERE location = ? AND year = ?
)
SELECT date, type, rainfall
FROM (
    SELECT date, 'Observed' AS type, observed_mm AS rainfall FROM base WHERE observed_mm IS NOT NULL
    UNION ALL
    SELECT date, 'Predicted', cv_predicted_mm FROM base WHERE cv_predicted_mm IS NOT NULL
    UNION ALL
    SELECT date, 'Extracted', extracted_mm FROM base WHERE extracted_mm IS NOT NULL
)
ORDER BY date;

-- name: daily_df
SELECT * FROM daily_rainfall_v2


-- name: metrics
SELECT
    AVG(ABS(observed_mm - cv_predicted_mm))                     AS mae,
    AVG(POWER(observed_mm - cv_predicted_mm, 2))                AS mse,
    SQRT(AVG(POWER(observed_mm - cv_predicted_mm, 2)))          AS rmse
FROM daily_rainfall_v2
WHERE observed_mm IS NOT NULL
  AND cv_predicted_mm IS NOT NULL;


-- name: r2
WITH stats AS (
    SELECT
        observed_mm,
        cv_predicted_mm,
        AVG(observed_mm) OVER () AS mean_observed
    FROM daily_rainfall_v2
    WHERE observed_mm IS NOT NULL
      AND cv_predicted_mm IS NOT NULL
)
SELECT
    1
    - SUM(POWER(observed_mm - cv_predicted_mm, 2))
      / SUM(POWER(observed_mm - mean_observed, 2)) AS r2
FROM stats;


-- name: zero-rain
SELECT
    100.0 * SUM(
        CASE
            WHEN extracted_mm = 0
             AND cv_predicted_mm > ?
            THEN 1
            ELSE 0
        END
    ) / NULLIF(
        SUM(
            CASE
                WHEN extracted_mm = 0 THEN 1
                ELSE 0
            END
        ),
        0
    ) AS zero_rain_false_rate_pct
FROM daily_rainfall_v2;


-- name: mae
SELECT AVG(observed_mm - cv_predicted_mm) as mae
FROM daily_rainfall_v2
WHERE observed_mm IS NOT NULL
  AND cv_predicted_mm IS NOT NULL


-- name: available_years
SELECT DISTINCT year
FROM weekly_rainfall_v2
ORDER BY year;


-- name: available_locations
SELECT DISTINCT location
FROM weekly_rainfall_v2
ORDER BY location;


-- name: shape
SELECT COUNT(*) AS n FROM daily_rainfall_v2;


-- name: available_years_by_location
SELECT DISTINCT strftime('%Y', date) AS year
FROM weekly_rainfall_v2
WHERE location = ?
ORDER BY year;
