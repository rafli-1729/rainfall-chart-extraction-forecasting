-- name: metrics
WITH base AS (
    SELECT
        extracted_mm,
        cv_predicted_mm,
        quantile(extracted_mm, 0.9) OVER () AS p90
    FROM daily_rainfall_v2
    WHERE location = ?
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