-- name: melt_df
WITH base AS (
    SELECT *
    FROM weekly_rainfall
    WHERE location = ? AND year = ?
)
SELECT Date, Type, Rainfall
FROM (
    SELECT Date, 'Observed' AS Type, observed_mm AS Rainfall FROM base WHERE observed_mm IS NOT NULL
    UNION ALL
    SELECT Date, 'Predicted', predicted_mm FROM base WHERE predicted_mm IS NOT NULL
    UNION ALL
    SELECT Date, 'Extracted', extracted_mm FROM base WHERE extracted_mm IS NOT NULL
)
WHERE Type IN ?
ORDER BY Date;

-- name: available_years
SELECT DISTINCT year
FROM weekly_rainfall
ORDER BY year;

-- name: available_locations
SELECT DISTINCT location
FROM weekly_rainfall
ORDER BY location;

-- name: shape
SELECT COUNT(*) AS n FROM daily_rainfall;
