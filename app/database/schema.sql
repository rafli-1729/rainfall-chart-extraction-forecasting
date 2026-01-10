CREATE TABLE IF NOT EXISTS daily_rainfall (
    date DATE,
    location TEXT,
    year INTEGER,
    observed_mm DOUBLE,
    predicted_mm DOUBLE,
    extracted_mm DOUBLE
);

CREATE TABLE IF NOT EXISTS weekly_rainfall (
    date DATE,
    location TEXT,
    year INTEGER,
    observed_mm DOUBLE,
    predicted_mm DOUBLE,
    extracted_mm DOUBLE
);