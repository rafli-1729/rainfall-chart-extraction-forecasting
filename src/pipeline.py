import requests
import pandas as pd

from src.config import PROJECT_ROOT

def get_daily_weather(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "windspeed_10m_mean",
            "windspeed_10m_max",
            "precipitation_sum"
        ],
        "timezone": "Asia/Singapore"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame({
        "date": data["time"],
        "mean_temperature_c": data["temperature_2m_mean"],
        "maximum_temperature_c": data["temperature_2m_max"],
        "minimum_temperature_c": data["temperature_2m_min"],
        "mean_wind_speed_kmh": data["windspeed_10m_mean"],
        "max_wind_speed_kmh": data["windspeed_10m_max"]
    })

    return df


def get_15min_rain(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "minutely_15": ["precipitation"],
        "timezone": "Asia/Singapore"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["minutely_15"]

    df = pd.DataFrame({
        "time": pd.to_datetime(data["time"]),
        "rain_15min_mm": data["precipitation"]
    })

    return df


def compute_highest_rainfall(df_15min):
    df = df_15min.copy()
    df = df.sort_values("time")

    # rolling windows
    df["rain_30"] = df["rain_15min_mm"].rolling(2).sum()
    df["rain_60"] = df["rain_15min_mm"].rolling(4).sum()
    df["rain_120"] = df["rain_15min_mm"].rolling(8).sum()

    return {
        "highest_30_min_rainfall_mm": df["rain_30"].max(),
        "highest_60_min_rainfall_mm": df["rain_60"].max(),
        "highest_120_min_rainfall_mm": df["rain_120"].max()
    }


def build_feature_row(lat, lon, location, date):
    daily = get_daily_weather(lat, lon, date, date)
    rain_15 = get_15min_rain(lat, lon, date, date)
    highest = compute_highest_rainfall(rain_15)

    row = daily.iloc[0].to_dict()
    row.update(highest)
    row["location"] = location

    return pd.DataFrame([row])


def build_features_from_api(location, date):
    mapping = pd.read_csv(PROJECT_ROOT/'src'/'locations.csv')

    latitude = (
        mapping.loc[mapping['location'] == location, 'latitude']
            .iloc[0]
    )

    longitude = (
        mapping.loc[mapping['location'] == location, 'longitude']
            .iloc[0]
    )

    return build_feature_row(float(latitude), float(longitude), location, date)


def is_date_within_forecast_window(date: str) -> bool:
    today = pd.Timestamp.today().normalize()
    max_date = today + pd.Timedelta(days=14)
    return today <= pd.to_datetime(date) <= max_date


if __name__ == '__main__':
    data = build_features_from_api(**{
        'location': 'Admiralty',
        'date': '2025-12-24'
    })
    print(data)