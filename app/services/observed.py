import requests
import pandas as pd
import numpy as np

from src.config import PROJECT_ROOT

def _get_nea_rainfall_raw(date: str) -> pd.DataFrame:
    url = "https://api.data.gov.sg/v1/environment/rainfall"
    params = {"date": date}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    records = []
    for item in data["items"]:
        timestamp = pd.to_datetime(item["timestamp"])
        for r_ in item["readings"]:
            records.append({
                "timestamp": timestamp,
                "station_id": r_["station_id"],
                "rainfall_mm": r_["value"]
            })

    return pd.DataFrame(records)


def _get_nea_stations() -> pd.DataFrame:
    """
    Returns NEA rainfall station metadata.
    """
    url = "https://api.data.gov.sg/v1/environment/rainfall"
    r = requests.get(url, timeout=10)
    r.raise_for_status()

    stations = r.json()["metadata"]["stations"]

    return pd.DataFrame(stations)[
        ["id", "location"]
    ].rename(columns={"id": "station_id"})


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def get_observed_daily_rainfall(location: str, date: str) -> pd.DataFrame:
    # load location mapping
    mapping = pd.read_csv(PROJECT_ROOT / "data" / "locations.csv")
    row = mapping.loc[mapping["location"] == location]

    if row.empty:
        raise ValueError(f"Unknown location: {location}")

    lat = float(row["latitude"].iloc[0])
    lon = float(row["longitude"].iloc[0])

    # NEA data
    stations = _get_nea_stations()
    rain = _get_nea_rainfall_raw(date)

    try:
        rain["date"] = rain["timestamp"].dt.date
    except:
        return pd.DataFrame()

    daily_rain = (
        rain.groupby(["station_id", "date"], as_index=False)
            .agg(daily_rainfall_total_mm=("rainfall_mm", "sum"))
    )

    df = daily_rain.merge(stations, on="station_id", how="left")
    if df.empty:
        raise RuntimeError(f"No NEA rainfall data for date {date}")

    loc_df = pd.json_normalize(df["location"])
    df["latitude"] = loc_df["latitude"]
    df["longitude"] = loc_df["longitude"]

    # pick nearest station
    df["distance_km"] = _haversine(
        lat, lon,
        df['latitude'], df['longitude']
    )

    nearest = df.sort_values("distance_km").iloc[0]

    return pd.DataFrame([{
        "date": date,
        "location": location,
        "daily_rainfall_total_mm": nearest["daily_rainfall_total_mm"]
    }])


if __name__=='__main__':
    from tqdm import tqdm

    observed = get_observed_daily_rainfall(location='Bukit_Panjang', date='2021-07-12')

    mapping = pd.read_csv(PROJECT_ROOT / "src" / "locations.csv")
    location_coords = {
        row["location"]: (row["latitude"], row["longitude"])
        for _, row in mapping.iterrows()
    }

    def get_observed_rainfall_batch(test_df: pd.DataFrame) -> dict:
        """
        Returns:
        {(date, location): daily_rainfall_mm}
        """
        results = {}

        unique_dates = sorted(test_df["date"].dt.strftime("%Y-%m-%d").unique())

        for date in tqdm(unique_dates, desc="Fetching NEA per date"):
            # 1 request per date
            rain_raw = _get_nea_rainfall_raw(date)

            rain_raw["date"] = rain_raw["timestamp"].dt.date
            daily_rain = (
                rain_raw.groupby("station_id", as_index=False)
                .agg(daily_rainfall_total_mm=("rainfall_mm", "sum"))
            )

            stations = _get_nea_stations()
            df = daily_rain.merge(stations, on="station_id", how="left")

            # expand location dict
            loc_df = pd.json_normalize(df["location"])
            df["latitude"] = loc_df["latitude"]
            df["longitude"] = loc_df["longitude"]

            for location, (lat, lon) in location_coords.items():
                tmp = df.copy()
                tmp["distance_km"] = _haversine(
                    lat, lon,
                    tmp["latitude"], tmp["longitude"]
                )
                nearest = tmp.sort_values("distance_km").iloc[0]

                results[(date, location)] = nearest["daily_rainfall_total_mm"]

        return results


    test = pd.read_csv(PROJECT_ROOT/'data'/'clean'/"test.csv")
    test["date"] = pd.to_datetime(test["date"])
    predictions = []
    lookup = get_observed_rainfall_batch(test)

    test["prediksi"] = [
        lookup[(row["date"].strftime("%Y-%m-%d"), row["location"])]
        for _, row in test.iterrows()
    ]

    test["tahun"] = test["date"].dt.year
    test["bulan"] = test["date"].dt.month
    test["hari"] = test["date"].dt.day

    submission = test.rename(columns={
        "location": "ID"
    })[["ID", "tahun", "bulan", "hari", "prediksi"]]

    submission.to_csv("submission.csv", index=False)