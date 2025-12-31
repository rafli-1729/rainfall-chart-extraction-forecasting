import pandas as pd
import requests
import time
from io import StringIO

from src.dataset_builder import clean_column_names
from src.config import DATA_DIR

BASE_URL = "https://www.weather.gov.sg/files/dailydata/"
OUT_BASE = DATA_DIR/"wss"
OUT_BASE.mkdir(exist_ok=True)

YEARS = range(1980, 2026)
MONTHS = range(1, 13)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/csv,*/*;q=0.8",
    "Referer": "https://www.weather.gov.sg/climate-historical-daily/",
})

session.get("https://www.weather.gov.sg/climate-historical-daily/", timeout=15)

stations = [
    {'station_code': 'S104', 'station_name': 'Admiralty'},
    {'station_code': 'S105', 'station_name': 'Admiralty West'},
    {'station_code': 'S109', 'station_name': 'Ang Mo Kio'},
    {'station_code': 'S86',  'station_name': 'Boon Lay (East)'},
    {'station_code': 'S63',  'station_name': 'Boon Lay (West)'},
    {'station_code': 'S120', 'station_name': 'Botanic Garden'},
    {'station_code': 'S55',  'station_name': 'Buangkok'},
    {'station_code': 'S64',  'station_name': 'Bukit Panjang'},
    {'station_code': 'S90',  'station_name': 'Bukit Timah'},
    {'station_code': 'S92',  'station_name': 'Buona Vista'},
    {'station_code': 'S61',  'station_name': 'Chai Chee'},
    {'station_code': 'S24',  'station_name': 'Changi'},
    {'station_code': 'S114', 'station_name': 'Choa Chu Kang (Central)'},
    {'station_code': 'S121', 'station_name': 'Choa Chu Kang (South)'},
    {'station_code': 'S11',  'station_name': 'Choa Chu Kang (West)'},
    {'station_code': 'S50',  'station_name': 'Clementi'},
    {'station_code': 'S118', 'station_name': 'Dhoby Ghaut'},
    {'station_code': 'S107', 'station_name': 'East Coast Parkway'},
    {'station_code': 'S39',  'station_name': 'Jurong (East)'},
    {'station_code': 'S101', 'station_name': 'Jurong (North)'},
    {'station_code': 'S44',  'station_name': 'Jurong (West)'},
    {'station_code': 'S117', 'station_name': 'Jurong Island'},
    {'station_code': 'S33',  'station_name': 'Jurong Pier'},
    {'station_code': 'S31',  'station_name': 'Kampong Bahru'},
    {'station_code': 'S71',  'station_name': 'Kent Ridge'},
    {'station_code': 'S122', 'station_name': 'Khatib'},
    {'station_code': 'S66',  'station_name': 'Kranji Reservoir'},
    {'station_code': 'S112', 'station_name': 'Lim Chu Kang'},
    {'station_code': 'S08',  'station_name': 'Lower Peirce Reservoir'},
    {'station_code': 'S07',  'station_name': 'Macritchie Reservoir'},
    {'station_code': 'S40',  'station_name': 'Mandai'},
    {'station_code': 'S108', 'station_name': 'Marina Barrage'},
    {'station_code': 'S113', 'station_name': 'Marine Parade'},
    {'station_code': 'S111', 'station_name': 'Newton'},
    {'station_code': 'S119', 'station_name': 'Nicoll Highway'},
    {'station_code': 'S116', 'station_name': 'Pasir Panjang'},
    {'station_code': 'S94',  'station_name': 'Pasir Ris (Central)'},
    {'station_code': 'S29',  'station_name': 'Pasir Ris (West)'},
    {'station_code': 'S06',  'station_name': 'Paya Lebar'},
    {'station_code': 'S106', 'station_name': 'Pulau Ubin'},
    {'station_code': 'S81',  'station_name': 'Punggol'},
    {'station_code': 'S77',  'station_name': 'Queenstown'},
    {'station_code': 'S25',  'station_name': 'Seletar'},
    {'station_code': 'S102', 'station_name': 'Semakau Island'},
    {'station_code': 'S80',  'station_name': 'Sembawang'},
    {'station_code': 'S60',  'station_name': 'Sentosa Island'},
    {'station_code': 'S36',  'station_name': 'Serangoon'},
    {'station_code': 'S110', 'station_name': 'Serangoon North'},
    {'station_code': 'S84',  'station_name': 'Simei'},
    {'station_code': 'S79',  'station_name': 'Somerset (Road)'},
    {'station_code': 'S43',  'station_name': 'Tai Seng'},
    {'station_code': 'S78',  'station_name': 'Tanjong Katong'},
    {'station_code': 'S72',  'station_name': 'Tanjong Pagar'},
    {'station_code': 'S23',  'station_name': 'Tengah'},
    {'station_code': 'S88',  'station_name': 'Toa Payoh'},
    {'station_code': 'S89',  'station_name': 'Tuas'},
    {'station_code': 'S115', 'station_name': 'Tuas South'},
    {'station_code': 'S82',  'station_name': 'Tuas West'},
    {'station_code': 'S35',  'station_name': 'Ulu Pandan'},
    {'station_code': 'S69',  'station_name': 'Upper Peirce Reservoir'},
    {'station_code': 'S46',  'station_name': 'Upper Thomson'},
    {'station_code': 'S123', 'station_name': 'Whampoa'},
    {'station_code': 'S91',  'station_name': 'Yishun'}
]

def normalize_df(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    df = clean_column_names(df)

    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["location"] = station_name
    df["location"] = (
        df["location"]
        .str.replace(r"[()]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    final_cols = [
        "date",
        "highest_30_min_rainfall_mm",
        "highest_60_min_rainfall_mm",
        "highest_120_min_rainfall_mm",
        "mean_temperature_c",
        "maximum_temperature_c",
        "minimum_temperature_c",
        "mean_wind_speed_kmh",
        "max_wind_speed_kmh",
        "location",
        "daily_rainfall_total_mm",
    ]

    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {list(df.columns)}")

    return df[final_cols]

if __name__ == "__main__":
    print("Scraping Weather Dataset...")
    for st in stations:
        code = st["station_code"]
        name = st["station_name"]

        station_dir = OUT_BASE / code
        station_dir.mkdir(exist_ok=True)

        print(f"\n=== {code} | {name} ===")

        for y in YEARS:
            for m in MONTHS:
                ym = f"{y}{m:02d}"
                url = f"{BASE_URL}DAILYDATA_{code}_{ym}.csv"

                try:
                    r = session.get(url, timeout=15)
                    if r.status_code != 200 or len(r.text) < 50:
                        continue

                    df_raw = pd.read_csv(StringIO(r.text), encoding="utf-8-sig")
                    df_final = normalize_df(df_raw, name)

                    out_file = station_dir / f"{code}_{ym}.csv"
                    df_final.to_csv(out_file, index=False)

                    print("OK", out_file.name)
                    time.sleep(0.25)

                    break

                except Exception as e:
                    print("ERR", code, ym, e)
            break
        break
