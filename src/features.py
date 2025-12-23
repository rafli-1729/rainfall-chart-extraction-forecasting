import pandas as pd

def create_features(df):
    """
    Membuat fitur-fitur baru pada DataFrame cuaca.
    """
    # Salin DataFrame untuk menghindari SettingWithCopyWarning
    df = df.copy()

    # 1. Pastikan kolom 'date' dalam format datetime
    df['date'] = pd.to_datetime(df['date'])

    # --- 1. Ekstraksi Fitur Waktu ---
    print("Membuat fitur waktu...")
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    # --- 2. Rentang Temperatur Harian ---
    print("Membuat rentang temperatur harian...")
    df['temp_range_c'] = df['maximum_temperature_c'] - df['minimum_temperature_c']

    # --- 3. Faktor Hembusan Angin ---
    print("Membuat faktor hembusan angin...")
    # Ganti pembagian dengan nol atau NaN dengan 1 (menandakan angin stabil)
    df['wind_gust_factor'] = df['max_wind_speed_kmh'] / df['mean_wind_speed_kmh']
    df['wind_gust_factor'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['wind_gust_factor'].fillna(1, inplace=True)

    # --- 4. Rasio Intensitas Hujan ---
    print("Membuat rasio intensitas hujan...")
    # Ganti pembagian dengan nol atau NaN dengan 1 (menandakan intensitas stabil)
    df['rain_intensity_ratio'] = df['highest_60_min_rainfall_mm'] / df['highest_30_min_rainfall_mm']
    df['rain_intensity_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['rain_intensity_ratio'].fillna(1, inplace=True)

    # --- 5. Fitur Lag (Data 1 hari sebelumnya) ---
    print("Membuat fitur lag...")
    df = df.sort_values(by='date').reset_index(drop=True) # Pastikan data terurut sebelum membuat lag
    lag_features = ['mean_temperature_c', 'highest_60_min_rainfall_mm', 'mean_wind_speed_kmh']
    for feature in lag_features:
        df[f'{feature}_lag1'] = df[feature].shift(1)

    # --- 6. Statistik Bergulir (Rolling Statistics) ---
    print("Membuat statistik bergulir...")
    df['mean_temp_roll_7d'] = df['mean_temperature_c'].rolling(window=7, min_periods=1).mean()
    df['max_rain_roll_3d'] = df['highest_60_min_rainfall_mm'].rolling(window=3, min_periods=1).max()
    df['mean_wind_roll_7d'] = df['mean_wind_speed_kmh'].rolling(window=7, min_periods=1).mean()

    # --- FITUR SIKLUS & INTERAKSI AWAL ---
    days_in_year = 366
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / days_in_year)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / days_in_year)
    df['wind_x_rain'] = df['mean_wind_speed_kmh'] * df['highest_60_min_rainfall_mm']

    # --- FITUR BARU DARI DATA EKSTERNAL ---
    print("Menambahkan fitur interaksi dari data eksternal...")

    df['oni_x_temp'] = df['feature_oni'] * df['mean_temperature_c']
    df['dmi_x_rainfall'] = df['feature_dmi'] * df['highest_60_min_rainfall_mm']
    df['heat_index_proxy'] = df['feature_rh'] * df['mean_temperature_c']
    df['aqi_x_temp_range'] = df['feature_aqi'] * df['temp_range_c']
    print("-" * 30)
    return df