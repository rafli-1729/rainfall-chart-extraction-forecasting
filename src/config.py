from pathlib import Path
import os
import tomllib


# ---------- helpers ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _resolve(p: str) -> Path:
    return PROJECT_ROOT / p


# ---------- config sections ----------
class PathsConfig:
    def __init__(self, data: dict):
        self.data       = _resolve(data["data_dir"])
        self.raw        = _resolve(data["raw_dir"])
        self.processed  = _resolve(data["processed_dir"])
        self.clean      = _resolve(data["clean_dir"])
        self.inference  = _resolve(data["inference_dir"])
        self.interim    = _resolve(data["interim_dir"])

        self.models     = _resolve(data["models_dir"])
        self.database   = _resolve(data["database_dir"])

        self.assets  = _resolve(data["assets_dir"])
        self.templates  = _resolve(data["template_dir"])
        self.styles     = _resolve(data["style_dir"])

        self.metadata   = _resolve(data["metadata_dir"])


class APIConfig:
    def __init__(self, data: dict):
        self.host: str = data["host"]
        self.port: int = int(data["port"])


class WeatherConfig:
    def __init__(self, data: dict):
        self.timezone: str = data["timezone"]


class FeatureConfig:
    def __init__(self, data: dict):
        self.rolling_windows = data["rolling_windows"]
        self.rain_extreme_columns = data["rain_extreme_columns"]
        self.meteorogical_columns = data["meteorogical_columns"]
        self.locations = data["locations"]


# ---------- root config ----------
class AppConfig:
    def __init__(self, toml_path: Path):
        with open(toml_path, "rb") as f:
            raw = tomllib.load(f)

        self.paths = PathsConfig(raw["paths"])
        self.api = APIConfig(raw["api"])
        self.weather = WeatherConfig(raw["weather"])
        self.features = FeatureConfig(raw["features"])


# ---------- load once ----------
CONFIG_PATH = Path(
    os.getenv("APP_CONFIG", PROJECT_ROOT / "config" / "config.toml")
)

config = AppConfig(CONFIG_PATH)