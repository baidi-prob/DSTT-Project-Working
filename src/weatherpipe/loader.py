"""Data Ingestion and Multi-City Extraction Module for WeatherPipe."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILES = {
    "temp": "temperature.csv",
    "humidity": "humidity.csv",
    "pressure": "pressure.csv",
    "wind_speed": "wind_speed.csv",
    "wind_dir": "wind_direction.csv",
    "weather_desc": "weather_description.csv",
}


def get_available_cities(raw_dir: str | Path = "data/raw") -> list[str]:
    """Retrieve the list of available cities from city_attributes or temperature.csv."""
    raw_path = Path(raw_dir)
    attr_file = raw_path / "city_attributes.csv"
    if attr_file.exists():
        df_attr = pd.read_csv(attr_file)
        if "City" in df_attr.columns:
            return sorted(df_attr["City"].dropna().tolist())

    temp_file = raw_path / "temperature.csv"
    if temp_file.exists():
        df_temp = pd.read_csv(temp_file, nrows=2)
        cities = [col for col in df_temp.columns if col.lower() != "datetime"]
        return sorted(cities)

    return []


def load_raw_city_data(
    city: str = "San Francisco",
    raw_dir: str | Path = "data/raw",
    sample_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract, align, and consolidate all meteorological variables for a specific city.

    Args:
        city: Name of the target city (e.g. 'San Francisco', 'Chicago', 'Phoenix').
        raw_dir: Path to directory containing raw CSV files.
        sample_path: Optional path to a pre-extracted sample CSV if raw files are absent.

    Returns:
        pd.DataFrame with standardized columns:
        ['timestamp', 'temp', 'humidity', 'pressure', 'wind_speed', 'wind_dir', 'weather_desc']
        Temperature is normalized from Kelvin to Celsius.
    """
    # If a direct sample file is specified or raw dir doesn't exist, try sample
    if sample_path and Path(sample_path).exists():
        df = pd.read_csv(sample_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    raw_path = Path(raw_dir)
    primary_file = raw_path / RAW_FILES["temp"]

    if not primary_file.exists():
        fallback_sample = Path("data/sample/weather_sample.csv")
        if fallback_sample.exists():
            df = pd.read_csv(fallback_sample)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path} and no sample file available."
        )

    # 1. Load datetime column and temperature
    df_temp = pd.read_csv(primary_file)
    if "datetime" not in df_temp.columns:
        raise ValueError("Raw CSV missing 'datetime' column.")
    if city not in df_temp.columns:
        available = [c for c in df_temp.columns if c != "datetime"]
        raise ValueError(f"City '{city}' not found. Available cities: {available[:5]}...")

    df_merged = pd.DataFrame()
    df_merged["timestamp"] = pd.to_datetime(df_temp["datetime"])

    # Convert Kelvin to Celsius: T(C) = T(K) - 273.15
    df_merged["temp"] = pd.to_numeric(df_temp[city], errors="coerce") - 273.15

    # 2. Extract remaining variables
    for var_key in ["humidity", "pressure", "wind_speed", "wind_dir", "weather_desc"]:
        var_file = raw_path / RAW_FILES[var_key]
        if var_file.exists():
            df_var = pd.read_csv(var_file)
            if city in df_var.columns:
                if var_key == "weather_desc":
                    df_merged[var_key] = df_var[city].astype(str).replace("nan", np.nan)
                else:
                    df_merged[var_key] = pd.to_numeric(df_var[city], errors="coerce")
            else:
                df_merged[var_key] = np.nan
        else:
            df_merged[var_key] = np.nan

    # Sort chronologically and reset index
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    return df_merged


def save_curated_dataset(
    df: pd.DataFrame,
    out_path: str | Path,
    max_rows: int | None = None,
) -> Path:
    """Save a curated or sample slice of the meteorological dataframe."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_df = df.iloc[:max_rows] if max_rows else df
    save_df.to_csv(out_file, index=False)
    return out_file
