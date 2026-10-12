"""Shared CH2 features and lightweight station calibration."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
LAT, LON = 46.021245, 8.239861
SOURCE = "meteoswiss_icon_ch2"
TARGETS = {
    "temperature": "temperature_2m", "humidity": "relative_humidity_2m",
    "wind_speed": "wind_speed_10m", "gust_speed": "wind_gusts_10m", "rain": "rain",
}
VARIABLES = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature', 'precipitation', 'rain', 'snowfall', 'weather_code', 'pressure_msl', 'cloud_cover', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m', 'shortwave_radiation', 'direct_radiation', 'cape', 'is_day']
# Compact, physically relevant inputs shared by training and inference.
INPUTS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain",
          "snowfall", "pressure_msl", "cloud_cover", "wind_speed_10m",
          "wind_gusts_10m", "shortwave_radiation", "direct_radiation", "cape"]


def features(frame):
    """Use exactly the same CH2, lead-time and UTC calendar inputs in training and serving."""
    result = frame.reindex(columns=INPUTS).astype(float).copy()
    times = pd.DatetimeIndex(frame["time"])
    result["lead_hours"] = frame["lead_hours"].to_numpy()
    for name, phase in [("hour", times.hour / 24), ("year", (times.dayofyear - 1) / 365.25)]:
        result[name + "_sin"] = np.sin(2 * np.pi * phase)
        result[name + "_cos"] = np.cos(2 * np.pi * phase)
    angle = np.deg2rad(frame["wind_direction_10m"].to_numpy(dtype=float))
    result["wind_sin"], result["wind_cos"] = np.sin(angle), np.cos(angle)
    return result.astype("float32")


def constrain(values, target):
    if target == "humidity":
        return np.clip(values, 0, 100)
    return np.maximum(values, 0) if target != "temperature" else values


def calibrate(prediction, raw, spec, target):
    prediction = prediction + raw if spec["mode"] == "residual" else prediction
    weight = spec.get("correction_weight", 1.0)
    return constrain(raw + weight * (prediction - raw), target)


def station_data():
    """Hourly verification labels, with no interpolation across missing intervals."""
    frame = pd.read_csv(ROOT / "station_data_combined.csv")
    frame.index = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    grid = pd.date_range(frame.index.min().floor("h"), frame.index.max().ceil("h"), freq="15min")
    numeric = frame[list(TARGETS)].reindex(grid)
    hours = grid[grid.minute == 0]
    # Instantaneous forecast variables: interpolate the two surrounding midpoint values.
    labels = pd.DataFrame(index=hours)
    for name in ["temperature", "humidity", "wind_speed"]:
        labels[name] = ((numeric[name].shift(1) + numeric[name].shift(-1)) / 2).reindex(hours)
    previous = frame.copy()
    previous.index = previous.index.floor("h") + pd.Timedelta(hours=1)
    rain = previous["rain"].groupby(level=0)
    labels["rain"] = rain.sum(min_count=2).reindex(hours)
    gust = previous["gust_speed"].groupby(level=0)
    labels["gust_speed"] = gust.max().where(gust.count() == 2).reindex(hours)
    return labels
