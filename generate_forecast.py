#!/usr/bin/env python3
"""
Generate debiased forecast for Morlongo station.
Fetches ICON-CH2 forecast (5 days) and applies XGBoost debiasing.
Outputs JSON for the static webpage.
"""

import json
import numpy as np
import requests
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Timezone
TZ = ZoneInfo("Europe/Zurich")

# Configuration
LAT = 46.021245
LON = 8.239861
LOCATION_NAME = "Morlongo"
MODEL_DIR = Path(__file__).parent / "models_v2"
OUTPUT_DIR = Path(__file__).parent / "docs"

# All forecast variables (must match training)
FORECAST_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "precipitation_probability", "rain", "snowfall", "snow_depth", "weather_code",
    "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration",
    "vapour_pressure_deficit", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation",
    "cape", "convective_inhibition", "freezing_level_height", "is_day"
]

TARGETS = ["temperature", "humidity", "wind_speed", "gust_speed", "rain"]


def fetch_forecast():
    """Fetch 5-day ICON-CH2 forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(FORECAST_VARS),
        "models": "meteoswiss_icon_ch2",
        "forecast_days": 5,
        "timezone": "Europe/Zurich"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def prepare_features(hourly_data):
    """Prepare feature matrix from forecast data."""
    n_hours = len(hourly_data["time"])

    # Parse times
    times = [datetime.fromisoformat(t) for t in hourly_data["time"]]

    # Build feature matrix
    features = {}
    for var in FORECAST_VARS:
        features[f"fc_{var}"] = hourly_data.get(var, [None] * n_hours)

    # Add time features
    hours = [t.hour for t in times]
    months = [t.month for t in times]
    doys = [t.timetuple().tm_yday for t in times]

    features["hour_sin"] = [np.sin(2 * np.pi * h / 24) for h in hours]
    features["hour_cos"] = [np.cos(2 * np.pi * h / 24) for h in hours]
    features["month_sin"] = [np.sin(2 * np.pi * m / 12) for m in months]
    features["month_cos"] = [np.cos(2 * np.pi * m / 12) for m in months]
    features["doy_sin"] = [np.sin(2 * np.pi * d / 365) for d in doys]
    features["doy_cos"] = [np.cos(2 * np.pi * d / 365) for d in doys]

    return times, features


def load_models():
    """Load trained XGBoost models."""
    models = {}
    for target in TARGETS:
        model_path = MODEL_DIR / f"xgb_{target}.json"
        if model_path.exists():
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            models[target] = model
    return models


def apply_debiasing(features, models):
    """Apply debiasing models to forecast."""
    # Get feature order from config
    config_path = MODEL_DIR / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    feature_order = config["features"]

    # Build feature matrix
    n_hours = len(features["hour_sin"])
    X = np.zeros((n_hours, len(feature_order)))

    for i, feat in enumerate(feature_order):
        if feat in features:
            vals = features[feat]
            X[:, i] = [v if v is not None else 0 for v in vals]

    # Apply each model
    predictions = {}
    for target, model in models.items():
        pred = model.predict(X)

        # Clip to reasonable ranges
        if target == "temperature":
            pred = np.clip(pred, -30, 50)
        elif target == "humidity":
            pred = np.clip(pred, 0, 100)
        else:
            pred = np.clip(pred, 0, None)

        predictions[target] = pred.tolist()

    return predictions


def generate_output(times, features, predictions):
    """Generate JSON output for webpage."""
    output = {
        "meta": {
            "location": LOCATION_NAME,
            "lat": LAT,
            "lon": LON,
            "generated_at": datetime.now(TZ).isoformat(),
            "model": "XGBoost debiasing on MeteoSwiss ICON-CH2",
            "forecast_hours": len(times)
        },
        "hourly": []
    }

    for i, t in enumerate(times):
        hour_data = {
            "time": t.isoformat(),
            "raw": {
                "temperature": features["fc_temperature_2m"][i],
                "apparent_temperature": features["fc_apparent_temperature"][i],
                "humidity": features["fc_relative_humidity_2m"][i],
                "precipitation": features["fc_precipitation"][i],
                "precipitation_probability": features["fc_precipitation_probability"][i],
                "wind_speed": features["fc_wind_speed_10m"][i],
                "wind_gusts": features["fc_wind_gusts_10m"][i],
                "wind_direction": features["fc_wind_direction_10m"][i],
                "cloud_cover": features["fc_cloud_cover"][i],
                "pressure": features["fc_pressure_msl"][i],
                "weather_code": features["fc_weather_code"][i],
                "vpd": features["fc_vapour_pressure_deficit"][i],
                "et0": features["fc_et0_fao_evapotranspiration"][i],
                "is_day": features["fc_is_day"][i],
            },
            "debiased": {
                "temperature": round(predictions["temperature"][i], 1),
                "humidity": round(predictions["humidity"][i], 1),
                "wind_speed": round(predictions["wind_speed"][i], 1),
                "gust_speed": round(predictions["gust_speed"][i], 1),
                "rain": round(predictions["rain"][i], 2),
            }
        }
        output["hourly"].append(hour_data)

    return output


def save_forecast_history(output):
    """Save forecast data for historical comparison.

    Stores forecasts in a simple format: for each hour we track what
    the forecast predicted. As time passes and that hour becomes "past",
    the observation data can be compared against these stored forecasts.
    """
    from datetime import timedelta
    history_path = OUTPUT_DIR / "forecast_history.json"

    # Load existing history or create new
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {"meta": {}, "hourly": []}

    generated_at = output["meta"]["generated_at"]
    now = datetime.now(TZ)

    # Convert existing hourly list to dict for easier merging
    existing = {h["time"]: h for h in history.get("hourly", [])}

    # Store forecasts for the next 48 hours
    # For past hours (that we may have missed), we already have entries
    for h in output["hourly"][:48]:
        valid_time = h["time"]

        # Only add/update if we don't have an entry yet, or if this is still
        # in the future (we want the latest forecast for future hours)
        valid_dt = datetime.fromisoformat(valid_time)
        if valid_dt.tzinfo is None:
            valid_dt = valid_dt.replace(tzinfo=TZ)

        is_future = valid_dt > now
        has_existing = valid_time in existing

        # For future hours: always update with latest forecast
        # For past hours: keep the original forecast (don't overwrite)
        if is_future or not has_existing:
            existing[valid_time] = {
                "time": valid_time,
                "temperature": h["debiased"]["temperature"],
                "rain": h["debiased"]["rain"],
                "humidity": h["debiased"]["humidity"],
                "wind_speed": h["debiased"]["wind_speed"],
                "forecast_made": generated_at
            }

    # Clean old entries (keep last 14 days)
    cutoff = (now - timedelta(days=14)).replace(tzinfo=None).isoformat()
    existing = {k: v for k, v in existing.items() if k >= cutoff}

    # Sort by time and save
    hourly = sorted(existing.values(), key=lambda x: x["time"])

    result = {
        "meta": {"last_update": generated_at},
        "hourly": hourly
    }

    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)

    # Count past vs future forecasts
    past_count = sum(1 for h in hourly if datetime.fromisoformat(h["time"]).replace(tzinfo=TZ) <= now)
    print(f"Updated forecast history: {len(hourly)} hours ({past_count} past, {len(hourly) - past_count} future)")


def main():
    print(f"Generating forecast for {LOCATION_NAME} ({LAT}, {LON})")
    print(f"Time: {datetime.now(TZ).isoformat()}")

    # Fetch forecast
    print("Fetching ICON-CH2 forecast...")
    data = fetch_forecast()

    # Prepare features
    print("Preparing features...")
    times, features = prepare_features(data["hourly"])
    print(f"  {len(times)} hours")

    # Load models
    print("Loading models...")
    models = load_models()
    print(f"  Loaded: {list(models.keys())}")

    # Apply debiasing
    print("Applying debiasing...")
    predictions = apply_debiasing(features, models)

    # Generate output
    print("Generating output...")
    output = generate_output(times, features, predictions)

    # Save JSON
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "forecast.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")

    # Save forecast history for comparison
    print("Saving forecast history...")
    save_forecast_history(output)

    # Print summary
    print("\nForecast summary (next 24h):")
    for h in output["hourly"][:24:6]:
        t = h["time"].split("T")[1][:5]
        raw_t = h["raw"]["temperature"]
        deb_t = h["debiased"]["temperature"]
        print(f"  {t}: {raw_t}°C (raw) -> {deb_t}°C (debiased)")


if __name__ == "__main__":
    main()
