#!/usr/bin/env python3
"""
Generate debiased forecast using V4 hybrid models.
- Hybrid Conv1D+MLP: Temperature, Humidity, Rain
- XGBoost+Lag: Wind Speed, Gust Speed
"""

import json
import numpy as np
import requests
import joblib
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Timezone
TZ = ZoneInfo("Europe/Zurich")

# Configuration
LAT = 46.021245
LON = 8.239861
LOCATION_NAME = "Morlongo"
MODEL_DIR = Path(__file__).parent / "models_v4"
OUTPUT_DIR = Path(__file__).parent / "docs"
SEQ_LENGTH = 12
LAG_HOURS = 6

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

TARGET_MAP = {
    'temperature': {'fc': 'fc_temperature_2m', 'model': 'hybrid'},
    'humidity': {'fc': 'fc_relative_humidity_2m', 'model': 'hybrid'},
    'rain': {'fc': 'fc_precipitation', 'model': 'hybrid'},
    'wind_speed': {'fc': 'fc_wind_speed_10m', 'model': 'xgboost'},
    'gust_speed': {'fc': 'fc_wind_gusts_10m', 'model': 'xgboost'},
}


class HybridConv1DMLP(nn.Module):
    def __init__(self, n_features, seq_length, hidden_dim=64):
        super().__init__()
        self.current_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.past_fc = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.combined_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        self.relu = nn.ReLU()

    def forward(self, x_current, x_past):
        current_features = self.current_mlp(x_current)
        x_past = x_past.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x_past)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        past_features = self.past_fc(x)
        combined = torch.cat([current_features, past_features], dim=1)
        output = self.combined_mlp(combined)
        return output.squeeze(-1)


def transform_rain(y, inverse=False):
    y = np.array(y, dtype=np.float64)
    if inverse:
        return np.expm1(np.clip(y, -10, 10))
    else:
        return np.sign(y) * np.log1p(np.abs(y))


def fetch_forecast():
    """Fetch forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(FORECAST_VARS),
        "models": "meteoswiss_icon_ch2",
        "past_hours": SEQ_LENGTH + 6,  # Need history for sequences
        "forecast_days": 6,  # Request 6 days to ensure 120h available from any time of day
        "timezone": "Europe/Zurich"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def prepare_features(hourly_data):
    """Prepare feature matrix and time features."""
    n_hours = len(hourly_data["time"])
    times = [datetime.fromisoformat(t) for t in hourly_data["time"]]

    # Build features dict
    features = {}
    for var in FORECAST_VARS:
        key = f"fc_{var}"
        features[key] = hourly_data.get(var, [None] * n_hours)

    # Time features
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
    """Load all trained models."""
    models = {}

    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)

    for target_name, target_info in TARGET_MAP.items():
        model_type = target_info['model']

        if model_type == 'hybrid':
            # Load PyTorch model
            checkpoint = torch.load(MODEL_DIR / f"hybrid_{target_name}.pt", map_location='cpu')
            model = HybridConv1DMLP(
                checkpoint['n_features'],
                checkpoint['seq_length'],
                checkpoint['hidden_dim']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            scaler_X = joblib.load(MODEL_DIR / f"scaler_X_{target_name}.pkl")
            scaler_y_path = MODEL_DIR / f"scaler_y_{target_name}.pkl"
            scaler_y = joblib.load(scaler_y_path) if scaler_y_path.exists() else None

            models[target_name] = {
                'type': 'hybrid',
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'features': config['models'][target_name]['features']
            }

        else:  # xgboost
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(str(MODEL_DIR / f"xgb_{target_name}.json"))

            scaler_X = joblib.load(MODEL_DIR / f"scaler_X_{target_name}.pkl")
            scaler_y = joblib.load(MODEL_DIR / f"scaler_y_{target_name}.pkl")

            models[target_name] = {
                'type': 'xgboost',
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'features': config['models'][target_name]['features']
            }

    return models, config


def apply_hybrid_model(model_info, features, times, fc_col):
    """Apply Hybrid Conv1D+MLP model."""
    model = model_info['model']
    scaler_X = model_info['scaler_X']
    scaler_y = model_info['scaler_y']
    feature_names = model_info['features']

    n_hours = len(times)
    predictions = []

    for i in range(n_hours):
        if i < SEQ_LENGTH - 1:
            # Not enough history, use raw forecast
            fc_val = features[fc_col][i]
            predictions.append(fc_val if fc_val is not None else 0)
            continue

        # Build current features
        current = np.array([features[f][i] if features.get(f) and features[f][i] is not None else 0
                          for f in feature_names], dtype=np.float64)

        # Build past window
        past = []
        for j in range(i - SEQ_LENGTH + 1, i):
            row = np.array([features[f][j] if features.get(f) and features[f][j] is not None else 0
                           for f in feature_names], dtype=np.float64)
            past.append(row)
        past = np.array(past, dtype=np.float64)

        # Scale
        current_scaled = scaler_X.transform(current.reshape(1, -1))
        past_scaled = np.zeros_like(past)
        for k in range(past.shape[0]):
            past_scaled[k] = scaler_X.transform(past[k].reshape(1, -1))

        # Predict
        with torch.no_grad():
            x_curr = torch.FloatTensor(current_scaled)
            x_past = torch.FloatTensor(past_scaled).unsqueeze(0)
            bias_pred = model(x_curr, x_past).item()

        # Inverse transform
        if scaler_y is not None:
            bias_pred = scaler_y.inverse_transform([[bias_pred]])[0, 0]
        else:
            # Rain uses log transform
            bias_pred = transform_rain(bias_pred, inverse=True)

        # Final prediction
        fc_val = features[fc_col][i] if features[fc_col][i] is not None else 0
        predictions.append(fc_val + bias_pred)

    return predictions


def apply_xgboost_model(model_info, features, times, fc_col):
    """Apply XGBoost+Lag model."""
    model = model_info['model']
    scaler_X = model_info['scaler_X']
    scaler_y = model_info['scaler_y']
    feature_names = model_info['features']

    n_hours = len(times)
    predictions = []

    # Need to build lag features dynamically
    key_features = ['fc_temperature_2m', 'fc_pressure_msl', 'fc_cloud_cover',
                    'fc_precipitation', 'fc_wind_speed_10m', 'fc_relative_humidity_2m']

    for i in range(n_hours):
        if i < LAG_HOURS:
            # Not enough history
            fc_val = features[fc_col][i]
            predictions.append(fc_val if fc_val is not None else 0)
            continue

        # Build feature vector
        feature_vec = []
        for f in feature_names:
            if f.endswith('_trend6h'):
                base = f.replace('_trend6h', '')
                val = (features.get(base, [0]*n_hours)[i] or 0) - (features.get(base, [0]*n_hours)[i-6] or 0)
            elif f.endswith('_trend3h'):
                base = f.replace('_trend3h', '')
                val = (features.get(base, [0]*n_hours)[i] or 0) - (features.get(base, [0]*n_hours)[i-3] or 0)
            elif '_lag' in f:
                parts = f.rsplit('_lag', 1)
                base = parts[0]
                lag = int(parts[1])
                val = features.get(base, [0]*n_hours)[i-lag] if i >= lag else 0
                val = val if val is not None else 0
            else:
                val = features.get(f, [0]*n_hours)[i]
                val = val if val is not None else 0
            feature_vec.append(val)

        feature_vec = np.array(feature_vec, dtype=np.float64).reshape(1, -1)

        # Scale and predict
        feature_scaled = scaler_X.transform(feature_vec)
        bias_pred_scaled = model.predict(feature_scaled)[0]
        bias_pred = scaler_y.inverse_transform([[bias_pred_scaled]])[0, 0]

        # Final prediction
        fc_val = features[fc_col][i] if features[fc_col][i] is not None else 0
        predictions.append(fc_val + bias_pred)

    return predictions


def generate_output(times, features, predictions):
    """Generate JSON output."""
    # Filter to future hours only
    now = datetime.now(TZ)
    start_idx = 0
    for i, t in enumerate(times):
        if t.tzinfo is None:
            t = t.replace(tzinfo=TZ)
        if t >= now:
            start_idx = i
            break

    # Find last index with valid raw data
    # MeteoSwiss ICON-CH2 has ~5.5 day horizon from 00:00 UTC, so valid hours depend on time of day
    last_valid_idx = len(times) - 1
    for i in range(len(times) - 1, -1, -1):
        if features["fc_temperature_2m"][i] is not None and features["fc_wind_speed_10m"][i] is not None:
            last_valid_idx = i
            break

    # Limit to 120 hours (5 days) of forecast, but stop at last valid data
    MAX_HOURS = 120
    available_hours = last_valid_idx - start_idx + 1
    end_idx = min(start_idx + MAX_HOURS, last_valid_idx + 1)
    actual_hours = end_idx - start_idx

    if actual_hours < MAX_HOURS:
        print(f"  Note: MeteoSwiss model provides {available_hours}h from now (limit: {MAX_HOURS}h)")

    output = {
        "meta": {
            "location": LOCATION_NAME,
            "lat": LAT,
            "lon": LON,
            "generated_at": datetime.now(TZ).isoformat(),
            "model": "V4 Hybrid: Conv1D+MLP (temp/humidity/rain) + XGBoost (wind)",
            "forecast_hours": end_idx - start_idx
        },
        "hourly": []
    }

    for i in range(start_idx, end_idx):
        t = times[i]
        hour_data = {
            "time": t.isoformat() if hasattr(t, 'isoformat') else str(t),
            "raw": {
                "temperature": features["fc_temperature_2m"][i],
                "apparent_temperature": features["fc_apparent_temperature"][i],
                "humidity": features["fc_relative_humidity_2m"][i],
                "precipitation": features["fc_precipitation"][i],
                "precipitation_probability": features.get("fc_precipitation_probability", [None]*len(times))[i],
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
                "temperature": round(predictions['temperature'][i], 1),
                "humidity": round(np.clip(predictions['humidity'][i], 0, 100), 1),
                "wind_speed": round(max(0, predictions['wind_speed'][i]), 1),
                "gust_speed": round(max(0, predictions['gust_speed'][i]), 1),
                "rain": round(max(0, predictions['rain'][i]), 2),
            }
        }
        output["hourly"].append(hour_data)

    return output


def save_forecast_history(output):
    """Save forecast history for comparison."""
    from datetime import timedelta
    history_path = OUTPUT_DIR / "forecast_history.json"

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {"meta": {}, "hourly": []}

    generated_at = output["meta"]["generated_at"]
    now = datetime.now(TZ)

    existing = {h["time"]: h for h in history.get("hourly", [])}

    for h in output["hourly"][:48]:
        valid_time = h["time"]
        valid_dt = datetime.fromisoformat(valid_time)
        if valid_dt.tzinfo is None:
            valid_dt = valid_dt.replace(tzinfo=TZ)

        is_future = valid_dt > now

        if is_future or valid_time not in existing:
            existing[valid_time] = {
                "time": valid_time,
                "temperature": h["debiased"]["temperature"],
                "raw_temperature": h["raw"]["temperature"],
                "rain": h["debiased"]["rain"],
                "humidity": h["debiased"]["humidity"],
                "wind_speed": h["debiased"]["wind_speed"],
                "forecast_made": generated_at
            }

    cutoff = (now - timedelta(days=14)).replace(tzinfo=None).isoformat()
    existing = {k: v for k, v in existing.items() if k >= cutoff}

    hourly = sorted(existing.values(), key=lambda x: x["time"])

    result = {"meta": {"last_update": generated_at}, "hourly": hourly}

    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)

    past_count = sum(1 for h in hourly if datetime.fromisoformat(h["time"]).replace(tzinfo=TZ) <= now)
    print(f"Updated forecast history: {len(hourly)} hours ({past_count} past, {len(hourly) - past_count} future)")


def main():
    print(f"Generating V4 forecast for {LOCATION_NAME}")
    print(f"Time: {datetime.now(TZ).isoformat()}")

    # Fetch forecast
    print("Fetching forecast...")
    data = fetch_forecast()

    # Prepare features
    print("Preparing features...")
    times, features = prepare_features(data["hourly"])
    print(f"  {len(times)} hours")

    # Load models
    print("Loading V4 models...")
    models, config = load_models()
    print(f"  Loaded: {list(models.keys())}")

    # Apply models
    print("Applying debiasing models...")
    predictions = {}

    for target_name, model_info in models.items():
        fc_col = TARGET_MAP[target_name]['fc']

        if model_info['type'] == 'hybrid':
            predictions[target_name] = apply_hybrid_model(model_info, features, times, fc_col)
        else:
            predictions[target_name] = apply_xgboost_model(model_info, features, times, fc_col)

        print(f"  {target_name}: done")

    # Generate output
    print("Generating output...")
    output = generate_output(times, features, predictions)

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "forecast.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")

    # Save history
    print("Saving forecast history...")
    save_forecast_history(output)

    # Print summary
    print(f"\nForecast summary (next 24h):")
    for h in output["hourly"][:24:6]:
        t = h["time"].split("T")[1][:5]
        raw_t = h["raw"]["temperature"]
        deb_t = h["debiased"]["temperature"]
        print(f"  {t}: {raw_t}°C (raw) -> {deb_t}°C (debiased)")


if __name__ == "__main__":
    main()
