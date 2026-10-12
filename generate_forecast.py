#!/usr/bin/env python3
"""Generate the website forecast with small CPU-only CH2 calibrators."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

from weather_model import LAT, LON, MODEL_DIR, ROOT, SOURCE, TARGETS, VARIABLES, calibrate, constrain, features

TZ = ZoneInfo('Europe/Zurich')
OUTPUT_DIR = ROOT / 'docs'
RAW_FIELDS = {
    'temperature': 'temperature_2m', 'apparent_temperature': 'apparent_temperature',
    'humidity': 'relative_humidity_2m', 'precipitation': 'precipitation',
    'precipitation_probability': 'precipitation_probability', 'wind_speed': 'wind_speed_10m',
    'wind_gusts': 'wind_gusts_10m', 'wind_direction': 'wind_direction_10m',
    'cloud_cover': 'cloud_cover', 'pressure': 'pressure_msl', 'weather_code': 'weather_code',
    'vpd': 'vapour_pressure_deficit', 'et0': 'et0_fao_evapotranspiration', 'is_day': 'is_day',
}


def predict(frame):
    config = json.loads((MODEL_DIR / 'config.json').read_text())
    if config['source'] != SOURCE:
        raise ValueError('Training and forecast source differ')
    X = features(frame)
    result = {}
    for target, raw_field in TARGETS.items():
        spec = config['models'][target]
        if spec.get('correction_weight', 1) == 0:
            result[target] = constrain(frame[raw_field].to_numpy(float), target)
            continue
        model = xgb.Booster(params={'nthread': 2})
        model.load_model(MODEL_DIR / f'{target}.ubj')
        pred = model.predict(xgb.DMatrix(X[spec['features']]))
        result[target] = calibrate(pred, frame[raw_field].to_numpy(float), spec, target)
        if target == 'rain':
            supported = (frame['temperature_2m'] > 2) & (frame['snowfall'].fillna(0) == 0)
            result[target] = np.where(supported, result[target], frame[raw_field])
    return result


def generate_output(data, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    frame = pd.DataFrame(data['hourly'])
    frame['time'] = pd.to_datetime(frame['time'], utc=True)
    frame = frame[frame.time >= now.ceil('h')].head(120).reset_index(drop=True)
    # A missing baseline is unavailable weather, never an invented zero reading.
    complete = frame[list(TARGETS.values())].notna().all(axis=1)
    if not complete.all():
        first_missing = np.flatnonzero(~complete.to_numpy())[0]
        frame = frame.iloc[:first_missing].copy()
    if frame.empty:
        raise ValueError('CH2 returned no complete future forecast hours')
    frame['lead_hours'] = (frame.time - now).dt.total_seconds() / 3600
    predictions = predict(frame)
    output = {'meta': {'location': 'Morlongo', 'lat': LAT, 'lon': LON,
                       'generated_at': now.tz_convert(TZ).isoformat(),
                       'model': 'XGBoost station calibration / MeteoSwiss ICON-CH2',
                       'forecast_hours': len(frame)}, 'hourly': []}
    for i, row in frame.iterrows():
        raw = {key: float(row[value]) if pd.notna(row.get(value)) else None for key, value in RAW_FIELDS.items()}
        output['hourly'].append({
            'time': row.time.tz_convert(TZ).isoformat(),
            'raw': raw,
            'debiased': {target: round(float(values[i]), 2 if target == 'rain' else 1) for target, values in predictions.items()},
        })
    return output


def save_forecast_history(output, directory=OUTPUT_DIR):
    path = directory / 'forecast_history.json'
    history = json.loads(path.read_text()) if path.exists() else {'hourly': []}
    now = datetime.fromisoformat(output['meta']['generated_at'])
    existing = {h['time']: h for h in history.get('hourly', [])}
    for hour in output['hourly'][:48]:
        valid = datetime.fromisoformat(hour['time'])
        if valid.tzinfo is None:
            valid = valid.replace(tzinfo=TZ)
        key = valid.astimezone(TZ).replace(tzinfo=None).isoformat()
        if valid > now or key not in existing:
            existing[key] = {'time': key, 'temperature': hour['debiased']['temperature'],
                                    'raw_temperature': hour['raw']['temperature'], 'rain': hour['debiased']['rain'],
                                    'humidity': hour['debiased']['humidity'], 'wind_speed': hour['debiased']['wind_speed'],
                                    'forecast_made': output['meta']['generated_at']}
    cutoff = (now - timedelta(days=14)).replace(tzinfo=None).isoformat()
    result = {'meta': {'last_update': output['meta']['generated_at']},
              'hourly': [existing[key] for key in sorted(existing) if key >= cutoff]}
    write_json(path, result)


def write_json(path, content):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Use a saved Open-Meteo JSON response for offline verification.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.input:
        data = json.loads(args.input.read_text())
    else:
        response = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': LAT, 'longitude': LON, 'models': SOURCE, 'timezone': 'UTC',
            'forecast_days': 6, 'hourly': ','.join([*VARIABLES, 'precipitation_probability']),
        }, timeout=60)
        response.raise_for_status()
        data = response.json()
    output = generate_output(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'forecast.json', output)
    save_forecast_history(output, args.output_dir)
    print(f"Saved {len(output['hourly'])} forecast hours to {args.output_dir}")


if __name__ == '__main__':
    main()
