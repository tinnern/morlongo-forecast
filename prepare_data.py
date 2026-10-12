#!/usr/bin/env python3
"""Cache archived CH2 forecasts at day offsets 0–4 and join station observations."""
import argparse
import gzip
import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from weather_model import LAT, LON, ROOT, SOURCE, TARGETS, VARIABLES, station_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2025-07-29')
    parser.add_argument('--end', default=(datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat())
    args = parser.parse_args()
    cache = ROOT / 'data' / 'ch2'
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    current, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    parts = []
    while current <= end:
        stop = min(current + pd.Timedelta(days=20), end)
        path = cache / f'{current.date()}_{stop.date()}.json.gz'
        if path.exists():
            with gzip.open(path, 'rt') as handle:
                data = json.load(handle)
        else:
            response = session.get('https://previous-runs-api.open-meteo.com/v1/forecast', params={
                'latitude': LAT, 'longitude': LON, 'models': SOURCE, 'timezone': 'UTC',
                'start_date': str(current.date()), 'end_date': str(stop.date()),
                'hourly': ','.join(f'{var}_previous_day{day}' for day in range(5) for var in VARIABLES),
            }, timeout=120)
            if not response.ok:
                raise RuntimeError(f'Open-Meteo HTTP {response.status_code}: {response.text[:500]}')
            data = response.json()
            with gzip.open(path.with_suffix('.tmp'), 'wt') as handle:
                json.dump(data, handle)
            path.with_suffix('.tmp').replace(path)
            time.sleep(1)
        hourly = data['hourly']
        for day in range(5):
            suffix = f'_previous_day{day}' if day else ''
            part = pd.DataFrame({var: hourly.get(var + suffix, [None] * len(hourly['time'])) for var in VARIABLES})
            part['time'] = pd.to_datetime(hourly['time'], utc=True)
            part['lead_hours'] = day * 24
            parts.append(part)
        print(f'Cached {current.date()} through {stop.date()}', flush=True)
        current = stop + pd.Timedelta(days=1)
    forecast = pd.concat(parts, ignore_index=True).sort_values(['time', 'lead_hours']).reset_index(drop=True)
    assert not forecast.duplicated(['time', 'lead_hours']).any()
    labels = station_data()
    labels.to_csv(ROOT / 'station_hourly.csv', index_label='datetime')
    training = forecast.merge(labels.add_prefix('obs_'), left_on='time', right_index=True, how='left')
    training.to_pickle(ROOT / 'data' / 'training.pkl')
    summary = {
        'source': SOURCE, 'archive': 'previous-runs', 'first': str(forecast.time.min()), 'last': str(forecast.time.max()),
        'rows': len(training), 'leads_hours': [0, 24, 48, 72, 96],
        'paired_rows': {target: int((training['obs_' + target].notna() & training[var].notna()).sum()) for target, var in TARGETS.items()},
        'note': 'Day 0 is a stitched short-lead forecast, not a day-ahead forecast. Day offsets approximate lead time; exact issue timestamps are not provided.',
    }
    (ROOT / 'data' / 'coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
