"""Compact, gap-preserving station archive for the static website."""
import argparse
import csv
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import requests

OUTPUT = Path(__file__).parent / 'docs' / 'station_history.json'
MODULES = {
    'NAModule1': ('outdoor', ['Temperature', 'Humidity'], ['temperature', 'humidity']),
    'NAMain': ('indoor', ['Pressure'], ['pressure']),
    'NAModule2': ('wind', ['WindStrength', 'WindAngle', 'GustStrength'], ['wind', 'direction', 'gust']),
    'NAModule3': ('rain', ['sum_rain'], ['rain']),
}
CSV_FIELDS = {'wind': 'wind_speed', 'direction': 'wind_direction', 'gust': 'gust_speed'}


def update_history(access_token=None, station=None, archive=None, now=None, path=OUTPUT, evaporation=False):
    now = int(now or time.time())
    end = now // 1800 * 1800
    # Keep 32 days so the UI can use 30 complete local calendar days, including DST.
    start = end - 32 * 86400
    previous = json.loads(path.read_text()) if path.exists() else {}
    rows = {r['time']: r for r in previous.get('readings', []) if start <= r['time'] < end}
    session = requests.Session()
    if access_token:
        session.headers['Authorization'] = f'Bearer {access_token}'
    for module_type, (name, types, fields) in MODULES.items():
        if archive:
            with (archive / f'{name}_30min.csv').open() as handle:
                for row in csv.DictReader(handle):
                    ts = int(row['timestamp'])
                    if start <= ts < end:
                        values = {key: float(row[CSV_FIELDS.get(key, key)])
                                  if row.get(CSV_FIELDS.get(key, key)) else None for key in fields}
                        rows.setdefault(ts, {'time': ts}).update(values)
        if not access_token:
            continue
        module = next((m for m in [station, *station.get('modules', [])]
                       if m['type'] == module_type), None)
        if not module:
            continue
        last = max((t for t, r in rows.items() if any(r.get(f) is not None for f in fields)), default=start)
        begin = max(start, (last // 1800) * 1800 - 7200)
        while begin < end:
            response = session.get('https://api.netatmo.com/api/getmeasure', params={
                'device_id': station['_id'], 'module_id': module['_id'],
                'scale': '30min', 'type': ','.join(types), 'date_begin': begin,
                'date_end': end - 1, 'limit': 1024, 'optimize': 'false', 'real_time': 'false',
            }, timeout=60)
            response.raise_for_status()
            payload = response.json()
            if payload.get('status') != 'ok':
                raise ValueError(f'History unavailable for {name}')
            page = payload['body']
            if not page:
                break
            for timestamp, values in page.items():
                ts = int(timestamp)
                if not begin <= ts < end or ts % 1800 != 900 or len(values) != len(fields):
                    raise ValueError(f'Unexpected history bucket for {name}')
                rows.setdefault(ts, {'time': ts}).update(zip(fields, values))
            begin = max(map(int, page)) + 900
    # Include empty buckets: charts must break across outages rather than join them.
    readings = [rows.get(ts, {'time': ts}) for ts in range(start + 900, end, 1800)]
    output = {'meta': {'updated_at': now, 'interval_seconds': 1800, 'start': start, 'end': end},
              'readings': readings,
              'evaporation': previous.get('evaporation', []),
              'evaporation_updated_at': previous.get('evaporation_updated_at')}
    if evaporation and now - (output['evaporation_updated_at'] or 0) > 6 * 3600:
        try:
            location = json.loads((Path(__file__).parent / 'docs' / 'horizon.json').read_text())['location']
            response = requests.get('https://historical-forecast-api.open-meteo.com/v1/forecast', params={
                'latitude': location['lat'], 'longitude': location['lon'],
                'models': 'meteoswiss_icon_ch2', 'hourly': 'et0_fao_evapotranspiration',
                'timezone': 'UTC', 'timeformat': 'unixtime',
                'start_date': datetime.fromtimestamp(start, timezone.utc).strftime('%Y-%m-%d'),
                'end_date': datetime.fromtimestamp(end, timezone.utc).strftime('%Y-%m-%d'),
            }, timeout=60)
            response.raise_for_status()
            hourly = response.json()['hourly']
            output['evaporation'] = [{'time': t, 'et0': value} for t, value in
                                      zip(hourly['time'], hourly['et0_fao_evapotranspiration']) if t < end]
            output['evaporation_updated_at'] = now
        except (requests.RequestException, KeyError, ValueError) as error:
            print(f'Evaporation history unavailable ({type(error).__name__}); previous data retained')
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--evaporation', action='store_true', help='Refresh Open-Meteo reference evaporation')
    args = parser.parse_args()
    result = update_history(archive=args.archive, evaporation=args.evaporation)
    print(f"Exported {len(result['readings'])} station intervals to {OUTPUT}")
