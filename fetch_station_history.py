#!/usr/bin/env python3
"""Download all available Netatmo history at 30-minute resolution.

Reads credentials from the local, Git-ignored .env file. Preserves missing
readings and writes UTC timestamps. Run: python3 fetch_station_history.py
"""

import csv
import json
import shlex
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import fetch_observations

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "extracted" / "netatmo"
FIELDS = ["temperature", "humidity", "rain", "wind_speed", "wind_direction",
          "gust_speed", "gust_direction"]
MODULES = {
    "NAMain": ("indoor", ["Temperature", "Humidity", "CO2", "Noise", "Pressure"],
               ["temperature", "humidity", "co2", "noise", "pressure"]),
    "NAModule1": ("outdoor", ["Temperature", "Humidity"], ["temperature", "humidity"]),
    "NAModule3": ("rain", ["sum_rain"], ["rain"]),
    "NAModule2": ("wind", ["WindStrength", "WindAngle", "GustStrength", "GustAngle"],
                  ["wind_speed", "wind_direction", "gust_speed", "gust_direction"]),
}


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def download_module(session, station, module, end):
    name, types, fields = MODULES[module["type"]]
    readings = {}
    start = 0
    while start <= end:
        response = session.get("https://api.netatmo.com/api/getmeasure", params={
            "device_id": station["_id"], "module_id": module["_id"],
            "scale": "30min", "type": ",".join(types),
            "date_begin": start, "date_end": end, "limit": 1024,
            "optimize": "false", "real_time": "false",
        }, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            raise RuntimeError(f"Netatmo history request failed for {name}")
        page = payload["body"]
        if not page:
            break
        for timestamp, values in page.items():
            timestamp = int(timestamp)
            if (timestamp < start or timestamp > end or timestamp % 1800 != 900
                    or len(values) != len(fields)):
                raise ValueError(f"Unexpected history page for {name}")
            readings[timestamp] = dict(zip(fields, values))
        # Netatmo anchors aggregation to date_begin and labels the bucket midpoint.
        # Continue at the next bucket boundary, not one second after its midpoint.
        start = max(readings) + 900
        print(f"  {name}: {len(readings)} readings through {utc(max(readings))} UTC", flush=True)
        time.sleep(0.3)
    rows = [{"timestamp": ts, "datetime": utc(ts), **values}
            for ts, values in sorted(readings.items())]
    write_csv(ARCHIVE / f"{name}_30min.csv", ["timestamp", "datetime", *fields], rows)
    return name, readings


def main():
    credentials_path = ROOT / ".env"
    credentials = dict(item.split("=", 1) for item in shlex.split(credentials_path.read_text()))
    response = requests.post("https://api.netatmo.com/oauth2/token", data={
        "grant_type": "refresh_token", "client_id": credentials["NETATMO_CLIENT_ID"],
        "client_secret": credentials["NETATMO_CLIENT_SECRET"],
        "refresh_token": credentials["NETATMO_REFRESH_TOKEN"],
    }, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Netatmo authentication failed: HTTP {response.status_code}")
    tokens = response.json()
    credentials["NETATMO_REFRESH_TOKEN"] = tokens["refresh_token"]
    credentials_path.chmod(0o600)
    credentials_path.write_text("".join(f"{key}={shlex.quote(value)}\n"
                                       for key, value in credentials.items()))
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + tokens["access_token"]
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    response = session.get(fetch_observations.STATIONS_URL, timeout=30)
    response.raise_for_status()
    stations = [s for s in response.json()["body"]["devices"]
                if s.get("station_name") == "Morlongo 17 (Weather Station)" and s.get("type") == "NAMain"]
    if len(stations) != 1:
        raise RuntimeError("Expected exactly one Morlongo weather station")
    station = stations[0]
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "station.json").write_text(json.dumps(station, indent=2) + "\n")
    # Exclude the unfinished 30-minute bucket; the current snapshot covers live data.
    end = int(time.time()) // 1800 * 1800 - 1
    combined = defaultdict(dict)
    summary = {"station_name": station["station_name"], "scale": "30min", "timezone": "UTC",
               "requested_through": utc(end), "modules": {}}
    for module in [station, *station.get("modules", [])]:
        if module["type"] not in MODULES:
            raise RuntimeError(f"Unsupported module type: {module['type']}")
        name, readings = download_module(session, station, module, end)
        timestamps = sorted(readings)
        summary["modules"][name] = {
            "rows": len(readings), "first": utc(timestamps[0]) if timestamps else None,
            "last": utc(timestamps[-1]) if timestamps else None,
            "missing_intervals_between_first_and_last":
                (timestamps[-1] - timestamps[0]) // 1800 + 1 - len(timestamps) if timestamps else 0,
        }
        if name != "indoor":
            for timestamp, values in readings.items():
                combined[timestamp].update(values)
    rows = [{"timestamp": ts, "datetime": utc(ts), **{field: values.get(field) for field in FIELDS}}
            for ts, values in sorted(combined.items())]
    if not rows:
        raise RuntimeError("No outdoor history returned; existing combined files preserved")
    write_csv(ROOT / "station_data_combined.csv", ["timestamp", "datetime", *FIELDS], rows)
    from weather_model import station_data
    hourly = station_data()
    hourly.to_csv(ROOT / "station_hourly.csv", index_label="datetime")
    summary["combined_rows"] = len(rows)
    summary["hourly_rows"] = len(hourly)
    summary["missing_values"] = {field: sum(row[field] is None for row in rows) for field in FIELDS}
    (ARCHIVE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fetch_observations.save_history(fetch_observations.load_history(),
                                    fetch_observations.extract_observation(station))
    from station_history import update_history
    update_history(archive=ARCHIVE)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
