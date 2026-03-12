#!/usr/bin/env python3
"""
Fetch observations from Netatmo weather station.
Auto-discovers the weather station and fetches indoor/outdoor data.
Maintains a rolling 7-day history in observations.json.

Requires environment variables:
    NETATMO_CLIENT_ID
    NETATMO_CLIENT_SECRET
    NETATMO_REFRESH_TOKEN
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Timezone
TZ = ZoneInfo("Europe/Zurich")

# Configuration
TOKEN_URL = "https://api.netatmo.com/oauth2/token"
STATIONS_URL = "https://api.netatmo.com/api/getstationsdata"
OUTPUT_DIR = Path(__file__).parent / "docs"
OUTPUT_FILE = OUTPUT_DIR / "observations.json"
HISTORY_DAYS = 7


def get_access_token():
    """Get fresh access token using refresh token."""
    client_id = os.environ.get("NETATMO_CLIENT_ID")
    client_secret = os.environ.get("NETATMO_CLIENT_SECRET")
    refresh_token = os.environ.get("NETATMO_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        print("Error: Missing Netatmo credentials in environment variables")
        print("  NETATMO_CLIENT_ID:", "set" if client_id else "MISSING")
        print("  NETATMO_CLIENT_SECRET:", "set" if client_secret else "MISSING")
        print("  NETATMO_REFRESH_TOKEN:", "set" if refresh_token else "MISSING")
        return None, None

    response = requests.post(TOKEN_URL, data={
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    })

    if response.status_code != 200:
        print(f"Error refreshing token: {response.status_code}")
        print(response.text)
        return None, None

    tokens = response.json()
    return tokens.get("access_token"), tokens.get("refresh_token")


def fetch_station_data(access_token):
    """Fetch data from all stations."""
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(STATIONS_URL, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching station data: {response.status_code}")
        print(response.text)
        return None

    return response.json()


def find_weather_station(data):
    """Auto-discover weather station (not thermostat)."""
    devices = data.get("body", {}).get("devices", [])

    for device in devices:
        # Weather stations have type "NAMain" (indoor module)
        # Thermostats have type "NAPlug" or "NATherm1"
        device_type = device.get("type")

        if device_type == "NAMain":
            return device

    print("No weather station found. Available devices:")
    for device in devices:
        print(f"  - {device.get('station_name')} (type: {device.get('type')})")

    return None


def extract_observation(station):
    """Extract current observation from station data."""
    now = datetime.now(TZ)

    # Indoor module (main device)
    dashboard = station.get("dashboard_data", {})
    indoor = {
        "temperature": dashboard.get("Temperature"),
        "humidity": dashboard.get("Humidity"),
        "pressure": dashboard.get("Pressure"),
        "co2": dashboard.get("CO2"),
        "noise": dashboard.get("Noise"),
        "time_utc": dashboard.get("time_utc")
    }

    # Find outdoor module
    outdoor = {}
    rain = {}
    wind = {}

    for module in station.get("modules", []):
        module_type = module.get("type")
        module_data = module.get("dashboard_data", {})

        if module_type == "NAModule1":  # Outdoor
            outdoor = {
                "temperature": module_data.get("Temperature"),
                "humidity": module_data.get("Humidity"),
                "time_utc": module_data.get("time_utc")
            }
        elif module_type == "NAModule3":  # Rain gauge
            rain = {
                "rain_1h": module_data.get("sum_rain_1"),
                "rain_24h": module_data.get("sum_rain_24"),
                "time_utc": module_data.get("time_utc")
            }
        elif module_type == "NAModule2":  # Wind gauge
            wind = {
                "wind_strength": module_data.get("WindStrength"),
                "wind_angle": module_data.get("WindAngle"),
                "gust_strength": module_data.get("GustStrength"),
                "gust_angle": module_data.get("GustAngle"),
                "time_utc": module_data.get("time_utc")
            }

    observation = {
        "time": now.isoformat(),
        "station_name": station.get("station_name"),
        "indoor": indoor,
        "outdoor": outdoor,
    }

    # Add rain/wind if available
    if rain:
        observation["rain"] = rain
    if wind:
        observation["wind"] = wind

    return observation


def load_history():
    """Load existing observation history."""
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Could not parse existing observations.json")
    return {"observations": []}


def save_history(history, current_obs):
    """Save observation history, keeping only last N days."""
    cutoff = datetime.now(TZ) - timedelta(days=HISTORY_DAYS)

    # Filter old observations (handle both tz-aware and tz-naive timestamps)
    recent = []
    for obs in history.get("observations", []):
        obs_time = datetime.fromisoformat(obs["time"])
        # If naive, assume it was Europe/Zurich
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=TZ)
        if obs_time > cutoff:
            recent.append(obs)

    # Add current observation
    recent.append(current_obs)

    # Sort by time
    recent.sort(key=lambda x: x["time"])

    output = {
        "meta": {
            "station_name": current_obs.get("station_name"),
            "last_update": current_obs["time"],
            "history_days": HISTORY_DAYS,
            "observation_count": len(recent)
        },
        "current": current_obs,
        "observations": recent
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    return output


def main():
    print(f"Fetching Netatmo observations...")
    print(f"Time: {datetime.now(TZ).isoformat()}")

    # Get access token
    access_token, new_refresh_token = get_access_token()
    if not access_token:
        print("Failed to get access token")
        return False

    # Note: If refresh_token changed, you'd need to update the secret
    # For now, Netatmo refresh tokens are long-lived

    # Fetch station data
    data = fetch_station_data(access_token)
    if not data:
        print("Failed to fetch station data")
        return False

    # Find weather station
    station = find_weather_station(data)
    if not station:
        print("No weather station found")
        return False

    print(f"Found station: {station.get('station_name')}")

    # Extract current observation
    observation = extract_observation(station)

    # Print summary
    outdoor = observation.get("outdoor", {})
    indoor = observation.get("indoor", {})

    print(f"\nCurrent readings:")
    if outdoor.get("temperature") is not None:
        print(f"  Outdoor: {outdoor['temperature']}°C, {outdoor.get('humidity')}%")
    if indoor.get("temperature") is not None:
        print(f"  Indoor:  {indoor['temperature']}°C, {indoor.get('humidity')}%")
    if indoor.get("pressure") is not None:
        print(f"  Pressure: {indoor['pressure']} hPa")

    # Load history and save
    history = load_history()
    output = save_history(history, observation)

    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"History: {output['meta']['observation_count']} observations over {HISTORY_DAYS} days")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
