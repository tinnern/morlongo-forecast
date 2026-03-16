#!/usr/bin/env python3
"""
One-time horizon profile generator using Open-Elevation API.
Run once per location, saves horizon mask to JSON.
"""
import requests
import json
import math
from pathlib import Path

# Configuration
LAT = 46.021245
LON = 8.239861
OUTPUT = Path("docs/horizon.json")

# Sample distances to check for terrain (meters)
DISTANCES = [500, 1000, 2000, 5000, 10000, 20000]
# Azimuth steps (degrees) - every 5 degrees = 72 points
AZIMUTH_STEP = 5


def get_elevations(points):
    """Query Open-Meteo Elevation API (more reliable than Open-Elevation)."""
    lats = ",".join(str(p[0]) for p in points)
    lons = ",".join(str(p[1]) for p in points)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()["elevation"]


def point_at_distance(lat, lon, distance_m, azimuth_deg):
    """Calculate lat/lon at given distance and azimuth from origin."""
    R = 6371000  # Earth radius in meters
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    az_rad = math.radians(azimuth_deg)

    d = distance_m / R
    new_lat = math.asin(
        math.sin(lat_rad) * math.cos(d) +
        math.cos(lat_rad) * math.sin(d) * math.cos(az_rad)
    )
    new_lon = lon_rad + math.atan2(
        math.sin(az_rad) * math.sin(d) * math.cos(lat_rad),
        math.cos(d) - math.sin(lat_rad) * math.sin(new_lat)
    )
    return math.degrees(new_lat), math.degrees(new_lon)


def compute_horizon_angle(observer_elev, target_elev, distance_m):
    """Compute horizon angle (elevation) from observer to target."""
    elev_diff = target_elev - observer_elev
    angle_rad = math.atan2(elev_diff, distance_m)
    return math.degrees(angle_rad)


def main():
    print(f"Computing horizon profile for {LAT}, {LON}")

    # Get observer elevation
    observer_elev = get_elevations([(LAT, LON)])[0]
    print(f"Observer elevation: {observer_elev}m")

    # Build list of all sample points
    all_points = []
    point_info = []  # (azimuth, distance)

    for az in range(0, 360, AZIMUTH_STEP):
        for dist in DISTANCES:
            lat, lon = point_at_distance(LAT, LON, dist, az)
            all_points.append((lat, lon))
            point_info.append((az, dist))

    print(f"Querying {len(all_points)} elevation points...")

    # Query in batches (API limit)
    BATCH_SIZE = 100
    all_elevations = []
    for i in range(0, len(all_points), BATCH_SIZE):
        batch = all_points[i:i+BATCH_SIZE]
        elevs = get_elevations(batch)
        all_elevations.extend(elevs)
        print(f"  Batch {i//BATCH_SIZE + 1}/{(len(all_points)-1)//BATCH_SIZE + 1}")

    # Compute max horizon angle per azimuth
    horizon = {}
    for (az, dist), elev in zip(point_info, all_elevations):
        angle = compute_horizon_angle(observer_elev, elev, dist)
        if az not in horizon or angle > horizon[az]:
            horizon[az] = angle

    # Ensure minimum 0 degrees (can't have negative horizon unless in a pit)
    horizon = {az: max(0, angle) for az, angle in horizon.items()}

    # Save
    output = {
        "location": {"lat": LAT, "lon": LON, "elevation": observer_elev},
        "horizon": [{"azimuth": az, "elevation": horizon[az]} for az in sorted(horizon.keys())]
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT}")
    print(f"Max horizon angles: E={horizon.get(90, 0):.1f}° S={horizon.get(180, 0):.1f}° W={horizon.get(270, 0):.1f}°")


if __name__ == "__main__":
    main()
