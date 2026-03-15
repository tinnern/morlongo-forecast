#!/usr/bin/env python3
"""
Prepare sequence data for temporal (Conv1D) model training.
Creates sliding window sequences from hourly data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuration
SEQ_LENGTH = 12  # 12 hours of history
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "sequences"

# Feature columns (must match training)
FORECAST_FEATURES = [
    "fc_temperature_2m", "fc_relative_humidity_2m", "fc_dew_point_2m",
    "fc_apparent_temperature", "fc_precipitation", "fc_rain", "fc_snowfall",
    "fc_snow_depth", "fc_weather_code", "fc_pressure_msl", "fc_surface_pressure",
    "fc_cloud_cover", "fc_cloud_cover_low", "fc_cloud_cover_mid", "fc_cloud_cover_high",
    "fc_et0_fao_evapotranspiration", "fc_vapour_pressure_deficit",
    "fc_wind_speed_10m", "fc_wind_direction_10m", "fc_wind_gusts_10m",
    "fc_shortwave_radiation", "fc_direct_radiation", "fc_diffuse_radiation",
    "fc_direct_normal_irradiance", "fc_global_tilted_irradiance", "fc_terrestrial_radiation",
    "fc_cape", "fc_convective_inhibition", "fc_freezing_level_height", "fc_is_day"
]

TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]

TARGET_VARS = {
    'obs_temperature': 'temperature',
    'obs_humidity': 'humidity',
    'obs_rain': 'rain',
    'obs_wind_speed': 'wind_speed',
    'obs_gust_speed': 'gust_speed',
}


def add_time_features(df):
    """Add cyclical time features."""
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    return df


def normalize_features(df, feature_cols):
    """Normalize features to zero mean, unit variance. Returns stats for inference."""
    stats = {}
    df_norm = df.copy()

    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                std = 1
            df_norm[col] = (df[col] - mean) / std
            stats[col] = {'mean': float(mean), 'std': float(std)}

    return df_norm, stats


def create_sequences(df, feature_cols, target_col, seq_length=12):
    """
    Create sliding window sequences.

    For each timestamp t, creates:
    - X: features from [t-seq_length+1, ..., t] (shape: seq_length x n_features)
    - y: target at time t
    """
    # Ensure data is sorted by time
    df = df.sort_values('datetime').reset_index(drop=True)

    # Get valid indices (where target and all features are not NaN)
    valid_mask = df[target_col].notna()
    for col in feature_cols:
        if col in df.columns:
            valid_mask &= df[col].notna()

    valid_indices = df[valid_mask].index.tolist()

    sequences = []
    targets = []
    timestamps = []

    for idx in valid_indices:
        if idx < seq_length - 1:
            continue

        # Check if we have continuous data for the sequence
        seq_indices = list(range(idx - seq_length + 1, idx + 1))

        # Verify continuity (each hour should be 1 hour apart)
        times = df.loc[seq_indices, 'datetime'].values
        time_diffs = np.diff(times).astype('timedelta64[h]').astype(int)

        if not all(d == 1 for d in time_diffs):
            continue  # Skip non-continuous sequences

        # Extract sequence
        seq_data = df.loc[seq_indices, feature_cols].values

        # Check for NaN in sequence
        if np.isnan(seq_data).any():
            continue

        sequences.append(seq_data)
        targets.append(df.loc[idx, target_col])
        timestamps.append(df.loc[idx, 'datetime'])

    return np.array(sequences), np.array(targets), timestamps


def daily_holdout_indices(timestamps, holdout_fraction=0.10, seed=42):
    """Get train/test indices based on daily holdout."""
    dates = [t.date() for t in timestamps]
    unique_dates = list(set(dates))

    n_holdout = max(1, int(len(unique_dates) * holdout_fraction))
    np.random.seed(seed)
    holdout_dates = set(np.random.choice(unique_dates, size=n_holdout, replace=False))

    train_idx = [i for i, d in enumerate(dates) if d not in holdout_dates]
    test_idx = [i for i, d in enumerate(dates) if d in holdout_dates]

    return train_idx, test_idx, holdout_dates


def main():
    print("=" * 60)
    print("PREPARE SEQUENCE DATA FOR TEMPORAL MODEL")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load data
    print("\nLoading training data...")
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Total rows: {len(df)}")

    # Add time features
    df = add_time_features(df)

    # Define all feature columns
    feature_cols = FORECAST_FEATURES + TIME_FEATURES
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"Available features: {len(available_features)}")

    # Normalize features
    print("\nNormalizing features...")
    df_norm, norm_stats = normalize_features(df, available_features)

    # Save normalization stats
    with open(OUTPUT_DIR / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats to {OUTPUT_DIR / 'norm_stats.json'}")

    # Create sequences for each target
    all_metadata = {
        'seq_length': SEQ_LENGTH,
        'n_features': len(available_features),
        'features': available_features,
        'targets': {}
    }

    for target_col, target_name in TARGET_VARS.items():
        print(f"\n{'='*40}")
        print(f"Processing target: {target_name}")
        print(f"{'='*40}")

        # Create sequences
        X, y, timestamps = create_sequences(
            df_norm, available_features, target_col, SEQ_LENGTH
        )

        if len(X) < 100:
            print(f"Not enough sequences ({len(X)}), skipping...")
            continue

        print(f"Created {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")

        # Split into train/test
        train_idx, test_idx, holdout_dates = daily_holdout_indices(timestamps)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Holdout days: {len(holdout_dates)}")

        # Save sequences
        np.savez(
            OUTPUT_DIR / f"sequences_{target_name}.npz",
            X_train=X_train.astype(np.float32),
            y_train=y_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_test=y_test.astype(np.float32)
        )
        print(f"Saved to {OUTPUT_DIR / f'sequences_{target_name}.npz'}")

        all_metadata['targets'][target_name] = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'holdout_days': len(holdout_dates),
            'y_train_mean': float(y_train.mean()),
            'y_train_std': float(y_train.std())
        }

    # Save metadata
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sequence length: {SEQ_LENGTH} hours")
    print(f"Number of features: {len(available_features)}")
    print(f"Output directory: {OUTPUT_DIR}")

    for target_name, meta in all_metadata['targets'].items():
        print(f"\n{target_name}:")
        print(f"  Train: {meta['train_samples']}, Test: {meta['test_samples']}")


if __name__ == "__main__":
    main()
