#!/usr/bin/env python3
"""
Train final V4 models for operational deployment.
Hybrid approach:
- Hybrid Conv1D+MLP: Temperature, Humidity, Rain (better temporal modeling)
- XGBoost+Lag: Wind Speed, Gust Speed (better with current data)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent
MODEL_DIR = DATA_DIR / "models_v4"
MODEL_DIR.mkdir(exist_ok=True)

LAG_HOURS = 6
SEQ_LENGTH = 12

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

TARGET_MAP = {
    'temperature': {'obs': 'obs_temperature', 'fc': 'fc_temperature_2m', 'model': 'hybrid'},
    'humidity': {'obs': 'obs_humidity', 'fc': 'fc_relative_humidity_2m', 'model': 'hybrid'},
    'rain': {'obs': 'obs_rain', 'fc': 'fc_precipitation', 'model': 'hybrid'},
    'wind_speed': {'obs': 'obs_wind_speed', 'fc': 'fc_wind_speed_10m', 'model': 'xgboost'},
    'gust_speed': {'obs': 'obs_gust_speed', 'fc': 'fc_wind_gusts_10m', 'model': 'xgboost'},
}


def add_time_features(df):
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


def add_lag_features(df, feature_cols, lag_hours=6):
    df = df.copy()
    key_features = ['fc_temperature_2m', 'fc_pressure_msl', 'fc_cloud_cover',
                    'fc_precipitation', 'fc_wind_speed_10m', 'fc_relative_humidity_2m']
    lag_cols = []
    for col in key_features:
        if col in df.columns:
            for lag in range(1, lag_hours + 1):
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df[col].shift(lag)
                lag_cols.append(lag_col)
            df[f"{col}_trend3h"] = df[col] - df[col].shift(3)
            df[f"{col}_trend6h"] = df[col] - df[col].shift(6)
            lag_cols.extend([f"{col}_trend3h", f"{col}_trend6h"])
    return df, lag_cols


def transform_rain(y, inverse=False):
    y = np.array(y, dtype=np.float64)
    if inverse:
        return np.expm1(np.clip(y, -10, 10))
    else:
        return np.sign(y) * np.log1p(np.abs(y))


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


def train_xgboost_model(df, target_name, feature_cols, lag_cols):
    """Train and save XGBoost model for wind targets."""
    print(f"\nTraining XGBoost for {target_name}...")

    obs_col = TARGET_MAP[target_name]['obs']
    fc_col = TARGET_MAP[target_name]['fc']

    df = df.copy()
    df['bias'] = df[obs_col] - df[fc_col]

    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    all_features = feature_cols + lag_cols + time_features
    all_features = [f for f in all_features if f in df.columns]

    valid_mask = df[obs_col].notna() & df[fc_col].notna()
    for col in all_features:
        valid_mask &= df[col].notna()

    df_valid = df[valid_mask].copy()

    X = df_valid[all_features].values
    y = df_valid['bias'].values

    # Fit scalers on ALL data (for production use)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Train on all data
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    model.fit(X_scaled, y_scaled, verbose=False)

    # Save model and scalers
    model.save_model(str(MODEL_DIR / f"xgb_{target_name}.json"))
    joblib.dump(scaler_X, MODEL_DIR / f"scaler_X_{target_name}.pkl")
    joblib.dump(scaler_y, MODEL_DIR / f"scaler_y_{target_name}.pkl")

    print(f"  Saved XGBoost model for {target_name}")

    return {'features': all_features}


def train_hybrid_model(df, target_name, feature_cols, seq_length=12):
    """Train and save Hybrid Conv1D+MLP model."""
    print(f"\nTraining Hybrid Conv1D+MLP for {target_name}...")

    obs_col = TARGET_MAP[target_name]['obs']
    fc_col = TARGET_MAP[target_name]['fc']

    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    all_features = feature_cols + time_features
    all_features = [f for f in all_features if f in df.columns]

    df = df.copy().sort_values('datetime').reset_index(drop=True)
    df['bias'] = df[obs_col] - df[fc_col]

    if target_name == 'rain':
        df['bias_transformed'] = transform_rain(df['bias'])
    else:
        df['bias_transformed'] = df['bias']

    # Create sequences
    valid_mask = df[obs_col].notna() & df[fc_col].notna()
    for col in all_features:
        if col in df.columns:
            valid_mask &= df[col].notna()

    valid_indices = df[valid_mask].index.tolist()

    X_current, X_past, y_bias = [], [], []

    for idx in valid_indices:
        if idx < seq_length - 1:
            continue
        seq_indices = list(range(idx - seq_length + 1, idx + 1))
        times = df.loc[seq_indices, 'datetime'].values
        time_diffs = np.diff(times).astype('timedelta64[h]').astype(int)
        if not all(d == 1 for d in time_diffs):
            continue

        current_data = df.loc[idx, all_features].values.astype(np.float64)
        past_indices = list(range(idx - seq_length + 1, idx))
        past_data = df.loc[past_indices, all_features].values.astype(np.float64)

        if np.isnan(current_data).any() or np.isnan(past_data).any():
            continue

        X_current.append(current_data)
        X_past.append(past_data)
        y_bias.append(df.loc[idx, 'bias_transformed'])

    X_current = np.array(X_current)
    X_past = np.array(X_past)
    y_bias = np.array(y_bias)

    print(f"  Sequences: {len(X_current)}")

    # Fit scalers on ALL data
    n_features = X_current.shape[1]
    scaler_X = StandardScaler()
    all_features_data = np.vstack([X_current, X_past.reshape(-1, n_features)])
    scaler_X.fit(all_features_data)

    X_current_scaled = scaler_X.transform(X_current)
    X_past_scaled = np.zeros_like(X_past)
    for i in range(X_past.shape[1]):
        X_past_scaled[:, i, :] = scaler_X.transform(X_past[:, i, :])

    if target_name != 'rain':
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_bias.reshape(-1, 1)).ravel()
    else:
        scaler_y = None
        y_scaled = y_bias

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create model
    model = HybridConv1DMLP(n_features, seq_length, hidden_dim=64)
    model = model.to(device)

    # Training
    dataset = TensorDataset(
        torch.FloatTensor(X_current_scaled),
        torch.FloatTensor(X_past_scaled),
        torch.FloatTensor(y_scaled)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        total_loss = 0
        for x_curr, x_past, y_batch in loader:
            x_curr, x_past, y_batch = x_curr.to(device), x_past.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_curr, x_past)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    # Save model and scalers
    model = model.cpu()
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': n_features,
        'seq_length': seq_length,
        'hidden_dim': 64
    }, MODEL_DIR / f"hybrid_{target_name}.pt")

    joblib.dump(scaler_X, MODEL_DIR / f"scaler_X_{target_name}.pkl")
    if scaler_y is not None:
        joblib.dump(scaler_y, MODEL_DIR / f"scaler_y_{target_name}.pkl")

    print(f"  Saved Hybrid model for {target_name}")

    return {'features': all_features}


def main():
    print("=" * 60)
    print("TRAINING V4 MODELS FOR OPERATIONAL DEPLOYMENT")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Total rows: {len(df)}")

    df = add_time_features(df)

    feature_cols = [c for c in FORECAST_FEATURES if c in df.columns]
    df, lag_cols = add_lag_features(df, feature_cols, LAG_HOURS)

    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']

    config = {
        'version': 'v4',
        'approach': 'hybrid',
        'models': {},
        'forecast_features': feature_cols,
        'time_features': time_features,
        'lag_hours': LAG_HOURS,
        'seq_length': SEQ_LENGTH,
    }

    for target_name, target_info in TARGET_MAP.items():
        model_type = target_info['model']

        if model_type == 'xgboost':
            result = train_xgboost_model(df, target_name, feature_cols, lag_cols)
            config['models'][target_name] = {
                'type': 'xgboost',
                'features': result['features']
            }
        else:
            result = train_hybrid_model(df, target_name, feature_cols, SEQ_LENGTH)
            config['models'][target_name] = {
                'type': 'hybrid',
                'features': result['features']
            }

    # Save config
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: {MODEL_DIR}")
    print("\nModel assignments:")
    for target, info in TARGET_MAP.items():
        print(f"  {target}: {info['model']}")


if __name__ == "__main__":
    main()
