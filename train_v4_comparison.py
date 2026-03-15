#!/usr/bin/env python3
"""
Train and compare models with temporal features:
1. XGBoost with lag features
2. Hybrid Conv1D + MLP (current timestep MLP + past window Conv1D)

Key improvements:
- Train on BIAS (observation - forecast) not raw values
- Proper standardization of features AND targets
- Handle precipitation distribution (log transform)
- Lag features for temporal context
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent
MODEL_DIR = DATA_DIR / "models_v4"
MODEL_DIR.mkdir(exist_ok=True)

LAG_HOURS = 6  # Hours of history to include
SEQ_LENGTH = 12  # For Conv1D past window

# Feature columns
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

# Map targets to their corresponding forecast features
TARGET_MAP = {
    'temperature': {'obs': 'obs_temperature', 'fc': 'fc_temperature_2m'},
    'humidity': {'obs': 'obs_humidity', 'fc': 'fc_relative_humidity_2m'},
    'rain': {'obs': 'obs_rain', 'fc': 'fc_precipitation'},
    'wind_speed': {'obs': 'obs_wind_speed', 'fc': 'fc_wind_speed_10m'},
    'gust_speed': {'obs': 'obs_gust_speed', 'fc': 'fc_wind_gusts_10m'},
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


def add_lag_features(df, feature_cols, lag_hours=6):
    """Add lag features and compute trends."""
    df = df.copy()

    # Key features to lag (most important for weather prediction)
    key_features = [
        'fc_temperature_2m', 'fc_pressure_msl', 'fc_cloud_cover',
        'fc_precipitation', 'fc_wind_speed_10m', 'fc_relative_humidity_2m'
    ]

    lag_cols = []
    for col in key_features:
        if col in df.columns:
            for lag in range(1, lag_hours + 1):
                lag_col = f"{col}_lag{lag}"
                df[lag_col] = df[col].shift(lag)
                lag_cols.append(lag_col)

            # Add trend features (change from N hours ago)
            df[f"{col}_trend3h"] = df[col] - df[col].shift(3)
            df[f"{col}_trend6h"] = df[col] - df[col].shift(6)
            lag_cols.extend([f"{col}_trend3h", f"{col}_trend6h"])

    return df, lag_cols


def transform_target(y, target_name, inverse=False):
    """Transform target variable to handle distribution."""
    y = np.array(y, dtype=np.float64)

    if target_name == 'rain':
        # Log transform for precipitation (handles skewness and zeros)
        # Clip negative values to 0 (can occur due to bias calculation)
        if inverse:
            return np.expm1(np.clip(y, -10, 10))  # inverse of log1p, clip to avoid overflow
        else:
            # For bias, we can have negative values - use signed log transform
            # sign(x) * log(1 + |x|)
            return np.sign(y) * np.log1p(np.abs(y))
    else:
        return y


def daily_holdout_split(df, holdout_fraction=0.10, seed=42):
    """Split by randomly holding out 10% of DAYS."""
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    unique_dates = df['date'].unique()
    n_holdout = max(1, int(len(unique_dates) * holdout_fraction))

    np.random.seed(seed)
    holdout_dates = set(np.random.choice(unique_dates, size=n_holdout, replace=False))

    train_mask = ~df['date'].isin(holdout_dates)
    test_mask = df['date'].isin(holdout_dates)

    return train_mask, test_mask, holdout_dates


# ============================================================
# XGBoost with Lag Features
# ============================================================

def train_xgboost_lag(df, target_name, feature_cols, lag_cols):
    """Train XGBoost model with lag features, predicting BIAS."""
    print(f"\n{'='*60}")
    print(f"XGBoost with Lag Features: {target_name}")
    print(f"{'='*60}")

    obs_col = TARGET_MAP[target_name]['obs']
    fc_col = TARGET_MAP[target_name]['fc']

    # Compute bias (observation - forecast)
    df = df.copy()
    df['bias'] = df[obs_col] - df[fc_col]

    # Transform bias for rain
    df['bias_transformed'] = transform_target(df['bias'], target_name)

    # All features: current + lags + time
    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    all_features = feature_cols + lag_cols + time_features
    all_features = [f for f in all_features if f in df.columns]

    # Filter valid rows
    valid_mask = df[obs_col].notna() & df[fc_col].notna() & df['bias'].notna()
    for col in all_features:
        valid_mask &= df[col].notna()

    df_valid = df[valid_mask].copy()
    print(f"Valid samples: {len(df_valid)}")

    if len(df_valid) < 100:
        print("Not enough data")
        return None

    # Daily holdout split
    train_mask, test_mask, holdout_dates = daily_holdout_split(df_valid)
    train_df = df_valid[train_mask]
    test_df = df_valid[test_mask]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Holdout days: {len(holdout_dates)}")

    X_train = train_df[all_features].values
    y_train = train_df['bias_transformed'].values
    X_test = test_df[all_features].values
    y_test_transformed = test_df['bias_transformed'].values
    y_test_raw = test_df['bias'].values

    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Standardize target (for non-rain targets)
    if target_name != 'rain':
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    else:
        scaler_y = None
        y_train_scaled = y_train

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train_scaled,
              eval_set=[(X_test_scaled, y_test_transformed if target_name == 'rain'
                        else scaler_y.transform(y_test_transformed.reshape(-1, 1)).ravel())],
              verbose=False)

    # Predict
    y_pred_scaled = model.predict(X_test_scaled)

    # Inverse transform predictions
    if target_name != 'rain':
        y_pred_bias = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_pred_bias = transform_target(y_pred_scaled, target_name, inverse=True)

    # Compute final prediction (forecast + predicted bias)
    fc_test = test_df[fc_col].values
    y_pred_final = fc_test + y_pred_bias
    y_true_final = test_df[obs_col].values

    # Metrics on final predictions
    mae = mean_absolute_error(y_true_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    r2 = r2_score(y_true_final, y_pred_final)

    # Compare to raw forecast
    raw_mae = mean_absolute_error(y_true_final, fc_test)
    raw_rmse = np.sqrt(mean_squared_error(y_true_final, fc_test))
    raw_r2 = r2_score(y_true_final, fc_test)

    print(f"\nRaw Forecast:  MAE={raw_mae:.4f}, RMSE={raw_rmse:.4f}, R²={raw_r2:.4f}")
    print(f"XGB+Lag:       MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    print(f"Improvement:   MAE: {(raw_mae-mae)/raw_mae*100:.1f}%, R²: {(r2-raw_r2)/(1-raw_r2)*100:.1f}%")

    # Feature importance (top 10)
    importance = dict(zip(all_features, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 features:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")

    return {
        'model': 'XGBoost+Lag',
        'target': target_name,
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'raw_mae': float(raw_mae),
        'raw_r2': float(raw_r2),
        'improvement_mae_pct': float((raw_mae-mae)/raw_mae*100),
        'top_features': dict(top_features)
    }


# ============================================================
# Hybrid Conv1D + MLP Model
# ============================================================

class HybridConv1DMLP(nn.Module):
    """
    Hybrid model:
    - Current timestep → MLP
    - Past window → Conv1D
    - Combined → MLP → Output
    """
    def __init__(self, n_features, seq_length, hidden_dim=64):
        super().__init__()

        # Current timestep MLP (more weight to current)
        self.current_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Past window Conv1D
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

        # Combined MLP
        self.combined_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        self.relu = nn.ReLU()

    def forward(self, x_current, x_past):
        """
        x_current: [batch, n_features] - current timestep
        x_past: [batch, seq_length-1, n_features] - past window
        """
        # Process current timestep
        current_features = self.current_mlp(x_current)  # [batch, hidden_dim]

        # Process past window with Conv1D
        x_past = x_past.permute(0, 2, 1)  # [batch, n_features, seq_length-1]
        x = self.relu(self.bn1(self.conv1(x_past)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # [batch, 64]
        past_features = self.past_fc(x)  # [batch, hidden_dim]

        # Combine
        combined = torch.cat([current_features, past_features], dim=1)  # [batch, hidden_dim*2]
        output = self.combined_mlp(combined)

        return output.squeeze(-1)


def prepare_hybrid_sequences(df, feature_cols, target_name, seq_length=12):
    """Prepare data for hybrid model: current + past window."""
    obs_col = TARGET_MAP[target_name]['obs']
    fc_col = TARGET_MAP[target_name]['fc']

    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # Compute bias
    df['bias'] = df[obs_col] - df[fc_col]
    df['bias_transformed'] = transform_target(df['bias'], target_name)

    # Valid mask
    valid_mask = df[obs_col].notna() & df[fc_col].notna()
    for col in feature_cols:
        if col in df.columns:
            valid_mask &= df[col].notna()

    valid_indices = df[valid_mask].index.tolist()

    X_current = []
    X_past = []
    y_bias = []
    timestamps = []

    for idx in valid_indices:
        if idx < seq_length - 1:
            continue

        # Check continuity
        seq_indices = list(range(idx - seq_length + 1, idx + 1))
        times = df.loc[seq_indices, 'datetime'].values
        time_diffs = np.diff(times).astype('timedelta64[h]').astype(int)

        if not all(d == 1 for d in time_diffs):
            continue

        # Current timestep features
        current_data = df.loc[idx, feature_cols].values.astype(np.float64)

        # Past window features (excluding current)
        past_indices = list(range(idx - seq_length + 1, idx))
        past_data = df.loc[past_indices, feature_cols].values.astype(np.float64)

        if np.isnan(current_data).any() or np.isnan(past_data).any():
            continue

        X_current.append(current_data)
        X_past.append(past_data)
        y_bias.append(df.loc[idx, 'bias_transformed'])
        timestamps.append(df.loc[idx, 'datetime'])

    return (np.array(X_current), np.array(X_past),
            np.array(y_bias), timestamps)


def train_hybrid_conv1d(df, target_name, feature_cols, seq_length=12):
    """Train hybrid Conv1D+MLP model."""
    print(f"\n{'='*60}")
    print(f"Hybrid Conv1D+MLP: {target_name}")
    print(f"{'='*60}")

    obs_col = TARGET_MAP[target_name]['obs']
    fc_col = TARGET_MAP[target_name]['fc']

    # Prepare sequences
    X_current, X_past, y_bias, timestamps = prepare_hybrid_sequences(
        df, feature_cols, target_name, seq_length
    )

    if len(X_current) < 100:
        print(f"Not enough sequences: {len(X_current)}")
        return None

    print(f"Total sequences: {len(X_current)}")
    print(f"X_current shape: {X_current.shape}")
    print(f"X_past shape: {X_past.shape}")

    # Daily holdout split
    dates = [t.date() for t in timestamps]
    unique_dates = list(set(dates))
    n_holdout = max(1, int(len(unique_dates) * 0.10))
    np.random.seed(42)
    holdout_dates = set(np.random.choice(unique_dates, size=n_holdout, replace=False))

    train_idx = [i for i, d in enumerate(dates) if d not in holdout_dates]
    test_idx = [i for i, d in enumerate(dates) if d in holdout_dates]

    X_current_train, X_current_test = X_current[train_idx], X_current[test_idx]
    X_past_train, X_past_test = X_past[train_idx], X_past[test_idx]
    y_train, y_test = y_bias[train_idx], y_bias[test_idx]

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}, Holdout days: {len(holdout_dates)}")

    # Standardize features
    n_features = X_current.shape[1]
    scaler_X = StandardScaler()

    # Fit on all training data (current + past flattened)
    all_train_features = np.vstack([
        X_current_train,
        X_past_train.reshape(-1, n_features)
    ])
    scaler_X.fit(all_train_features)

    X_current_train_scaled = scaler_X.transform(X_current_train)
    X_current_test_scaled = scaler_X.transform(X_current_test)

    # Scale past sequences
    X_past_train_scaled = np.zeros_like(X_past_train)
    X_past_test_scaled = np.zeros_like(X_past_test)
    for i in range(X_past_train.shape[1]):
        X_past_train_scaled[:, i, :] = scaler_X.transform(X_past_train[:, i, :])
    for i in range(X_past_test.shape[1]):
        X_past_test_scaled[:, i, :] = scaler_X.transform(X_past_test[:, i, :])

    # Standardize target (for non-rain)
    if target_name != 'rain':
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    else:
        scaler_y = None
        y_train_scaled = y_train
        y_test_scaled = y_test

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Create model
    model = HybridConv1DMLP(n_features, seq_length, hidden_dim=64)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_current_train_scaled),
        torch.FloatTensor(X_past_train_scaled),
        torch.FloatTensor(y_train_scaled)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_current_test_scaled),
        torch.FloatTensor(X_past_test_scaled),
        torch.FloatTensor(y_test_scaled)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 15

    for epoch in range(100):
        model.train()
        train_loss = 0
        for x_curr, x_past, y_batch in train_loader:
            x_curr = x_curr.to(device)
            x_past = x_past.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_curr, x_past)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_curr, x_past, y_batch in test_loader:
                x_curr = x_curr.to(device)
                x_past = x_past.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_curr, x_past)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_curr, x_past, _ in test_loader:
            x_curr = x_curr.to(device)
            x_past = x_past.to(device)
            y_pred = model(x_curr, x_past)
            all_preds.extend(y_pred.cpu().numpy())

    y_pred_scaled = np.array(all_preds)

    # Inverse transform predictions
    if target_name != 'rain':
        y_pred_bias = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_pred_bias = transform_target(y_pred_scaled, target_name, inverse=True)

    # Get raw bias for comparison
    y_true_bias = transform_target(y_test, target_name, inverse=True) if target_name == 'rain' else y_test

    # Get forecast values for test set to compute final predictions
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    test_timestamps = [timestamps[i] for i in test_idx]

    fc_test = []
    obs_test = []
    for ts in test_timestamps:
        row = df_sorted[df_sorted['datetime'] == ts]
        if len(row) > 0:
            fc_test.append(row[fc_col].values[0])
            obs_test.append(row[obs_col].values[0])

    fc_test = np.array(fc_test)
    obs_test = np.array(obs_test)

    # Final predictions
    y_pred_final = fc_test + y_pred_bias
    y_true_final = obs_test

    # Metrics
    mae = mean_absolute_error(y_true_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    r2 = r2_score(y_true_final, y_pred_final)

    # Raw forecast comparison
    raw_mae = mean_absolute_error(y_true_final, fc_test)
    raw_rmse = np.sqrt(mean_squared_error(y_true_final, fc_test))
    raw_r2 = r2_score(y_true_final, fc_test)

    print(f"\nRaw Forecast:  MAE={raw_mae:.4f}, RMSE={raw_rmse:.4f}, R²={raw_r2:.4f}")
    print(f"Hybrid Conv:   MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    print(f"Improvement:   MAE: {(raw_mae-mae)/raw_mae*100:.1f}%, R²: {(r2-raw_r2)/(1-raw_r2)*100:.1f}%")

    return {
        'model': 'HybridConv1D+MLP',
        'target': target_name,
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'raw_mae': float(raw_mae),
        'raw_r2': float(raw_r2),
        'improvement_mae_pct': float((raw_mae-mae)/raw_mae*100),
        'n_params': n_params
    }


def main():
    print("=" * 70)
    print("MODEL COMPARISON: XGBoost+Lag vs Hybrid Conv1D+MLP")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Total rows: {len(df)}")

    # Add time features
    df = add_time_features(df)

    # Get available feature columns
    feature_cols = [c for c in FORECAST_FEATURES if c in df.columns]
    time_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']

    # Add lag features
    df, lag_cols = add_lag_features(df, feature_cols, LAG_HOURS)

    print(f"Forecast features: {len(feature_cols)}")
    print(f"Lag features: {len(lag_cols)}")
    print(f"Time features: {len(time_features)}")

    all_results = {'xgboost_lag': {}, 'hybrid_conv': {}}

    for target_name in TARGET_MAP.keys():
        print(f"\n{'#'*70}")
        print(f"# TARGET: {target_name.upper()}")
        print(f"{'#'*70}")

        # XGBoost with lag features
        xgb_result = train_xgboost_lag(df, target_name, feature_cols, lag_cols)
        if xgb_result:
            all_results['xgboost_lag'][target_name] = xgb_result

        # Hybrid Conv1D+MLP
        hybrid_result = train_hybrid_conv1d(df, target_name, feature_cols + time_features, SEQ_LENGTH)
        if hybrid_result:
            all_results['hybrid_conv'][target_name] = hybrid_result

    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Save results
    with open(MODEL_DIR / "comparison_results.json", "w") as f:
        json.dump(convert_to_native(all_results), f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Model Comparison")
    print("=" * 70)
    print(f"\n{'Target':<12} | {'Model':<20} | {'MAE':>8} | {'R²':>8} | {'MAE Impr':>10}")
    print("-" * 70)

    for target_name in TARGET_MAP.keys():
        for model_type in ['xgboost_lag', 'hybrid_conv']:
            if target_name in all_results[model_type]:
                r = all_results[model_type][target_name]
                model_name = r['model']
                print(f"{target_name:<12} | {model_name:<20} | {r['mae']:>8.4f} | {r['r2']:>8.4f} | {r['improvement_mae_pct']:>9.1f}%")
        print("-" * 70)

    print(f"\nResults saved to: {MODEL_DIR / 'comparison_results.json'}")


if __name__ == "__main__":
    main()
