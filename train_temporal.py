#!/usr/bin/env python3
"""
Train 1D Convolutional Neural Network for temporal weather debiasing.
Uses past N hours of forecast data to predict current observation.
"""

import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
SEQ_DIR = Path(__file__).parent / "sequences"
MODEL_DIR = Path(__file__).parent / "models_v3"
MODEL_DIR.mkdir(exist_ok=True)

TARGETS = ['temperature', 'humidity', 'rain', 'wind_speed', 'gust_speed']

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15  # Early stopping patience


class WeatherConv1D(nn.Module):
    """1D Convolutional model for temporal weather prediction."""

    def __init__(self, n_features, seq_length, hidden_dim=64):
        super().__init__()

        # Conv layers: capture temporal patterns
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Pooling and dense layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [batch, seq_length, n_features]
        x = x.permute(0, 2, 1)  # [batch, n_features, seq_length]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)  # [batch, 128]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x.squeeze(-1)


def train_model(target_name, X_train, y_train, X_test, y_test, n_features, seq_length):
    """Train a Conv1D model for a specific target."""

    print(f"\n{'='*60}")
    print(f"Training Conv1D for: {target_name}")
    print(f"{'='*60}")

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create model
    model = WeatherConv1D(n_features, seq_length, hidden_dim=64)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.extend(y_pred.cpu().numpy())
            all_true.extend(y_batch.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_true)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\nFinal Results:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Save model
    model_path = MODEL_DIR / f"conv1d_{target_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': n_features,
        'seq_length': seq_length,
        'hidden_dim': 64
    }, model_path)
    print(f"Model saved to: {model_path}")

    return {
        'target': target_name,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'n_params': n_params,
        'epochs_trained': epoch + 1
    }


def main():
    print("=" * 60)
    print("TRAIN 1D CONVOLUTIONAL MODELS")
    print("=" * 60)

    # Load metadata
    with open(SEQ_DIR / "metadata.json") as f:
        metadata = json.load(f)

    n_features = metadata['n_features']
    seq_length = metadata['seq_length']
    features = metadata['features']

    print(f"\nSequence length: {seq_length}")
    print(f"Number of features: {n_features}")

    all_results = {}

    for target_name in TARGETS:
        seq_file = SEQ_DIR / f"sequences_{target_name}.npz"

        if not seq_file.exists():
            print(f"\nSkipping {target_name}: no sequence file found")
            continue

        # Load sequences
        data = np.load(seq_file)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        print(f"\nLoaded {target_name}:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")

        # Train model
        result = train_model(
            target_name, X_train, y_train, X_test, y_test,
            n_features, seq_length
        )
        all_results[target_name] = result

    # Save config and metrics
    config = {
        'model_type': 'Conv1D',
        'seq_length': seq_length,
        'n_features': n_features,
        'features': features,
        'targets': TARGETS,
        'architecture': {
            'conv_filters': [32, 64, 128],
            'kernel_size': 3,
            'hidden_dim': 64,
            'dropout': 0.3
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'max_epochs': EPOCHS,
            'early_stopping_patience': PATIENCE
        }
    }

    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Copy normalization stats
    import shutil
    shutil.copy(SEQ_DIR / "norm_stats.json", MODEL_DIR / "norm_stats.json")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  MAE:  {result['mae']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  R²:   {result['r2']:.4f}")

    print(f"\nModels saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
