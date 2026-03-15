# Plan: Temporal Model with 1D Convolution

## Current State

**Model**: XGBoost (point-in-time features only)
- 36 features: 30 forecast variables + 6 cyclical time features
- Each hour predicted independently - no temporal context
- Rain prediction poor (R² = 0.196) because weather patterns are inherently temporal

**Limitations**:
- Cannot capture approaching weather systems
- Cannot see trends (pressure dropping, clouds building)
- Cannot learn "precipitation continues if it was raining before"

## Proposed Approach

### Architecture: 1D Convolutional Neural Network

Use past N hours of forecast data to predict the current hour's debiased value.

```
Input: [batch, seq_length, n_features]  e.g., [32, 12, 36]
                    ↓
         1D Convolution layers
                    ↓
         Global pooling / flatten
                    ↓
         Dense layers
                    ↓
Output: [batch, 1]  (single target value)
```

**Why 1D CNN over LSTM?**
- Faster training and inference
- Captures local temporal patterns effectively
- Less prone to overfitting with limited data (~5000 samples)
- Easier to deploy (no hidden state management)

### Hyperparameters to tune:
- **Sequence length**: 6, 12, or 24 hours of history
- **Conv layers**: 2-3 layers with increasing filters (32→64→128)
- **Kernel size**: 3 or 5 (captures 3-5 hour patterns)
- **Dropout**: 0.2-0.3 for regularization

## Implementation Steps

### Step 1: Prepare Sequence Data (`prepare_sequences.py`)

1. Load `training_data_v2.csv`
2. For each timestamp t, create a sequence: [t-seq_len, ..., t-1, t]
3. Target is observation at time t
4. Handle missing values (interpolation or masking)
5. Save as `.npz` or `.pt` files

```python
# Pseudo-code
def create_sequences(df, seq_length=12):
    sequences = []
    targets = []
    for i in range(seq_length, len(df)):
        seq = df.iloc[i-seq_length:i][feature_cols].values
        target = df.iloc[i][target_col]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
```

### Step 2: Define Model Architecture (`models/conv1d_model.py`)

```python
import torch
import torch.nn as nn

class WeatherConv1D(nn.Module):
    def __init__(self, n_features, seq_length, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [batch, seq_length, n_features]
        x = x.permute(0, 2, 1)  # [batch, n_features, seq_length]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [batch, 128]
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze(-1)
```

### Step 3: Training Script (`train_temporal.py`)

1. Load sequence data
2. Daily hold-out split (same as current)
3. Create DataLoader with batching
4. Train with Adam optimizer, MSE loss
5. Early stopping based on validation loss
6. Save model checkpoints

### Step 4: Update Inference (`generate_forecast.py`)

1. Maintain a rolling buffer of past N hours
2. Load both XGBoost (fallback) and Conv1D models
3. For each forecast hour:
   - Build sequence from past hours + current forecast
   - Run through Conv1D model
4. Handle edge cases (first N hours use XGBoost)

### Step 5: Hybrid Approach (Optional)

Combine XGBoost + Conv1D predictions:
- Conv1D: Better for rain, wind (temporal patterns)
- XGBoost: Better for temperature (strong point correlation)
- Weighted ensemble based on validation performance

## File Structure

```
models_v3/
├── conv1d_temperature.pt
├── conv1d_humidity.pt
├── conv1d_rain.pt
├── conv1d_wind_speed.pt
├── conv1d_gust_speed.pt
├── config.json
└── metrics.json

scripts/
├── prepare_sequences.py
├── train_temporal.py
└── evaluate_temporal.py
```

## Expected Improvements

| Target | Current R² | Expected R² | Reason |
|--------|------------|-------------|--------|
| Temperature | 0.971 | 0.975 | Already good, minor gains |
| Humidity | 0.865 | 0.90 | Humidity follows diurnal patterns |
| Rain | 0.196 | 0.40-0.50 | **Major improvement** - precipitation is highly temporal |
| Wind Speed | 0.602 | 0.70 | Wind patterns are temporal |
| Gust Speed | 0.465 | 0.55 | Gusts often follow sustained wind |

## Risks and Mitigations

1. **Overfitting with limited data (~5000 samples)**
   - Use aggressive dropout (0.3)
   - Use data augmentation (small noise)
   - Keep model small (128 params max)

2. **Inference complexity**
   - Need to maintain history buffer
   - Fallback to XGBoost for missing history

3. **Training time**
   - PyTorch is fast on M1/M2 Macs
   - ~5-10 minutes per model expected

## Dependencies

```
torch>=2.0
numpy
pandas
scikit-learn
```

## Next Steps

1. [ ] Implement `prepare_sequences.py`
2. [ ] Implement `conv1d_model.py`
3. [ ] Implement `train_temporal.py`
4. [ ] Train models and evaluate
5. [ ] Update `generate_forecast.py` for inference
6. [ ] A/B test vs current XGBoost
