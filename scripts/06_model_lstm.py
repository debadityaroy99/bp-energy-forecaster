"""
06_model_lstm.py
----------------
Stage 6 of the forecasting pipeline: train a lightweight LSTM (Long Short-Term
Memory) neural network and compare it against the statistical baselines.

Why LSTM for time series?
  Standard feedforward networks treat every input independently. An LSTM has a
  "cell state" — a memory highway that can carry information across many time
  steps. This lets it learn patterns like "consumption grew for 3 decades then
  plateaued", which a single linear coefficient cannot represent.

Sliding-window approach (look_back = 3):
  The sequence [y_t, y_{t+1}, y_{t+2}] is used as input features to predict
  y_{t+3}.  With look_back=3 applied to the 9-point test set this produces 6
  evaluable predictions (for test years 4-9, i.e. 2019-2024).

Run from the `scripts/` directory:
    python 06_model_lstm.py
"""

import os

# Suppress TensorFlow's verbose C++ / oneDNN startup messages before import.
os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]       = "0"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# 1. Paths & baseline scores
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

TRAIN_CSV  = os.path.join(DATA_DIR, "train_data.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test_data.csv")
PLOT_PATH  = os.path.join(DATA_DIR, "lstm_forecast.png")

BASELINE_MAE  = 402_485.43
BASELINE_RMSE = 526_577.50

LOOK_BACK  = 3   # number of past years fed as input features
EPOCHS     = 100
BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
print("[1/6] Loading data ...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"      Train : {train_df.shape[0]} rows  "
      f"({train_df['Year'].min()} - {train_df['Year'].max()})")
print(f"      Test  : {test_df.shape[0]} rows  "
      f"({test_df['Year'].min()} - {test_df['Year'].max()})")

# ---------------------------------------------------------------------------
# 3. Scale to [0, 1]  --  fit ONLY on training data to prevent data leakage
# ---------------------------------------------------------------------------
print("[2/6] Scaling data ...")

# Scaler expects shape (n, 1); reshape from (n,) accordingly.
train_values = train_df["Oil_Consumption_m3_day"].values.reshape(-1, 1)
test_values  = test_df["Oil_Consumption_m3_day"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))

# fit_transform on train: learns the min/max from training data only.
# transform on test: applies the SAME scale — never refit on test data.
train_scaled = scaler.fit_transform(train_values)
test_scaled  = scaler.transform(test_values)

# ---------------------------------------------------------------------------
# 4. Sliding-window sequence builder
# ---------------------------------------------------------------------------
def create_dataset(dataset, look_back=3):
    """
    Convert a 1-D scaled array into supervised (X, y) pairs.

    For every position i where a full window fits:
        X[i] = dataset[i : i + look_back]        (input sequence)
        y[i] = dataset[i + look_back]             (next-step target)

    Returns plain numpy arrays (still scaled).
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

print("[3/6] Creating sliding-window sequences ...")

X_train, y_train = create_dataset(train_scaled, LOOK_BACK)
X_test,  y_test  = create_dataset(test_scaled,  LOOK_BACK)

# Keras LSTM requires 3-D input: [samples, time_steps, features].
# We have one feature (consumption), so the last dimension is 1.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],   X_test.shape[1],  1)

print(f"      X_train shape : {X_train.shape}   y_train shape : {y_train.shape}")
print(f"      X_test  shape : {X_test.shape}    y_test  shape : {y_test.shape}")

# ---------------------------------------------------------------------------
# 5. Build the LSTM model
# ---------------------------------------------------------------------------
print("[4/6] Building LSTM model ...")

model = Sequential([
    # 50 LSTM units learn temporal dependencies across the look_back window.
    # relu avoids the vanishing-gradient problem common with tanh on deep nets,
    # and works well for monotonic trends like energy consumption.
    LSTM(50, activation="relu", input_shape=(LOOK_BACK, 1)),

    # Single Dense neuron: produce one scalar forecast per sequence.
    Dense(1),
])

# Adam adapts the learning rate per-parameter — the standard choice for LSTMs.
# MSE loss penalises large deviations heavily, matching our RMSE evaluation.
model.compile(optimizer="adam", loss="mse")
model.summary()

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
print(f"\n[5/6] Training for up to {EPOCHS} epochs (batch_size={BATCH_SIZE}) ...")

# EarlyStopping watches the training loss and halts if it stops improving for
# 15 epochs, then restores the weights from the best epoch.  This prevents
# overfitting on the small training set without requiring manual epoch tuning.
early_stop = EarlyStopping(
    monitor="loss",
    patience=15,
    restore_best_weights=True,
    verbose=0,
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)

stopped_at = len(history.history["loss"])
print(f"\n      Training finished at epoch {stopped_at}  "
      f"(final loss: {history.history['loss'][-1]:.6f})")

# ---------------------------------------------------------------------------
# 7. Predict & inverse-scale
# ---------------------------------------------------------------------------
print("[6/6] Predicting and evaluating ...")

# model.predict() returns scaled values in [0, 1].
# inverse_transform() maps them back to real m3/day units.
test_preds_scaled = model.predict(X_test, verbose=0)

# inverse_transform expects shape (n, 1).
lstm_preds = scaler.inverse_transform(test_preds_scaled).flatten()

# y_test is a 1-D array of scaled values; reshape before inverse transform.
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# ---------------------------------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------------------------------
mae  = mean_absolute_error(y_test_real, lstm_preds)
rmse = np.sqrt(mean_squared_error(y_test_real, lstm_preds))

print("\n--- LSTM Model Evaluation ---")
print(f"  MAE  : {mae:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_MAE:>15,.2f}  |  "
      f"delta: {BASELINE_MAE - mae:+,.2f})")
print(f"  RMSE : {rmse:>15,.2f} m3/day   "
      f"(baseline: {BASELINE_RMSE:>15,.2f}  |  "
      f"delta: {BASELINE_RMSE - rmse:+,.2f})")
print("-----------------------------\n")

if mae < BASELINE_MAE and rmse < BASELINE_RMSE:
    print("  [PASS] LSTM beats the Linear Regression baseline on both metrics.")
elif mae < BASELINE_MAE or rmse < BASELINE_RMSE:
    print("  [PARTIAL] LSTM beats the baseline on one metric.")
else:
    print("  [INFO] LSTM did not improve over baseline. "
          "Consider increasing EPOCHS, tuning LOOK_BACK, or adding a second LSTM layer.")

# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------
# With look_back=3, the first prediction corresponds to test index 3 (year 4
# of the test window, i.e. 2019).  Slice the test years to match exactly.
pred_years = test_df["Year"].values[LOOK_BACK:]   # 2019 … 2024  (6 points)

fig, ax = plt.subplots(figsize=(12, 6))

# Full training history for context.
ax.plot(
    train_df["Year"], train_df["Oil_Consumption_m3_day"],
    color="steelblue", linewidth=2,
    label="Train Data (1965-2015)"
)

# All 9 actual test points (unsliced) so the viewer sees the full test window.
ax.plot(
    test_df["Year"], test_df["Oil_Consumption_m3_day"],
    color="darkorange", linewidth=2,
    label="Test Data Actual (2016-2024)"
)

# LSTM predictions aligned to the years they correspond to (2019-2024).
ax.plot(
    pred_years, lstm_preds,
    color="teal", linewidth=2, linestyle="--",
    label=f"LSTM Prediction (look_back={LOOK_BACK})"
)

# Shade the region where LSTM has no predictions (2016-2018) to make the
# look_back gap visually explicit — a common detail reviewers will ask about.
ax.axvspan(
    test_df["Year"].iloc[0], test_df["Year"].iloc[LOOK_BACK - 1] + 0.5,
    alpha=0.07, color="grey",
    label=f"Look-back warm-up ({LOOK_BACK} yrs)"
)

# Split boundary.
ax.axvline(x=2015.5, color="grey", linestyle=":", linewidth=1.2, alpha=0.6)
ax.text(2015.6, ax.get_ylim()[0], " Split (2015)",
        color="grey", fontsize=8, va="bottom")

ax.set_title("LSTM Deep Learning Forecast vs Actual", fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Oil Consumption (m3/day)", fontsize=11)
ax.legend(fontsize=9.5)
ax.grid(True, linestyle="--", alpha=0.4)

# Metric annotation box.
metric_text = (
    f"LSTM     MAE  = {mae/1e6:.3f}M\n"
    f"LSTM     RMSE = {rmse/1e6:.3f}M\n"
    f"Baseline MAE  = {BASELINE_MAE/1e6:.3f}M\n"
    f"Baseline RMSE = {BASELINE_RMSE/1e6:.3f}M\n"
    f"Epochs run    = {stopped_at}"
)
ax.text(
    0.02, 0.97, metric_text,
    transform=ax.transAxes,
    fontsize=8.5, verticalalignment="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e0f7f7", alpha=0.9)
)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"      Plot saved -> {PLOT_PATH}")
print("\n[OK] LSTM script complete.")
