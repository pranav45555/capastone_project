"""
CollideX -- Step 7.3: LSTM Trajectory Prediction Model
===========================================================
Architecture:
  Input  : sequence of 3 orbital state vectors [h1, h6, h12]
             each vector = (x, y, z, vx, vy, vz) -> shape (3, 6)
  Output : predicted state at h24 -> shape (6,)

Model layers:
  LSTM(64, return_sequences=True)  -> captures short-term trajectory
  Dropout(0.2)
  LSTM(32)                         -> refines temporal pattern
  Dropout(0.1)
  Dense(16, relu)                  -> non-linear projection
  Dense(6)                         -> position + velocity prediction

Saved outputs:
  CollideX/models/lstm_trajectory_model.keras
  CollideX/models/lstm_trajectory_model.h5   (legacy format)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, "..")
DATASET_FILE= os.path.join(ROOT_DIR, "data", "processed", "lstm_dataset.csv")
MODEL_DIR   = os.path.join(ROOT_DIR, "models")
MODEL_KERAS = os.path.join(MODEL_DIR, "lstm_trajectory_model.keras")
MODEL_H5    = os.path.join(MODEL_DIR, "lstm_trajectory_model.h5")

SEQ_LEN     = 3      # input time steps
N_FEATURES  = 6      # x, y, z, vx, vy, vz
HORIZONS    = [1, 6, 12]

# ===========================================================================
def load_sequences(path: str):
    """Load lstm_dataset.csv and reshape into (N, SEQ_LEN, N_FEATURES)."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} satellite sequences")

    X_cols = []
    for h in HORIZONS:
        for feat in ["future_x_km", "future_y_km", "future_z_km",
                     "vel_x_km_s",  "vel_y_km_s",  "vel_z_km_s"]:
            X_cols.append(f"X_h{h}_{feat}")

    y_cols = [f"y_{f}" for f in
              ["future_x_km", "future_y_km", "future_z_km",
               "vel_x_km_s",  "vel_y_km_s",  "vel_z_km_s"]]

    X_flat = df[X_cols].values                          # (N, 18)
    X      = X_flat.reshape(-1, SEQ_LEN, N_FEATURES)   # (N, 3, 6)
    y      = df[y_cols].values                          # (N, 6)

    return X, y


def build_model(seq_len: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(n_features),          # predict all 6 state variables
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Per-variable breakdown
    feat_names = ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
    print("\n  Per-variable MAE (normalized units):")
    for i, name in enumerate(feat_names):
        print(f"    {name:10s} : {mean_absolute_error(y_test[:, i], y_pred[:, i]):.6f}")

    return rmse, mae, y_pred


# ===========================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("  CollideX -- Step 7.3: LSTM Trajectory Training")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # Step 7.3.1 -- Load data
    # -----------------------------------------------------------------------
    print("\n[7.3.1] Loading LSTM dataset ...")
    X, y = load_sequences(DATASET_FILE)
    print(f"  X shape : {X.shape}  (samples, seq_len, features)")
    print(f"  y shape : {y.shape}  (samples, 6 output features)")

    # -----------------------------------------------------------------------
    # Step 7.3.2 -- Train / val / test split
    # -----------------------------------------------------------------------
    print("\n[7.3.2] Splitting data (70/15/15) ...")
    n        = len(X)
    n_train  = int(n * 0.70)
    n_val    = int(n * 0.15)

    X_train, y_train = X[:n_train],         y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],   y[n_train+n_val:]

    print(f"  Train : {len(X_train):,}  |  Val : {len(X_val):,}  "
          f"|  Test : {len(X_test):,}")

    # -----------------------------------------------------------------------
    # Step 7.3.3 -- Build model
    # -----------------------------------------------------------------------
    print("\n[7.3.3] Building LSTM model ...")
    model = build_model(SEQ_LEN, N_FEATURES)
    model.summary()

    # -----------------------------------------------------------------------
    # Step 7.3.4 -- Train
    # -----------------------------------------------------------------------
    print("\n[7.3.4] Training LSTM model ...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Training summary
    best_epoch  = np.argmin(history.history["val_loss"]) + 1
    best_val    = min(history.history["val_loss"])
    final_train = history.history["loss"][-1]
    print(f"\n  Best epoch      : {best_epoch}")
    print(f"  Best val_loss   : {best_val:.6f}")
    print(f"  Final train_loss: {final_train:.6f}")

    # -----------------------------------------------------------------------
    # Step 7.3.5 -- Evaluate on test set
    # -----------------------------------------------------------------------
    print("\n[7.3.5] Evaluating on test set ...")
    rmse, mae, y_pred = evaluate(model, X_test, y_test)
    print(f"\n  RMSE (normalized) : {rmse:.6f}")
    print(f"  MAE  (normalized) : {mae:.6f}")

    # -----------------------------------------------------------------------
    # Step 7.3.6 -- Save outputs
    # -----------------------------------------------------------------------
    print("\n[7.3.6] Saving model ...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Native Keras format (recommended)
    model.save(MODEL_KERAS)
    size_mb = os.path.getsize(MODEL_KERAS) / 1_048_576
    print(f"  Saved -> {MODEL_KERAS}  ({size_mb:.2f} MB)")

    # Legacy .h5 format for compatibility
    model.save(MODEL_H5)
    size_mb_h5 = os.path.getsize(MODEL_H5) / 1_048_576
    print(f"  Saved -> {MODEL_H5}  ({size_mb_h5:.2f} MB)")

    # Save test predictions for downstream fusion
    pred_df = pd.DataFrame(
        y_pred,
        columns=["pred_x_km", "pred_y_km", "pred_z_km",
                 "pred_vx_km_s", "pred_vy_km_s", "pred_vz_km_s"]
    )
    pred_file = os.path.join(
        ROOT_DIR, "data", "processed", "lstm_predictions.csv"
    )
    pred_df.to_csv(pred_file, index=False)
    print(f"  Saved -> {pred_file}")

    print("\n" + "=" * 62)
    print("  STEP 7.3 COMPLETE -- LSTM Trajectory Model Trained")
    print("=" * 62)
    print(f"  Architecture : LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(6)")
    print(f"  Input shape  : ({SEQ_LEN}, {N_FEATURES})  [h1, h6, h12 states]")
    print(f"  Output shape : (6,)  [predicted h24 state]")
    print(f"  Test RMSE    : {rmse:.6f}  (normalized)")
    print(f"  Test MAE     : {mae:.6f}  (normalized)")
    print()
