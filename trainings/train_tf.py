import os
import time
import json
import shutil
import joblib
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import hopsworks
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("train_lstm")

# Hopsworks connection
load_dotenv()
api_key = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("Hopsworks API key not found.")

log.info("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
mr = project.get_model_registry()

def load_feature_data():
    fg = fs.get_feature_group(name="aqi_features", version=1)
    for attempt in range(3):
        try:
            df = fg.read()
            log.info("Data loaded successfully via Arrow Flight.")
            return df
        except Exception as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    log.info("Falling back to pandas engine for data load.")
    return fg.read(read_options={"engine": "pandas"})

df = load_feature_data()
log.info(f"Loaded dataset with shape {df.shape}")

# Data cleaning
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

target_col = "aqi_aqicn"
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=feature_cols + [target_col])

if df.empty:
    raise ValueError("No data available after cleaning.")

# Sequence generation
sequence_len = 7
X, y = [], []
for i in range(sequence_len, len(df)):
    X.append(df[feature_cols].iloc[i - sequence_len:i].values)
    y.append(df[target_col].iloc[i])

X, y = np.array(X), np.array(y)
log.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

# Scaling
os.makedirs("models", exist_ok=True)
scaler_X = StandardScaler()
n_samples, n_timesteps, n_features = X.shape
X_scaled = scaler_X.fit_transform(X.reshape(n_samples * n_timesteps, n_features)).reshape(n_samples, n_timesteps, n_features)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

joblib.dump(scaler_X, "models/tf_scaler_X.joblib")
joblib.dump(scaler_y, "models/tf_scaler_y.joblib")
log.info("Feature and target scalers saved.")

# Model setup
model = Sequential([
    LSTM(128, activation="tanh", return_sequences=True, input_shape=(sequence_len, len(feature_cols))),
    Dropout(0.3),
    LSTM(64, activation="tanh"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
log.info("Model compiled successfully.")

# Training
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

history = model.fit(
    X_scaled, y_scaled,
    validation_split=0.2,
    epochs=150,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    shuffle=True,
    verbose=1
)

# Evaluation
train_loss = float(history.history["loss"][-1])
val_loss = float(history.history["val_loss"][-1])
val_mae = float(history.history["mae"][-1])

y_pred = model.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y_scaled, y_pred))
mae = mean_absolute_error(y_scaled, y_pred)

log.info(
    f"Training completed. "
    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
    f"Val MAE={val_mae:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
)

# Save model and artifacts
model_dir = "models/tf_lstm_model"
os.makedirs(model_dir, exist_ok=True)
model.save(f"{model_dir}/model.keras")
joblib.dump(scaler_X, f"{model_dir}/scaler_X.joblib")
joblib.dump(scaler_y, f"{model_dir}/scaler_y.joblib")

metadata = {
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_mae": val_mae,
    "rmse": rmse,
    "mae": mae,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(f"{model_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

log.info("Model and training metadata saved locally.")

# Upload to Hopsworks
artifact_dir = os.path.join("models", "tf_lstm_artifact")
os.makedirs(artifact_dir, exist_ok=True)
for file in os.listdir(model_dir):
    shutil.copy(os.path.join(model_dir, file), os.path.join(artifact_dir, file))

model_meta = mr.python.create_model(
    name="tf_lstm_aqi_model",
    metrics={
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "rmse": rmse,
        "mae": mae
    },
    description="LSTM model for AQI forecasting trained on Karachi weather and AQI data."
)
model_meta.save(artifact_dir)

log.info("Model uploaded to Hopsworks successfully.")
log.info(f"Final metrics — RMSE: {rmse:.4f}, MAE: {mae:.4f}, Val Loss: {val_loss:.4f}")
