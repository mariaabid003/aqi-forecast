# train_lstm.py

import os
import time
import shutil
import joblib
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import hopsworks
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ──────────────────────────────────────────────────────────────
# 🪵 Logging Configuration
# ──────────────────────────────────────────────────────────────
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 🔐 Connect to Hopsworks
# ──────────────────────────────────────────────────────────────
log.info("🔐 Connecting to Hopsworks...")
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

# ──────────────────────────────────────────────────────────────
# 📥 Load Data from Feature Store
# ──────────────────────────────────────────────────────────────
def load_feature_data():
    fg = fs.get_feature_group(name="aqi_features", version=1)
    for attempt in range(3):
        try:
            df = fg.read()
            log.info("✅ Data loaded via Arrow Flight.")
            return df
        except Exception as e:
            log.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    log.info("🔁 Using fallback (pandas engine)...")
    return fg.read(read_options={"engine": "pandas"})

df = load_feature_data()
log.info(f"✅ Loaded data shape: {df.shape}")

# ──────────────────────────────────────────────────────────────
# 🧹 Data Preparation
# ──────────────────────────────────────────────────────────────
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

TARGET = "aqi_aqicn"
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=FEATURE_COLS + [TARGET])
if df.empty:
    raise ValueError("🚨 Dataset empty after cleaning!")

# ──────────────────────────────────────────────────────────────
# 🔁 Create Time-Series Sequences
# ──────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 7  # use last 7 observations for prediction
X_seq, y_seq = [], []
for i in range(SEQUENCE_LENGTH, len(df)):
    X_seq.append(df[FEATURE_COLS].iloc[i - SEQUENCE_LENGTH:i].values)
    y_seq.append(df[TARGET].iloc[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
log.info(f"✅ Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

# ──────────────────────────────────────────────────────────────
# ⚖️ Scaling
# ──────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# Scale input features
scaler_X = StandardScaler()
n_samples, n_timesteps, n_features = X_seq.shape
X_scaled_flat = scaler_X.fit_transform(X_seq.reshape(n_samples * n_timesteps, n_features))
X_scaled = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
joblib.dump(scaler_X, "models/tf_scaler_X.joblib")

# Scale target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, 1))
joblib.dump(scaler_y, "models/tf_scaler_y.joblib")

log.info("✅ Scalers saved successfully.")

# ──────────────────────────────────────────────────────────────
# 🧠 Build LSTM Model
# ──────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(128, activation="tanh", input_shape=(SEQUENCE_LENGTH, len(FEATURE_COLS)), return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation="tanh"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)
log.info("✅ LSTM model compiled successfully.")

# ──────────────────────────────────────────────────────────────
# 🚀 Train Model
# ──────────────────────────────────────────────────────────────
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

train_loss = float(history.history["loss"][-1])
val_loss = float(history.history["val_loss"][-1])
val_mae = float(history.history["mae"][-1])
log.info(f"✅ Training complete → Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

# ──────────────────────────────────────────────────────────────
# 💾 Save Model & Artifacts
# ──────────────────────────────────────────────────────────────
MODEL_DIR = "models/tf_lstm_model"
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(f"{MODEL_DIR}/model.keras")
joblib.dump(scaler_X, f"{MODEL_DIR}/scaler_X.joblib")
joblib.dump(scaler_y, f"{MODEL_DIR}/scaler_y.joblib")

metadata = {
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_mae": val_mae,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    import json
    json.dump(metadata, f, indent=4)

log.info("✅ Model and artifacts saved locally.")

# ──────────────────────────────────────────────────────────────
# ☁️ Upload Model to Hopsworks Registry
# ──────────────────────────────────────────────────────────────
artifact_dir = os.path.join("models", "tf_lstm_artifact")
os.makedirs(artifact_dir, exist_ok=True)
for file in os.listdir(MODEL_DIR):
    shutil.copy(os.path.join(MODEL_DIR, file), os.path.join(artifact_dir, file))

model_meta = mr.python.create_model(
    name="tf_lstm_aqi_model",
    metrics={
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mae": val_mae
    },
    description="LSTM model for AQI forecasting trained on weather + AQI data for Karachi."
)
model_meta.save(artifact_dir)

log.info("🚀 Uploaded tf_lstm_aqi_model to Hopsworks Model Registry.")
log.info(f"📈 Metrics: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")
log.info("🏁 LSTM training and deployment pipeline completed successfully!")
