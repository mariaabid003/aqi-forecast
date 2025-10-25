import os
import time
import pandas as pd
import numpy as np
import joblib
import hopsworks
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import shutil
import logging

# ============================================================
# 🪶 SETUP LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================
# 🔐 CONNECT TO HOPSWORKS
# ============================================================
log.info("🔐 Connecting to Hopsworks...")
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

# ============================================================
# 📥 LOAD DATA FROM FEATURE STORE (with retry + fallback)
# ============================================================
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
log.info(f"✅ Data shape: {df.shape}")

# ============================================================
# 🧹 CLEAN DATA
# ============================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

TARGET = "aqi_aqicn"
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=[TARGET] + feature_cols)
if df.empty:
    raise ValueError("🚨 Dataset empty after cleaning!")

# ============================================================
# 🔁 CREATE SEQUENCES FOR LSTM
# ============================================================
SEQUENCE_LENGTH = 7
X_seq, y_seq = [], []
for i in range(SEQUENCE_LENGTH, len(df)):
    X_seq.append(df[feature_cols].iloc[i - SEQUENCE_LENGTH:i].values)
    y_seq.append(df[TARGET].iloc[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
log.info(f"✅ Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

# ============================================================
# ⚖️ SCALING
# ============================================================
os.makedirs("models", exist_ok=True)
scaler_X = StandardScaler()
nsamples, ntimesteps, nfeatures = X_seq.shape
X_scaled_flat = scaler_X.fit_transform(X_seq.reshape(nsamples * ntimesteps, nfeatures))
X_scaled = X_scaled_flat.reshape(nsamples, ntimesteps, nfeatures)
joblib.dump(scaler_X, "models/tf_scaler.joblib")

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, 1))
joblib.dump(scaler_y, "models/tf_y_scaler.joblib")
log.info("✅ Scalers saved!")

# ============================================================
# 🧱 BUILD LSTM MODEL
# ============================================================
model = Sequential([
    LSTM(128, activation="tanh", input_shape=(SEQUENCE_LENGTH, len(feature_cols)), return_sequences=True),
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

# ============================================================
# 🚀 TRAIN MODEL
# ============================================================
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

log.info(f"✅ Training done: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

# ============================================================
# 💾 SAVE MODEL LOCALLY
# ============================================================
MODEL_PATH = "models/tf_lstm_model.keras"
model.save(MODEL_PATH)
log.info(f"✅ TensorFlow LSTM model saved → {MODEL_PATH}")

# ============================================================
# ☁️ UPLOAD MODEL TO HOPSWORKS MODEL REGISTRY
# ============================================================
timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
model_dir = os.path.join("models", "tf_lstm_artifact")
os.makedirs(model_dir, exist_ok=True)

# Copy artifacts
for file in ["tf_lstm_model.keras", "tf_scaler.joblib", "tf_y_scaler.joblib"]:
    shutil.copy(f"models/{file}", os.path.join(model_dir, file))

model_meta = mr.python.create_model(
    name="tf_lstm_aqi_model_final",
    metrics={"train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae},
    description=f"Final LSTM AQI forecasting model trained on {timestamp}"
)
model_meta.save(model_dir)

log.info("🚀 Uploaded tf_lstm_aqi_model_final to Hopsworks Model Registry.")
log.info(f"📂 Included files: {os.listdir(model_dir)}")
log.info(f"📈 Metrics: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")
log.info("🏁 TensorFlow LSTM training & upload completed successfully!")
