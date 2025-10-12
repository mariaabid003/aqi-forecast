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
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 🔐 Connect to Hopsworks
# -----------------------------
print("🔐 Connecting to Hopsworks...")
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key!")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# -----------------------------
# 📥 Load data from Feature Store
# -----------------------------
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
print("✅ Data loaded! Dataset shape:", df.shape)

# -----------------------------
# 🧹 Data Cleaning
# -----------------------------
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
    raise ValueError("🚨 Dataset is empty after cleaning!")

# -----------------------------
# 🔁 Create sequences for LSTM
# -----------------------------
SEQUENCE_LENGTH = 7  # past 7 days to predict next day
X_seq, y_seq = [], []
for i in range(SEQUENCE_LENGTH, len(df)):
    X_seq.append(df[feature_cols].iloc[i-SEQUENCE_LENGTH:i].values)
    y_seq.append(df[TARGET].iloc[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print("✅ Sequence shape:", X_seq.shape, y_seq.shape)

# -----------------------------
# ⚖️ Feature Scaling
# -----------------------------
scaler_X = StandardScaler()
nsamples, ntimesteps, nfeatures = X_seq.shape
X_reshaped = X_seq.reshape((nsamples * ntimesteps, nfeatures))
X_scaled_flat = scaler_X.fit_transform(X_reshaped)
X_scaled = X_scaled_flat.reshape((nsamples, ntimesteps, nfeatures))
joblib.dump(scaler_X, "models/tf_scaler.joblib")
print("✅ Feature scaler saved to models/tf_scaler.joblib")

# -----------------------------
# ⚖️ Target Scaling
# -----------------------------
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, 1))
joblib.dump(scaler_y, "models/tf_y_scaler.joblib")
print("✅ Target scaler saved to models/tf_y_scaler.joblib")

# -----------------------------
# 🧱 Build LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, activation="tanh", input_shape=(SEQUENCE_LENGTH, len(feature_cols)), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation="tanh"),
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# -----------------------------
# 🚀 Train Model
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(
    X_scaled, y_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[early_stop],
    shuffle=True,
    verbose=1
)

# -----------------------------
# 💾 Save Full Model
# -----------------------------
MODEL_PATH = "models/tf_lstm_model.keras"
model.save(MODEL_PATH)
print(f"✅ LSTM model saved to {MODEL_PATH}")

# -----------------------------
# ☁️ Upload Model to Hopsworks Model Registry
# -----------------------------
mr = project.get_model_registry()
timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
model_meta = mr.python.create_model(
    name="tf_lstm_aqi_model",
    metrics={"val_loss": float(history.history['val_loss'][-1])},
    description=f"LSTM AQI forecasting model trained on {timestamp}"
)
model_meta.save(MODEL_PATH)
print("🚀 Model uploaded to Hopsworks Model Registry!")
print("🏁 TensorFlow LSTM training completed successfully!")
