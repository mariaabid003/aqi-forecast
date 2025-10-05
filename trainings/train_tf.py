import os
import time
import pandas as pd
import numpy as np
import joblib
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ============================================================
# 🔐 CONNECT TO HOPSWORKS
# ============================================================
print("🔐 Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# ============================================================
# 🧠 LOAD OR CREATE FEATURE VIEW
# ============================================================
feature_view_name = "aqi_features"
version = 1

try:
    feature_view = fs.get_feature_view(name=feature_view_name, version=version)
    print("✅ Feature view found!")
except Exception:
    print("⚠️ Feature view not found — creating a new one.")
    try:
        feature_group = fs.get_feature_group(name="aqi_features", version=1)
    except Exception as e:
        raise ValueError(f"🚨 Feature group 'aqi_features' not found in Hopsworks. Run backfill first! ({e})")

    query = feature_group.select_all()
    feature_view = fs.create_feature_view(
        name=feature_view_name,
        version=version,
        description="AQI features for training",
        query=query
    )
    print("✅ Feature view created successfully!")

# ============================================================
# 📥 FETCH DATA
# ============================================================
df = feature_view.get_batch_data()
print("✅ Data loaded from Hopsworks Feature Store!")
print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# 🧹 DATA CLEANING
# ============================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

TARGET = "aqi_aqicn"
non_numeric_cols = ["timestamp_utc"]
df = df.drop(columns=non_numeric_cols, errors="ignore")

if TARGET not in df.columns:
    raise ValueError(f"🚨 Target column '{TARGET}' not found in dataset.")

df = df.dropna(subset=[TARGET])
if df.empty:
    raise ValueError("🚨 Dataset is empty after cleaning. Please check feature group data.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

FEATURES = X.columns.tolist()
print("Training features:", FEATURES)
print(f"Total samples: {len(df)}")

if len(df) < 10:
    raise ValueError("🚨 Not enough data for training. Please backfill more samples!")

# ============================================================
# ✂️ TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# ⚖️ FEATURE SCALING
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
SCALER_FILE = os.path.join(MODEL_DIR, "tf_scaler.joblib")
joblib.dump(scaler, SCALER_FILE)
print(f"✅ Scaler saved to {SCALER_FILE}")

# ============================================================
# 🧱 BUILD TENSORFLOW MODEL
# ============================================================
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ============================================================
# 🚀 TRAIN MODEL
# ============================================================
print("🚀 Training TensorFlow model...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

# ============================================================
# 📊 EVALUATE MODEL
# ============================================================
y_pred = model.predict(X_test_scaled).flatten()
mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean(np.abs(y_test - y_pred))
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print("\n📈 TensorFlow Model Evaluation:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ============================================================
# 💾 SAVE MODEL LOCALLY
# ============================================================
MODEL_FILE = os.path.join(MODEL_DIR, "tf_model.keras")
model.save(MODEL_FILE)
print(f"✅ TensorFlow model saved to {MODEL_FILE}")

# ============================================================
# ☁️ UPLOAD TO HOPSWORKS MODEL REGISTRY (AUTO VERSION)
# ============================================================
mr = project.get_model_registry()

# Create a unique model name using timestamp
unique_id = int(time.time())
model_name = f"tf_aqi_model_{unique_id}"
print(f"🧩 Uploading as {model_name} ...")

# Create metadata
model_meta = mr.python.create_model(
    name=model_name,
    metrics={"MSE": mse, "MAE": mae, "R2": r2},
    description=f"TensorFlow AQI forecasting model (auto-uploaded {unique_id})"
)

# Upload both model + scaler folder
upload_dir = MODEL_DIR  # upload entire models directory
model_meta.save(upload_dir)

print(f"🚀 Model uploaded to Hopsworks successfully!")
print(f"Explore it here: {project.get_url()}/models/{model_meta.name}/{model_meta.version}")

print("\n🏁 Training completed successfully!")
