import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ----------------------------
# Paths
# ----------------------------
DATA_FILE = "data/features/training_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "tf_model.keras")  # Keras format
SCALER_FILE = os.path.join(MODEL_DIR, "tf_scaler.joblib")  # Save scaler

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)
df.ffill(inplace=True)  # Forward-fill missing values

# ----------------------------
# Features and Target
# ----------------------------
non_numeric_cols = ["timestamp_utc"]
TARGET = "aqi_aqicn"

df_aqi = df.drop(columns=non_numeric_cols)
X = df_aqi.drop(columns=[TARGET]).values
y = df_aqi[TARGET].values

FEATURES = df_aqi.drop(columns=[TARGET]).columns.tolist()
print("Training features:", FEATURES)

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Feature Scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for evaluation/serving
joblib.dump(scaler, SCALER_FILE)
print(f"✅ Scaler saved to {SCALER_FILE}")

# ----------------------------
# Build TensorFlow Model
# ----------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)  # Regression output
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ----------------------------
# Train Model
# ----------------------------
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    verbose=1
)

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mae = np.mean(np.abs(y_test - y_pred))

print("\nTensorFlow Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# ----------------------------
# Save Model
# ----------------------------
model.save(MODEL_FILE)
print(f"✅ TensorFlow model saved to {MODEL_FILE}")
