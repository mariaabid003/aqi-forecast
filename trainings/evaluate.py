import os
import pandas as pd
import numpy as np
import hopsworks
import joblib
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()  # ✅ Loads variables from .env file into environment

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_COL = "aqi_aqicn"
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_DIR = "models"

# ============================================================
# CONNECT TO HOPSWORKS (Non-interactive)
# ============================================================
print("🔐 Connecting to Hopsworks...")

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY") or os.getenv("aqi_forecast_api_key")
if not HOPSWORKS_API_KEY:
    raise ValueError("❌ Missing Hopsworks API key! Please set it in your .env file or as an environment variable.")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# ============================================================
# LOAD DATA FROM FEATURE GROUP
# ============================================================
try:
    feature_group = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
except Exception as e:
    raise ValueError(f"🚨 Feature group '{FEATURE_GROUP_NAME}' not found: {e}")

df = feature_group.read()

# Clean data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
df = df.dropna(subset=[TARGET_COL])

if TARGET_COL not in df.columns:
    raise ValueError(f"🚨 Target column '{TARGET_COL}' not found in dataset.")

# Split into features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

print("✅ Data loaded for evaluation!")
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# ============================================================
# DEFINE EVALUATION FUNCTION
# ============================================================
def evaluate_model(model_type, model_file, scaler_file):
    """Evaluate a given model and return metrics."""
    if not os.path.exists(model_file):
        print(f"🚨 {model_type} model not found at {model_file}")
        return None

    if not os.path.exists(scaler_file):
        print(f"🚨 {model_type} scaler not found at {scaler_file}")
        return None

    print(f"\n🔍 Evaluating {model_type} model...")

    # Load scaler and transform input data
    scaler = joblib.load(scaler_file)
    X_scaled = scaler.transform(X)

    # Load and predict using the model
    if model_type == "sklearn":
        model = joblib.load(model_file)
        y_pred = model.predict(X_scaled)
    elif model_type == "tensorflow":
        model = tf.keras.models.load_model(model_file)
        y_pred = model.predict(X_scaled).flatten()
    else:
        raise ValueError("Invalid model_type. Must be 'sklearn' or 'tensorflow'.")

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"✅ {model_type} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    return {"Model": model_type, "RMSE": rmse, "MAE": mae, "R²": r2}

# ============================================================
# EVALUATE BOTH MODELS
# ============================================================
results = []

# Sklearn (Random Forest)
sklearn_model = os.path.join(MODEL_DIR, "rf_model.joblib")
sklearn_scaler = os.path.join(MODEL_DIR, "rf_scaler.joblib")
sklearn_metrics = evaluate_model("sklearn", sklearn_model, sklearn_scaler)
if sklearn_metrics:
    results.append(sklearn_metrics)

# TensorFlow Model
tf_model = os.path.join(MODEL_DIR, "tf_model.keras")
tf_scaler = os.path.join(MODEL_DIR, "tf_scaler.joblib")
tf_metrics = evaluate_model("tensorflow", tf_model, tf_scaler)
if tf_metrics:
    results.append(tf_metrics)

# ============================================================
# DISPLAY RESULTS
# ============================================================
if results:
    results_df = pd.DataFrame(results)
    print("\n📊 Model Evaluation Comparison:")
    print(results_df.to_string(index=False))
else:
    print("🚨 No models were successfully evaluated!")

print("\n🏁 Evaluation completed successfully!")
