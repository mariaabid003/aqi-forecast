"""
🌸 AQI Forecast Dashboard — Light Pastel Theme (Improved)
--------------------------------------------------
Features:
✅ Today's Forecast + Next 3-Day Forecast
✅ SHAP Explainability (LSTM-safe, fallback only)
✅ Forecast Chart, Heatmap, Summary Stats
✅ Elegant pastel design (lavender, pink, blue)
"""

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# 🩵 Compatibility Fix (for SHAP + NumPy)
# -----------------------------
if not hasattr(np, "bool"):
    np.bool = bool

# -----------------------------
# 🌸 Streamlit Config & Styling
# -----------------------------
st.set_page_config(page_title="AQI Forecast Dashboard 🌤️", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e3f2fd, #fce4ec, #ede7f6);
        color: #333;
        font-family: "Poppins", sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        color: #6a1b9a;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #1565c0;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #ba68c8;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ab47bc;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🧩 Load Environment & Model
# -----------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")

st.title("🌤️ AQI Forecast Dashboard")
st.caption("Real-time Air Quality Prediction and Explainability")

try:
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    st.success("✅ Data successfully loaded from Hopsworks!")
except Exception as e:
    st.error(f"Error connecting to Hopsworks: {e}")
    st.stop()

TARGET = "aqi_aqicn"
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=[TARGET] + FEATURE_COLS)
if df.empty:
    st.error("Dataset is empty after cleaning!")
    st.stop()

# -----------------------------
# 🔮 Load Model & Scalers
# -----------------------------
MODEL_PATH = "models/tf_lstm_model.keras"
SCALER_X_PATH = "models/tf_scaler.joblib"
SCALER_Y_PATH = "models/tf_y_scaler.joblib"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    st.success("✅ Model and scalers loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scalers: {e}")
    st.stop()

# -----------------------------
# 🧮 Helper — AQI Category
# -----------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "🟢 Good"
    elif aqi <= 100:
        return "🟡 Moderate"
    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "🔴 Unhealthy"
    elif aqi <= 300:
        return "🟣 Very Unhealthy"
    else:
        return "⚫ Hazardous"

# -----------------------------
# 🔮 Generate Forecast (Today + Next 3 Days)
# -----------------------------
SEQUENCE_LENGTH = 7
latest_seq = df[FEATURE_COLS].iloc[-SEQUENCE_LENGTH:].values
nsamples, nfeatures = latest_seq.shape
latest_scaled = scaler_X.transform(latest_seq.reshape(-1, nfeatures)).reshape((1, SEQUENCE_LENGTH, nfeatures))

FORECAST_DAYS = 4  # today + next 3
preds, timestamps = [], []
base_ts = pd.Timestamp.now()

for i in range(FORECAST_DAYS):
    pred_scaled = model.predict(latest_scaled, verbose=0)
    pred_aqi = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
    preds.append(pred_aqi)
    timestamps.append(base_ts + pd.Timedelta(days=i))
    next_input = np.concatenate([latest_scaled[:, 1:, :], latest_scaled[:, -1:, :]], axis=1)
    latest_scaled = next_input

forecast_df = pd.DataFrame({"Date": timestamps, "Predicted AQI": preds})
forecast_df["Category"] = forecast_df["Predicted AQI"].apply(aqi_category)

# -----------------------------
# 🌤️ Display Forecast
# -----------------------------
st.subheader("🌞 Today's Forecast")
today_forecast = forecast_df.iloc[0]
st.metric(
    label=f"Predicted AQI (Today) — {today_forecast['Category']}",
    value=f"{today_forecast['Predicted AQI']:.2f}"
)

st.subheader("📅 AQI Forecast for Next 3 Days")
st.dataframe(
    forecast_df.iloc[1:].style.format({"Predicted AQI": "{:.2f}"}),
    use_container_width=True
)

# -----------------------------
# 📈 AQI History Chart
# -----------------------------
st.markdown("<h3 class='sub-header'>📊 AQI Trends (Past 30 Days)</h3>", unsafe_allow_html=True)
plt.figure(figsize=(10, 4))
plt.plot(df[TARGET].tail(30), color="#ab47bc", linewidth=2)
plt.title("Recent AQI Levels", fontsize=12)
plt.xlabel("Days")
plt.ylabel("AQI")
st.pyplot(plt.gcf())

# -----------------------------
# 🔗 Correlation Heatmap
# -----------------------------
st.markdown("<h3 class='sub-header'>🔗 Feature Correlation Heatmap</h3>", unsafe_allow_html=True)
corr = df[FEATURE_COLS + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

# -----------------------------
# 🧠 Explainability (Fallback Only)
# -----------------------------
st.markdown("<h3 class='sub-header'>🧠 Explainability</h3>", unsafe_allow_html=True)
st.info("Falling back to approximate permutation importance...")

try:
    X_sample = latest_seq.copy()
    base_pred = scaler_y.inverse_transform(
        model.predict(latest_scaled, verbose=0).reshape(-1, 1)
    )[0, 0]

    importances = {}
    for i, col in enumerate(FEATURE_COLS):
        X_perturbed = X_sample.copy()
        np.random.shuffle(X_perturbed[:, i])
        pert_scaled = scaler_X.transform(
            X_perturbed.reshape(-1, len(FEATURE_COLS))
        ).reshape((1, SEQUENCE_LENGTH, len(FEATURE_COLS)))
        pred_pert = scaler_y.inverse_transform(
            model.predict(pert_scaled, verbose=0).reshape(-1, 1)
        )[0, 0]
        importances[col] = abs(base_pred - pred_pert)

    imp_df = pd.DataFrame.from_dict(importances, orient="index", columns=["Importance"]).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(8, 5), facecolor="#f3e5f5")
    sns.barplot(x="Importance", y=imp_df.index, data=imp_df, palette="cool")
    plt.title("Approximate Feature Importance (Permutation-Based)")
    st.pyplot(plt.gcf())

except Exception as e:
    st.warning(f"Explainability section failed: {e}")

# -----------------------------
# 📋 Summary Stats
# -----------------------------
st.markdown("<h3 class='sub-header'>📋 Summary Statistics</h3>", unsafe_allow_html=True)
summary = df[FEATURE_COLS + [TARGET]].describe().T.round(2)
st.dataframe(summary, use_container_width=True)

# -----------------------------
# 🔁 Refresh Button
# -----------------------------
if st.sidebar.button("🔄 Refresh Dashboard"):
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    except Exception as e:
        st.warning(f"Could not refresh: {e}")
