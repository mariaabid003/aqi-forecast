"""
🌸 AQI Forecast Dashboard — Scikit-learn Version (Random Forest)
--------------------------------------------------
Uses the Random Forest model trained and saved as rf_model.joblib
Features:
✅ Today's Forecast + Next 3-Day Forecast
✅ AQI Trends, Heatmap, and Summary Stats
✅ Pastel UI Design
"""

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta

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
# 🧩 Load Environment & Connect to Hopsworks
# -----------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("aqi_forecast_api_key") or os.getenv("HOPSWORKS_API_KEY")

st.title("🌤️ AQI Forecast Dashboard")
st.caption("Air Quality Prediction and Trends")

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

# -----------------------------
# 🎯 Define Columns
# -----------------------------
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
# 🔮 Load Random Forest Model and Scaler
# -----------------------------
MODEL_PATH = "models/rf_model.joblib"
SCALER_PATH = "models/rf_scaler.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.success("✅ Random Forest model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
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
# 🔮 Generate Forecast
# -----------------------------
X_scaled = scaler.transform(df[FEATURE_COLS])
preds = model.predict(X_scaled)

df["predicted_aqi"] = preds
forecast_df = df.tail(4).copy()
forecast_df["Date"] = [datetime.now() + timedelta(days=i) for i in range(len(forecast_df))]
forecast_df["Category"] = forecast_df["predicted_aqi"].apply(aqi_category)

# -----------------------------
# 🌤️ Display Forecast
# -----------------------------
st.subheader("🌞 Today's Forecast")
today = forecast_df.iloc[-1]
st.metric(
    label=f"Predicted AQI (Today) — {today['Category']}",
    value=f"{today['predicted_aqi']:.2f}"
)

st.subheader("📅 AQI Forecast for Next 3 Days")
st.dataframe(
    forecast_df[["Date", "predicted_aqi", "Category"]].style.format({"predicted_aqi": "{:.2f}"}),
    use_container_width=True
)

# -----------------------------
# 📈 AQI Trends Chart
# -----------------------------
st.markdown("<h3 class='sub-header'>📊 AQI Trends (Past 30 Days)</h3>", unsafe_allow_html=True)
plt.figure(figsize=(10, 4))
plt.plot(df[TARGET].tail(30).values, color="#ab47bc", linewidth=2)
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
# 📋 Summary Stats
# -----------------------------
st.markdown("<h3 class='sub-header'>📋 Summary Statistics</h3>", unsafe_allow_html=True)
summary = df[FEATURE_COLS + [TARGET]].describe().T.round(2)
st.dataframe(summary, use_container_width=True)

# -----------------------------
# 🔁 Refresh Button
# -----------------------------
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.experimental_rerun()
