import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(api_key_value=os.getenv("aqi_forecast_api_key"))
fs = project.get_feature_store()

# Get feature group
fg = fs.get_feature_group(name="aqi_features", version=1)

# Read all rows
df = fg.read()

# Sort by timestamp and get last one
latest_row = df.sort_values("timestamp_utc").tail(1)

print("\n🆕 Latest Row (Full):\n")
print(latest_row.to_string(index=False))

# Explicitly print AQI
if "aqi_aqicn" in latest_row.columns:
    print("\n💨 Latest AQI:", latest_row["aqi_aqicn"].values[0])
else:
    print("\n⚠️ Column 'aqi_aqicn' not found in data.")
