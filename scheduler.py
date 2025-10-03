# scheduler.py
import schedule
import time
import subprocess
import pandas as pd
import os

def run_backfill():
    print("⏳ Running backfill.py...")
    subprocess.run(["python", "features/backfill.py"], check=True)

    # After backfill, show current dataset size
    csv_path = "data/features/training_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"📊 Current dataset size: {len(df)} rows")
    else:
        print("⚠️ No dataset file found yet.")

# Run every 15 minutes
schedule.every(15).minutes.do(run_backfill)

print("✅ Scheduler started. Running backfill every 15 minutes...")
while True:
    schedule.run_pending()
    time.sleep(10)

