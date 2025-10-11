import schedule
import time
import subprocess
import os
from datetime import datetime

def run_backfill():
    backfill_path = os.path.join(os.getcwd(), "features", "backfill.py")

    if not os.path.exists(backfill_path):
        print(f"🚨 backfill.py not found at: {backfill_path}")
        return

    print(f"🕐 Running backfill at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ...")
    try:
        subprocess.run(["python", backfill_path], check=True)
        print("✅ Backfill run complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Backfill failed with error code {e.returncode}")
    except Exception as e:
        print(f"⚠️ Unexpected error during backfill: {e}")

# Run every 15 minutes
schedule.every(15).minutes.do(run_backfill)

print("🚀 Scheduler started. Running every 15 minutes...")

while True:
    schedule.run_pending()
    time.sleep(60)

