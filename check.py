import hopsworks
import os

print("🔐 Connecting to Hopsworks...")
project = hopsworks.login()

mr = project.get_model_registry()

# Get all versions of the model
models = mr.get_models("rf_aqi_model")

if not models:
    print("❌ No models found with name 'rf_aqi_model'")
else:
    latest_model = max(models, key=lambda m: m.version)
    print(f"✅ Latest model: {latest_model.name} (version {latest_model.version})")

    model_dir = latest_model.download()
    print(f"\n📂 Model downloaded at: {model_dir}\n")

    # Walk through directory and show all files
    print("📁 Directory structure:\n")
    for root, dirs, files in os.walk(model_dir):
        level = root.replace(model_dir, "").count(os.sep)
        indent = "    " * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        subindent = "    " * (level + 1)
        for f in files:
            print(f"{subindent}- {f}")

    # Check specifically for model.pkl and scaler.pkl
    model_pkl = None
    scaler_pkl = None

    for root, _, files in os.walk(model_dir):
        if "model.pkl" in files:
            model_pkl = os.path.join(root, "model.pkl")
        if "scaler.pkl" in files:
            scaler_pkl = os.path.join(root, "scaler.pkl")

    if model_pkl and scaler_pkl:
        print(f"\n✅ Found both files:")
        print(f"   - model.pkl: {model_pkl}")
        print(f"   - scaler.pkl: {scaler_pkl}")
    else:
        print("\n❌ model.pkl or scaler.pkl not found in downloaded model.")
