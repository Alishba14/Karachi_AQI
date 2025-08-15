# feature_script.py
import hopsworks
import pandas as pd
import os

# --- 1. Connect to Hopsworks ---
# This uses the GitHub Secret to get your API key
project = hopsworks.login(
    project="aqi_features_dataset", 
    api_key_value=os.environ["HPSW_API_KEY"]
)
fs = project.get_feature_store()

# --- 2. Load and Prepare Data ---
DATASET_FILENAME = 'karachi_AQI_features_engineered.csv'

# Check if the file exists locally before trying to load it
if not os.path.exists(DATASET_FILENAME):
    print(f"Error: Dataset file not found at {DATASET_FILENAME}. Please ensure it's in the project root.")
    exit(1)

df_features = pd.read_csv(DATASET_FILENAME)

# Convert the 'time' column to a Unix timestamp (BIGINT)
df_features['time'] = pd.to_datetime(df_features['time']).astype('int64') // 10**6

# --- 3. Create or Get Feature Group and Insert Data ---
print("Connecting to Feature Group...")
feature_group = fs.get_or_create_feature_group(
    name="aqi_feature_group", 
    version=1,
    description="Engineered features for hourly Air Quality Index (AQI) prediction in Karachi.",
    primary_key=["time"],
    event_time="time",
    online_enabled=True
)

print("Inserting data into Feature Group...")
feature_group.insert(df_features, write_options={"wait_for_job": True})

print("✅ Feature ingestion successful!")