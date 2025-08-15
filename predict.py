# predict.py - Upgraded for MLOps
import pandas as pd
import hopsworks
import joblib
import sys # To read arguments from command line
import os # To check if file exists
from hsfs.client.exceptions import RestAPIError

# --- 1. Connect to Hopsworks and Get Model Registry ---
def load_latest_model(model_name, version, project_name="aqi_features_dataset", api_key="rVIbPsK6ibgCfJnj.NNYtCUFCV0CpXSiHU3XRnZaxPiyHps4HgS2sJfKWJKK2ILoLTfJQxib3LnU5qygY"):
    """Loads the pre-trained machine learning model from Hopsworks Model Registry."""
    try:
        project = hopsworks.login(project=project_name, api_key_value=api_key)
        mr = project.get_model_registry()
        
        model_instance = mr.get_model(name=model_name, version=version)
        if model_instance is None:
            print(f"Error: Model '{model_name}' version {version} not found in Hopsworks registry.")
            sys.exit(1)
        model_dir = model_instance.download() # Downloads the model artifact to a local directory
        # Load the model artifact itself
        model_filename = f"{model_dir}/aqi_model.joblib" # Adjust this to the actual filename
        model = joblib.load(model_filename) 
        print(f"Successfully loaded model '{model_name}' version {version} from Hopsworks.")
        return model
    except RestAPIError as e:
        print(f"Error loading model from Hopsworks: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Model artifact not found in downloaded directory.")
        sys.exit(1)

def load_data(file_path):
    """Loads the preprocessed feature-engineered dataset."""
    try:
        # Load the CSV without setting the index
        df = pd.read_csv(file_path)
        
        # Convert the 'time' column to a Unix timestamp (milliseconds)
        df['time'] = pd.to_datetime(df['time']).astype('int64') // 10**6

        if 'target_pm2_5' in df.columns and 'aqi' not in df.columns:
            df = df.rename(columns={'target_pm2_5': 'aqi'})
            
        df.interpolate(method='linear', inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        sys.exit(1)

def get_latest_features(df, num_rows=1):
    """
    Extracts the latest features from the DataFrame for prediction.
    """
    features_to_exclude = ['aqi', 'calculated_aqi', 'target_pm2_5'] 
    """
    Extracts the latest features from the DataFrame for forecasting.
    """
    num_rows = min(num_rows, 168)  # Limit to 1 week (168 hours)
    features_to_exclude = ['aqi', 'calculated_aqi', 'target_pm2_5']
    X_cols = [col for col in df.columns if col not in features_to_exclude]
    latest_data = df[X_cols].tail(num_rows)
    if latest_data.empty:
        print("Error: No data available for prediction.")
        sys.exit(1)
    return latest_data
    return latest_data


def make_prediction(model, features):
    """
    Makes a prediction using the loaded model and features.
    """
    prediction = model.predict(features)
    return prediction

if __name__ == "__main__":
    # Note: The model is now loaded from Hopsworks, so MODEL_FILENAME is no longer needed
    DATASET_FILENAME = 'karachi_AQI_features_engineered.csv'
    
    # Define model to be fetched from the registry
    MODEL_NAME = 'aqi_predictor_model' 
    MODEL_VERSION = 1

    print(f"Loading data from: {DATASET_FILENAME}")

    # Load model and data
    model = load_latest_model(MODEL_NAME, MODEL_VERSION)
    df = load_data(DATASET_FILENAME)

    # Ask user for prediction period (number of hours, up to 72)
    try:
        num_hours = int(input("Enter number of hours to forecast (1-72): "))
        if num_hours < 1 or num_hours > 72:
            print("Invalid input. Defaulting to 1 hour.")
            num_hours = 1
    except Exception:
        print("Invalid input. Defaulting to 1 hour.")
        num_hours = 1

    # Get the latest features from the dataset for prediction
    latest_features = get_latest_features(df, num_rows=num_hours)

    print("\nFeatures used for prediction:")
    print(latest_features)

    # Make prediction
    prediction = make_prediction(model, latest_features)

    for i in range(num_hours):
        print(f"Predicted AQI for hour {i+1}: {prediction[i]:.2f}")
        if prediction[i] > 150:
            print("ALERT: Predicted AQI is unhealthy!")