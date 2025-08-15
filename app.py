# app.py - Karachi AQI Prediction Dashboard
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import hopsworks
from hsfs.client.exceptions import RestAPIError

# connecting with hopsworks
HOPSWORKS_PROJECT_NAME = "aqi_features_dataset"
HOPSWORKS_API_KEY = "rVIbPsK6ibgCfJnj.NNYtCUFCV0CpXSiHU3XRnZaxPiyHps4HgS2sJfKWJKK2ILoLTfJQxib3LnU5qygY"


@st.cache_resource 
def load_latest_model(model_name="aqi_predictor_model", version=1):
    """Loads the pre-trained machine learning model from Hopsworks Model Registry."""
    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
        mr = project.get_model_registry()
        
        # Download the model artifact
        model_instance = mr.get_model(name=model_name, version=version)
        model_dir = model_instance.download()
        
        # Load the model artifact itself. Ensure the filename matches what was saved by training_script.py.
        # Based on previous conversations, your model is saved as 'aqi_model.joblib'
        model = joblib.load(f"{model_dir}/aqi_model.joblib")
        
        # st.success(f"Successfully loaded model '{model_name}' version {version} from Hopsworks.")
        return model
    except RestAPIError as e:
        st.error(f"Error loading model from Hopsworks: {e}. Please ensure the model is registered and API key is correct.")
        return None
    except FileNotFoundError:
        st.error(f"Error: Model artifact 'aqi_model.joblib' not found in downloaded directory. This might indicate an issue with model saving or naming in the training script.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None

@st.cache_data # Caches data loading to prevent re-running on every Streamlit interaction
def load_data(file_path):
    """
    Loads the preprocessed feature-engineered dataset from a CSV file.
    Returns two DataFrames: one for model input and one for display.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at '{file_path}'. Please ensure 'karachi_AQI_features_engineered.csv' is in the same directory as app.py or provide the full path.")
        return None, None
    try:
        # Load the CSV without setting 'time' as index initially
        df_raw = pd.read_csv(file_path)
        
        # Create a copy for model input: convert 'time' to numerical Unix timestamp (milliseconds)
        df_model_input = df_raw.copy()
        df_model_input['time'] = pd.to_datetime(df_model_input['time']).astype('int64') // 10**6
        
        # Create a separate copy for display: convert 'time' to datetime and set as index
        df_display = df_raw.copy()
        df_display['time'] = pd.to_datetime(df_display['time'])
        df_display.set_index('time', inplace=True)

        # Rename target_pm2_5 to aqi if it exists and aqi doesn't (for consistency)
        if 'target_pm2_5' in df_model_input.columns and 'aqi' not in df_model_input.columns:
            df_model_input = df_model_input.rename(columns={'target_pm2_5': 'aqi'})
        if 'target_pm2_5' in df_display.columns and 'aqi' not in df_display.columns:
            df_display = df_display.rename(columns={'target_pm2_5': 'aqi'})
        
        # Interpolate missing values (as done in the notebook)
        df_model_input.interpolate(method='linear', inplace=True)
        df_display.interpolate(method='linear', inplace=True)
        
        return df_model_input, df_display
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None


# --- Main Application Logic ---

# Load the model and data
model = load_latest_model()
df_for_model, df_for_display = load_data('karachi_AQI_features_engineered.csv')

# Determine feature columns for the model
feature_columns = []
if df_for_model is not None:
    # Exclude the target and any other non-feature columns from the model input
    feature_columns = [col for col in df_for_model.columns if col not in ['aqi', 'calculated_aqi']]


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Karachi AQI Predictor")

st.title("🌏 Karachi AQI Predictor Dashboard")
st.markdown("Predicting future Air Quality Index (AQI) for Karachi using machine learning.")

if df_for_model is None or df_for_display is None or model is None or not feature_columns:
    st.warning("Application could not load data or model. Please check file paths, API key, and ensure data/model integrity.")
else:
    # --- Display Latest AQI Trends ---
    st.header("📈 Latest Historical AQI Trends")
    st.markdown("Displaying the last 7 days of recorded AQI data.")
    
    last_7_days_df = df_for_display['aqi'].last('7D')
    
    if not last_7_days_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(last_7_days_df.index, last_7_days_df.values, color='blue', linewidth=2)
        ax.set_title('Last 7 Days AQI Trend')
        ax.set_xlabel('Date')
        ax.set_ylabel('AQI')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough data to display last 7 days AQI trend.")

    st.write("Current AQI (last recorded value):", f"**{df_for_display['aqi'].iloc[-1]:.2f}**")
    st.write(f"Last updated on: **{df_for_display.index[-1].strftime('%Y-%m-%d %H:%M')}**")

    # --- Prediction Section ---
    # st.header("🔮 Forecast Next Hour's AQI")
    

    st.header("🌏 Forecast AQI for Multiple Hours")
    num_hours = st.number_input("Select number of hours to forecast (1-72):", min_value=1, max_value=72, value=1)
    predict_button = st.button(f"Predict AQI for Next {num_hours} Hour(s)")

    if predict_button:
        st.subheader("Predicted AQI:")
        # Get the last N recorded features for the prediction
        latest_features_for_prediction = df_for_model[feature_columns].tail(num_hours)
        try:
            predictions = model.predict(latest_features_for_prediction)
            for i in range(num_hours):
                pred = predictions[i]
                aqi_category = "Unknown"
                if pred <= 50:
                    aqi_category = "Good"
                elif pred <= 100:
                    aqi_category = "Moderate"
                elif pred <= 150:
                    aqi_category = "Unhealthy for Sensitive Groups"
                elif pred <= 200:
                    aqi_category = "Unhealthy"
                elif pred <= 300:
                    aqi_category = "Very Unhealthy"
                else:
                    aqi_category = "Hazardous"
                st.metric(label=f"Predicted AQI (Hour {i+1})", value=f"{pred:.2f}", help=f"Forecasted AQI for hour {i+1}")
                st.success(f"Category: **{aqi_category}**")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            
    #Feature Importance Visualization
    st.header("📊 Feature Importance")
    st.markdown("Understanding which features are most impactful in the AQI predictions.")
    
    SHAP_BAR_PLOT_PATH = 'shap_feature_importance_bar_lasso_regression.png'
    SHAP_SUMMARY_PLOT_PATH = 'shap_feature_importance_summary_lasso_regression.png'

    if os.path.exists(SHAP_BAR_PLOT_PATH):
        st.image(SHAP_BAR_PLOT_PATH, caption='SHAP Feature Importance (Average Impact)', use_column_width=True)
    else:
        st.info(f"SHAP bar plot not found at '{SHAP_BAR_PLOT_PATH}'. Please ensure it's generated and in the correct directory.")

    if os.path.exists(SHAP_SUMMARY_PLOT_PATH):
        st.image(SHAP_SUMMARY_PLOT_PATH, caption='SHAP Feature Importance (Impact Distribution)', use_column_width=True)
    else:
        st.info(f"SHAP summary plot not found at '{SHAP_SUMMARY_PLOT_PATH}'. Please ensure it's generated and in the correct directory.")
    
    st.markdown("---")
    st.markdown("Project Developed for AQI Prediction in Karachi.")