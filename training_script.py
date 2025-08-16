import os 
import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split



# --- 1. Connect to Hopsworks ---
project = hopsworks.login(
    project="aqi_features_dataset", 
    api_key_value="t29A93H0Tnz50i2X.44jnj8Zbktd3HhUtXeMsXKBfixNUULLxJRf1XDAr3QUKxAJW3Eax40ZhZ5OmkqQ9"
)
fs = project.get_feature_store()

# --- 2. Get Data ---
feature_view = fs.get_feature_view(name="aqi_prediction_fv", version=1)
X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=0.2)

joblib.dump(X_train, "X_train.joblib")
joblib.dump(X_test, "X_test.joblib")
joblib.dump(y_train, "y_train.joblib")
joblib.dump(y_test, "y_test.joblib")

# Convert to numpy arrays
# y_train = y_train.values.ravel()
# y_test = y_test.values.ravel()

# --- 3. Train Model ---
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Evaluate Model ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# --- 5. Save Model ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/aqi_model.joblib")

# --- 6. Register in Hopsworks ---
mr = project.get_model_registry()

try:
    aqi_model = mr.python.create_model(
        name="aqi_predictor_model",
        description="Linear Regression AQI prediction model"
    )
    saved_model = aqi_model.save(
        model_path="model",
        # metrics={"rmse": rmse, "mae": mae, "r2_score": r2}
    )
    print(f"Model registered successfully! Version: {saved_model.version}")
except Exception as e:
    print(f"Registration failed: {e}")
    print("Model trained and saved locally in 'model' folder")
    
# Analyzing model performance
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Train R2:", r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
