# app.py for Multi_Predicto
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from time import sleep

# ---------------------------
# Paths (adjust per dataset/model)
# ---------------------------
MODEL_DIR = "models"
LOG_FILE = "logs/input_errors.log"

# Datasets & models mapping
PROJECTS = {
    "Laptop Prices": {
        "model": f"{MODEL_DIR}/laptop_best_model.pkl",
        "features": f"{MODEL_DIR}/laptop_features.pkl",
        "bounds": (100, 6920)
    },
    "Mobile Prices": {
        "model": f"{MODEL_DIR}/mobile_best_model.pkl",
        "features": f"{MODEL_DIR}/mobile_features.pkl",
        "bounds": (50, 2000)
    },
    "Crypto Prices": {
        "model": f"{MODEL_DIR}/crypto_best_model.pkl",
        "features": f"{MODEL_DIR}/crypto_features.pkl",
        "bounds": (0, 100000)
    }
}

# ---------------------------
# Utilities
# ---------------------------
def log_event(kind, field, value, message):
    """Log invalid input or warnings."""
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {kind.upper()} | Field: {field} | Value: '{value}' | {message}\n")
    except Exception as e:
        print("Logging failed:", e)

def load_model_and_features(project_name):
    """Load model and feature names."""
    try:
        model_path = PROJECTS[project_name]["model"]
        features_path = PROJECTS[project_name]["features"]
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model/features: {e}")
        return None, None

def preprocess_input(sample_dict, feature_names):
    """Encode user input to match training features."""
    if feature_names is None:
        return None
    sample = pd.DataFrame([sample_dict])
    sample_encoded = pd.get_dummies(sample)
    sample_encoded = sample_encoded.reindex(columns=feature_names, fill_value=0)
    return sample_encoded

def predict(model, input_data):
    """Predict using loaded model."""
    if model is None or input_data is None:
        return None
    try:
        pred = model.predict(input_data)[0]
        return round(float(pred), 4)
    except Exception as e:
        log_event("error", "PREDICT", str(input_data), f"Prediction failed: {e}")
        st.warning("Prediction failed. Check inputs.")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Multi_Predicto – Predict Prices & More")
st.write("Select a dataset and enter features to predict values.")

# Select project
project = st.selectbox("Choose a project:", list(PROJECTS.keys()))

# Load model/features
model, feature_names = load_model_and_features(project)
bounds = PROJECTS[project]["bounds"]

# Example generic inputs (update per dataset)
if project == "Laptop Prices":
    company = st.selectbox("Company", ["Dell", "HP", "Lenovo", "Apple"])
    typename = st.selectbox("Type", ["Ultrabook", "Gaming", "Notebook"])
    ram = st.slider("RAM (GB)", 4, 128, 8)
    ssd = st.number_input("SSD (GB)", 32, 4000, 512, 32)
    hdd = st.number_input("HDD (GB)", 0, 6000, 0, 500)
    weight = st.number_input("Weight (kg)", 0.5, 5.0, 1.5, 0.1)
    inches = st.slider("Screen Size (inches)", 10.0, 18.0, 13.3, 0.1)
    touch = st.selectbox("Touchscreen", ["Yes", "No"])
    
    input_dict = {
        "Company": company,
        "TypeName": typename,
        "Ram": ram,
        "SSD": ssd,
        "HDD": hdd,
        "Weight": weight,
        "Inches": inches,
        "Touch": touch
    }

# Add other datasets here (Mobile Prices, Crypto, etc.)
# Example placeholder:
# elif project == "Mobile Prices":
#     input_dict = {...}

# Predict button
if st.button("Predict"):
    encoded_input = preprocess_input(input_dict, feature_names)
    prediction = predict(model, encoded_input)
    if prediction is not None:
        if prediction < bounds[0] or prediction > bounds[1]:
            st.warning(f"Prediction seems unrealistic: {prediction}")
            log_event("warn", "Prediction", str(input_dict), f"Unrealistic prediction: {prediction}")
        else:
            st.success(f"Predicted value: {prediction}")

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    font-size: 12px;
    color: gray;
}
</style>
<div class="footer">© 2025 Multi_Predicto</div>
""", unsafe_allow_html=True)
