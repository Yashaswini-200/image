import joblib
import numpy as np
from featureExtraction import extract_features

MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# Load model and scaler once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_image(image_path):
    features = extract_features(image_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return "AI-generated" if prediction == 1 else "Real"

if __name__ == "__main__":
    test_path = r"C:\Users\YASHASWINI\Downloads\ChatGPT Image Apr 4, 2025, 06_59_17 PM.png"
    print(predict_image(test_path))
