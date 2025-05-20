import joblib
import numpy as np
from featureExtraction import extract_features
import requests
from PIL import Image
from io import BytesIO
import os

def predict_image(image_path):
    MODEL_PATH = 'models/model.pkl'
    SCALER_PATH = 'models/scaler.pkl'

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(image_path)
    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    result = "AI-generated" if prediction == 1 else "Real"
    print(f"Prediction: {result}")
    return result

if __name__ == "__main__":
    # Local file prediction
    test_image_path = "training_data/AI/7e34225424b78a952f0a3d160b.jpg"  # change this to your image path
    predict_image(test_image_path)