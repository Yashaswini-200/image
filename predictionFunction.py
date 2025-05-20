import joblib
import numpy as np
from featureExtraction import extract_features
import os

def predict_image(image_path):
    MODEL_PATH = 'models/model.pkl'
    SCALER_PATH = 'models/scaler.pkl'

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[ERROR] Model or scaler not found.")
        return "Prediction could not be determined"

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"[Model Load ERROR] {e}")
        return "Prediction could not be determined"

    features = extract_features(image_path)

    if features is None or len(features) == 0:
        print("[ERROR] No features extracted.")
        return "Prediction could not be determined"
    if np.all(features == 0):
        print("[ERROR] All features are zero.")
        return "Prediction could not be determined"

    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        result = "AI-generated" if prediction == 1 else "Real"
        print(f"[Prediction] {result}")
        return result
    except Exception as e:
        print(f"[Prediction ERROR] {e}")
        return "Prediction could not be determined"

if __name__ == "__main__":
    test_image_path = "training_data/AI/sample.jpg"
    predict_image(test_image_path)
