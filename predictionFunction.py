# predictionFunction.py
import joblib
from featureExtraction import extract_features

MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_image(image_input):
    if model is None or scaler is None:
        raise ValueError("Model and scaler must be provided!")
    features = extract_features(image_input)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return "AI-generated" if prediction == 1 else "Real"

if __name__ == "__main__":
    test_path = "training_data/AI/7e34225424b78a952f0a3d160b.jpg"
    print(predict_image(test_path))