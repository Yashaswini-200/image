import joblib
import numpy as np
from featureExtraction import extract_fft_features, extract_metadata_features

def predict_image(image_path, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    fft_feat = extract_fft_features(image_path)
    meta_feat = extract_metadata_features(image_path)
    combined = np.concatenate((fft_feat, meta_feat)).reshape(1, -1)

    scaled = scaler.transform(combined)
    prediction = model.predict(scaled)[0]
    return "AI-Generated" if prediction == 0 else "Real"

if __name__ == "__main__":
    image_path = r"C:\Users\YASHASWINI\Downloads\ai image.jpg"

    print(f"Prediction: {predict_image(image_path)}")
