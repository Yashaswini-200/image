import os
import joblib
import numpy as np
from featureExtraction import extract_features
import requests
from PIL import Image
from io import BytesIO

# Load model and scaler (adjust paths if needed)
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

def predict_image(image_path):
    features = extract_features(image_path)
    if features is None or np.any(np.isnan(features)):
        return "Error: Invalid or corrupt image."

    features = features.reshape(1, -1)
    features_normalized = scaler.transform(features)
    prediction = model.predict(features_normalized)[0]
    return "Real" if prediction == 0 else "AI-generated"

def predict_from_url(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        temp_path = "temp_downloaded_image.jpg"
        img.save(temp_path)
        result = predict_image(temp_path)
        os.remove(temp_path)
        return result
    except Exception as e:
        return f"Error: {e}"

def evaluate_on_folder(folder_path, label):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    correct = 0
    total = len(files)
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        pred = predict_image(filepath)
        pred_label = 0 if pred == "Real" else 1
        if pred_label == label:
            correct += 1
        print(f"{filename}: Predicted={pred}, Actual={'Real' if label == 0 else 'AI-generated'}")

    accuracy = correct / total if total > 0 else 0
    error_rate = 1 - accuracy
    print(f"Accuracy on {'Real' if label == 0 else 'AI-generated'} images: {accuracy:.2%}")
    print(f"Error rate on {'Real' if label == 0 else 'AI-generated'} images: {error_rate:.2%}")
    return accuracy, error_rate

if __name__ == "__main__":
    # Change these paths to your test images or URLs
    local_image_path = r"C:\Users\YASHASWINI\OneDrive\Documents\ai_folder\ai image7.jpg"
    test_image_url = "https://ai-pro.org/wp-content/uploads/2024/01/dream-photo-hd-1.png"

    print("🧠 Prediction (local file):", predict_image(local_image_path))
    print("🧠 Prediction (from URL):", predict_from_url(test_image_url))