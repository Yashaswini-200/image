import os
import joblib
import numpy as np
from featureExtraction import extract_features
import requests
from PIL import Image
from io import BytesIO

# Load model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

def predict_image(image_path):
    features = extract_features(image_path)
    if features is None or np.any(np.isnan(features)):
        return "Error: Invalid or corrupt image."
    features = np.array(features).reshape(1, -1)
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

def evaluate_on_folder(folder, label):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    correct = 0
    total = 0
    for f in files:
        path = os.path.join(folder, f)
        pred = predict_image(path)
        pred_label = 0 if pred == "Real" else 1
        if pred_label == label:
            correct += 1
        total += 1
        print(f"{f}: Predicted={pred}, Actual={'Real' if label==0 else 'AI'}")
    accuracy = correct / total if total > 0 else 0
    error_rate = 1 - accuracy
    print(f"Accuracy on {'Real' if label==0 else 'AI'} images: {accuracy:.2%}")
    print(f"Error rate on {'Real' if label==0 else 'AI'} images: {error_rate:.2%}")
    return accuracy, error_rate

if __name__ == "__main__":
    # Example usage for local file
    image_path = r"C:\Users\YASHASWINI\OneDrive\Documents\ai_folder\ai image7.jpg"  # Set your test image path here
    if image_path:
        result = predict_image(image_path)
        print("🧠 Prediction (local file):", result)

    # Example usage for URL
    image_url = "https://ai-pro.org/wp-content/uploads/2024/01/dream-photo-hd-1.png"  # Replace with a real image URL
    if image_url:
        result_url = predict_from_url(image_url)
        print("🧠 Prediction (from URL):", result_url)