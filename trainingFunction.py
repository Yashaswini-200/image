import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from featureExtraction import extract_features

def load_images_and_labels(root_folder):
    features = []
    labels = []
    label_map = {"Real": 0, "AI": 1}
    counts = {"Real": 0, "AI": 0}

    for label_name, label_value in label_map.items():
        folder_path = os.path.join(root_folder, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(folder_path, filename)
                try:
                    feature_vector = extract_features(full_path)
                    features.append(feature_vector)
                    labels.append(label_value)
                    counts[label_name] += 1
                except Exception as e:
                    print(f"Error extracting features from {full_path}: {e}")

    print(f"Loaded {counts['Real']} real images and {counts['AI']} AI images.")
    if abs(counts['Real'] - counts['AI']) > 0:
        print("⚠️  Warning: Dataset is unbalanced. Consider balancing for better results.")

    return np.array(features), np.array(labels)

def main():
    root_folder = "training_data"
    X, y = load_images_and_labels(root_folder)
    print("Feature matrix shape:", X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    y_pred = model.predict(X_scaled)
    print("Training Accuracy:", accuracy_score(y, y_pred))

    scores = cross_val_score(model, X_scaled, y, cv=5)
    print("Cross-val Accuracy:", scores.mean())

    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix (rows: true, cols: pred):")
    print("        Pred_Real  Pred_AI")
    print(f"True_Real   {cm[0,0]:8}  {cm[0,1]:7}")
    print(f"True_AI     {cm[1,0]:8}  {cm[1,1]:7}")

if __name__ == "__main__":
    main()