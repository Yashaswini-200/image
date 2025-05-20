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
            print(f"[WARNING] Folder missing: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(folder_path, filename)
                try:
                    feature_vector = extract_features(full_path)
                    if np.all(feature_vector == 0):
                        print(f"[SKIPPED] Zero feature vector for {filename}")
                        continue
                    features.append(feature_vector)
                    labels.append(label_value)
                    counts[label_name] += 1
                except Exception as e:
                    print(f"[ERROR] Feature extraction failed for {filename}: {e}")

    print(f"\n✅ Loaded: {counts['Real']} Real | {counts['AI']} AI")
    if abs(counts["Real"] - counts["AI"]) > 0:
        print("⚠️ Dataset is unbalanced. Consider balancing it for better accuracy.\n")

    return np.array(features), np.array(labels)

def main():
    root_folder = "training_data"
    X, y = load_images_and_labels(root_folder)

    if len(X) == 0 or len(y) == 0:
        print("[ERROR] No data found. Training aborted.")
        return

    print(f"🧩 Feature matrix shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n✅ Training Accuracy: {accuracy:.4f}")

    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"🔄 Cross-Validation Accuracy: {scores.mean():.4f}")

    cm = confusion_matrix(y, y_pred)
    print("\n📊 Confusion Matrix (rows = true, cols = predicted):")
    print("             Pred_Real  Pred_AI")
    print(f"True_Real     {cm[0, 0]:>9}  {cm[0, 1]:>8}")
    print(f"True_AI       {cm[1, 0]:>9}  {cm[1, 1]:>8}")

if __name__ == "__main__":
    main()