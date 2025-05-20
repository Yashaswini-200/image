import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from featureExtraction import extract_features

def load_dataset(folder_real, folder_ai):
    features = []
    labels = []

    for label, folder in [(0, folder_real), (1, folder_ai)]:
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder, filename)
                try:
                    feat = extract_features(path)
                    features.append(feat)
                    labels.append(label)
                except Exception as e:
                    print(f"Failed to extract features from {filename}: {e}")

    return np.array(features), np.array(labels)

def train_and_save_model():
    folder_real = 'training_data/Real'
    folder_ai = 'training_data/AI'

    X, y = load_dataset(folder_real, folder_ai)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)

    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validation accuracy: {scores.mean():.4f}")
    y_pred = model.predict(X_scaled)
    print(f"✅ Training Accuracy: {accuracy_score(y, y_pred):.4f}")
    cm = confusion_matrix(y, y_pred)
    print("📊 Confusion Matrix (rows = true, cols = predicted):")
    print("             Pred_Real  Pred_AI")
    print(f"True_Real   {cm[0,0]:10}  {cm[0,1]:7}")
    print(f"True_AI     {cm[1,0]:10}  {cm[1,1]:7}")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

if __name__ == "__main__":
    train_and_save_model()
