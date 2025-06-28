import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from featureExtraction import extract_all_features

def train_model(real_dir, ai_dir, model_path='model.pkl', scaler_path='scaler.pkl'):
    X, y = [], []

    # Load real images
    for file in os.listdir(real_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(real_dir, file)
            features = extract_all_features(path)
            X.append(features)
            y.append(0)

    # Load AI images
    for file in os.listdir(ai_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(ai_dir, file)
            features = extract_all_features(path)
            X.append(features)
            y.append(1)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
