import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from featureExtraction import extract_features_from_folder
from sklearn.metrics import confusion_matrix, accuracy_score

# Load features
real_features, real_labels = extract_features_from_folder("training_data/Real", 0)
fake_features, fake_labels = extract_features_from_folder("training_data/AI", 1)

X = np.array(real_features + fake_features)
y = np.array(real_labels + fake_labels)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
train_acc=accuracy_score(y_train, y_train_pred)
print(f"Training accuracy: {train_acc:.4f}")

cm_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix (Train):")
print(cm_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

cm_test = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Test):")
print(cm_test)

joblib.dump(clf, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model and scaler saved.")