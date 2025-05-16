import os
import numpy as np 
from skimage import io, color
from skimage.transform import resize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from PIL import Image
from PIL.ExifTags import TAGS
def extract_features(image_path, num_bins=20, size=(256, 256)):
    print(f"Processing image: {image_path}")
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = color.rgba2rgb(img)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = resize(img, size, anti_aliasing=True)

    # FFT features
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    log_F = np.log(1 + np.abs(Fshift))

    fft_mean = np.mean(log_F)
    fft_std = np.std(log_F)
    fft_entropy = -np.sum(log_F * np.log(log_F + 1e-10))

    print("FFT Mean:", fft_mean, "FFT Std:", fft_std, "FFT Entropy:", fft_entropy)

    hist_vals, _ = np.histogram(log_F.flatten(), bins=num_bins, density=True)
    print("Histogram:", hist_vals)

    # Metadata features
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        metadata_features = [0, 0, 0, 0, 0]
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, (str, int, float)):
                    metadata_features.append(len(str(value)))
        metadata_features = metadata_features[:5] if len(metadata_features) >= 5 else metadata_features + [0]*(5 - len(metadata_features))
    except:
        metadata_features = [0, 0, 0, 0, 0]

    print("Metadata:", metadata_features)

    return np.concatenate([[fft_mean, fft_std, fft_entropy], hist_vals, metadata_features])

# ----- Training -----
def get_all_images(folder):
    exts = ('.jpg', '.jpeg', '.png')
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

def train_model(real_folder, ai_folder):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    real_files = get_all_images(real_folder)
    ai_files = get_all_images(ai_folder)

    n = min(len(real_files), len(ai_files))
    real_files = real_files[:n]
    ai_files = ai_files[:n]
    print(f"Training with {n} images from each class.")

    features = []
    labels = []
    skipped = 0

    for i in range(n):
        try:
            f_real = extract_features(os.path.join(real_folder, real_files[i]))
            f_ai = extract_features(os.path.join(ai_folder, ai_files[i]))
            if np.any(np.isnan(f_real)) or np.any(np.isnan(f_ai)):
                skipped += 1
                continue
            features.append(f_real)
            labels.append(0)  # real
            features.append(f_ai)
            labels.append(1)  # ai
        except Exception as e:
            print(f"Skipping due to error: {e}")
            skipped += 1

    print(f"✅ Extracted {len(features)} valid feature vectors. Skipped {skipped}.")

    features = np.array(features)
    labels = np.array(labels)

    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("💾 Scaler saved to models/scaler.pkl")

    if np.any(np.isnan(features_norm)):
        print("⚠️ NaNs detected and removed.")
        mask = ~np.any(np.isnan(features_norm), axis=1)
        features_norm = features_norm[mask]
        labels = labels[mask]

    if features_norm.shape[0] == 0:
        raise ValueError("❌ No usable features. Check image quality or extractor.")

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(features_norm, labels)
    #Training accuracy
    train_preds = model.predict(features_norm)
    train_acc = accuracy_score(labels, train_preds)
    print(f"🎯 Training Accuracy: {train_acc:.4f}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(model, features_norm, labels, cv=cv)
    loss = 1 - accuracy_score(labels, preds)
    print(f"✅ Cross-validation loss: {loss:.4f}")

    print("📊 Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    joblib.dump(model, 'models/model.pkl')
    print("💾 Model saved to models/model.pkl")

    return model


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_model('training_data/real_folder', 'training_data/ai_folder')
