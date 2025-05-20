import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def extract_fft_features(image_path, size=256):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (size, size))
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum.flatten()[:1000]

def extract_metadata_features(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            return [0] * 5
        decoded = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
        return [
            1 if 'Make' in decoded else 0,
            1 if 'Model' in decoded else 0,
            1 if 'ISOSpeedRatings' in decoded else 0,
            1 if 'ExposureTime' in decoded else 0,
            1 if 'DateTime' in decoded else 0
        ]
    except:
        return [0] * 5

def extract_features_from_folder(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Use your actual feature extraction function here
                feat = extract_fft_features(os.path.join(folder, filename))
                meta = extract_metadata_features(os.path.join(folder, filename))
                combined = np.concatenate((feat, meta))
                features.append(combined)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return features, labels
def extract_features_from_folder(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Use your actual feature extraction function here
                feat = extract_fft_features(os.path.join(folder, filename))
                meta = extract_metadata_features(os.path.join(folder, filename))
                combined = np.concatenate((feat, meta))
                features.append(combined)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return features, labels