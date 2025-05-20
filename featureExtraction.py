import numpy as np
from PIL import Image, ExifTags
import cv2

def extract_fft_features(image_array, num_features=1000):
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift).flatten()
        magnitude_spectrum = np.sort(magnitude_spectrum)[::-1]
        if len(magnitude_spectrum) < num_features:
            magnitude_spectrum = np.pad(magnitude_spectrum, (0, num_features - len(magnitude_spectrum)), mode='constant')
        return magnitude_spectrum[:num_features]
    except Exception as e:
        print(f"[FFT ERROR] {e}")
        return np.zeros(num_features)

def extract_metadata_features(image_path, num_metadata_features=50):
    metadata_vector = []
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data:
            values = list(exif_data.values())
            for val in values:
                if isinstance(val, (int, float)):
                    metadata_vector.append(float(val))
                else:
                    metadata_vector.append(0.0)
        else:
            print("[Metadata] No EXIF data found.")
    except Exception as e:
        print(f"[Metadata ERROR] {e}")
    
    if len(metadata_vector) < num_metadata_features:
        metadata_vector += [0.0] * (num_metadata_features - len(metadata_vector))
    return metadata_vector[:num_metadata_features]

def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        fft_features = extract_fft_features(img_array)
        metadata_features = extract_metadata_features(image_path)

        combined = np.concatenate([fft_features, metadata_features])
        if np.all(combined == 0):
            print("[Warning] Feature vector is all zeros!")
        return combined
    except Exception as e:
        print(f"[Feature Extraction ERROR] {e}")
        return np.zeros(1050)
