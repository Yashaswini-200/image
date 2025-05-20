import numpy as np
from PIL import Image, ExifTags
import cv2

def extract_fft_features(image_array, num_features=1000):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift).flatten()
    magnitude_spectrum = np.sort(magnitude_spectrum)[::-1]  # descending
    if len(magnitude_spectrum) < num_features:
        magnitude_spectrum = np.pad(magnitude_spectrum, (0, num_features - len(magnitude_spectrum)), mode='constant')
    return magnitude_spectrum[:num_features]

def extract_metadata_features(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is not None:
            return [float(v) if isinstance(v, (int, float)) else 0.0 for v in exif_data.values()]
        else:
            return []
    except Exception:
        return []

def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        fft_features = extract_fft_features(img_array)

        metadata_features = extract_metadata_features(image_path)
        metadata_features = metadata_features[:50]
        if len(metadata_features) < 50:
            metadata_features += [0.0] * (50 - len(metadata_features))

        return np.concatenate([fft_features, metadata_features])
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.zeros(1050)