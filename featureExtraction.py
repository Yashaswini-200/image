from PIL import Image
import numpy as np
import cv2

def extract_fft_features(image_array, num_features=1000):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift).flatten()
    magnitude_spectrum = np.sort(magnitude_spectrum)[::-1]
    if len(magnitude_spectrum) < num_features:
        magnitude_spectrum = np.pad(magnitude_spectrum, (0, num_features - len(magnitude_spectrum)), mode='constant')
    return magnitude_spectrum[:num_features]

def extract_metadata_features(image):
    try:
        exif_data = image._getexif()
        if exif_data:
            values = []
            for v in exif_data.values():
                if isinstance(v, (int, float)):
                    values.append(float(v))
                else:
                    values.append(0.0)
            return values
        else:
            return []
    except Exception:
        return []

def extract_features(image_input):
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
        img_array = np.array(img)
        metadata_features = extract_metadata_features(img)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        img_array = np.array(img)
        metadata_features = extract_metadata_features(img)
    elif isinstance(image_input, np.ndarray):
        img_array = image_input
        metadata_features = []
    else:
        raise ValueError("Unsupported image input type")

    fft_features = extract_fft_features(img_array)

    metadata_features = metadata_features[:50]
    if len(metadata_features) < 50:
        metadata_features += [0.0] * (50 - len(metadata_features))

    return np.concatenate([fft_features, metadata_features])