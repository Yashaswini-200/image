from PIL import Image, ExifTags
import numpy as np
import cv2

NUM_FFT_FEATURES = 1000
NUM_META_FEATURES = 50
TOTAL_FEATURES = NUM_FFT_FEATURES + NUM_META_FEATURES

def extract_fft_features(image_array, num_features=NUM_FFT_FEATURES):
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift).flatten()
        magnitude_spectrum = np.sort(magnitude_spectrum)[::-1]  # Top features
        if len(magnitude_spectrum) < num_features:
            magnitude_spectrum = np.pad(magnitude_spectrum, (0, num_features - len(magnitude_spectrum)), mode='constant')
        return magnitude_spectrum[:num_features]
    except Exception as e:
        print(f"[FFT Error] {e}")
        return np.zeros(num_features)

def extract_metadata_features(pil_image, num_features=NUM_META_FEATURES):
    try:
        exif_data = pil_image._getexif()
        metadata = []
        if exif_data:
            for tag, value in exif_data.items():
                if isinstance(value, (int, float)):
                    metadata.append(float(value))
                else:
                    metadata.append(0.0)
        # Pad or truncate to fixed size
        metadata = metadata[:num_features]
        if len(metadata) < num_features:
            metadata += [0.0] * (num_features - len(metadata))
        return metadata
    except Exception as e:
        print(f"[Metadata Error] {e}")
        return [0.0] * num_features

def extract_features(image_input):
    try:
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
            metadata_features = [0.0] * NUM_META_FEATURES  # can't get metadata
        else:
            raise ValueError("Unsupported image input type.")

        fft_features = extract_fft_features(img_array)
        combined = np.concatenate([fft_features, metadata_features])
        return combined
    except Exception as e:
        print(f"[Feature Extraction Error] {e}")
        return np.zeros(TOTAL_FEATURES)

# Test it standalone
if __name__ == "__main__":
    test_path = r"C:\Users\YASHASWINI\Downloads\ChatGPT Image Apr 4, 2025, 06_59_17 PM.png"
    features = extract_features(test_path)
    print(f"Feature vector shape: {features.shape}")
    print(features)
    print(f"Feature vector length: {len(features)}")
    print(f"Feature vector: {features}")
    print(f"Feature vector (first 10 values): {features[:10]}")
    print(f"Feature vector (last 10 values): {features[-10:]}")