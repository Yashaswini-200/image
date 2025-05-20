from PIL import Image, ExifTags
import numpy as np
import cv2

def extract_fft_features(image_array, num_features=1000):
    """
    Extracts FFT-based features from an image array.
    """
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

def extract_metadata_features(image):
    """
    Extracts numeric EXIF metadata from a PIL image.
    """
    try:
        exif_data = image._getexif()
        metadata = []
        if exif_data:
            for tag, value in exif_data.items():
                if isinstance(value, (int, float)):
                    metadata.append(float(value))
                else:
                    metadata.append(0.0)
        return metadata
    except Exception as e:
        print(f"[Metadata Error] {e}")
        return []

def extract_features(image_input):
    """
    Combines FFT and EXIF metadata features into a single vector.
    Accepts image file path, PIL image, or NumPy array.
    """
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
            metadata_features = []
        else:
            raise ValueError("Unsupported image input type.")

        fft_features = extract_fft_features(img_array)

        # Normalize metadata length
        metadata_features = metadata_features[:50]
        if len(metadata_features) < 50:
            metadata_features += [0.0] * (50 - len(metadata_features))

        # Combine FFT and metadata features
        return np.concatenate([fft_features, metadata_features])

    except Exception as e:
        print(f"[Feature Extraction Error] {e}")
        return np.zeros(1000 + 50)
if __name__ == "__main__":
    test_path = r"C:\Users\YASHASWINI\Downloads\ChatGPT Image Apr 4, 2025, 06_59_17 PM.png"
    features = extract_features(test_path)
    print(f"Extracted {len(features)} features.")
    print(features)