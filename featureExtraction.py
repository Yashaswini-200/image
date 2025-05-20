from PIL import Image, ExifTags
import numpy as np
import cv2
from scipy.stats import entropy, skew

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
        print(f"[FFT Error] {e}")
        return np.zeros(num_features)

def extract_metadata_features(image):
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

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def extract_color_histogram(img_rgb, bins=16):
    hist = []
    for i in range(3):
        channel_hist, _ = np.histogram(img_rgb[..., i], bins=bins, range=(0, 256))
        hist.extend(channel_hist)
    hist = np.array(hist).astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def extract_entropy(img_gray):
    histogram, _ = np.histogram(img_gray, bins=256, range=(0,1))
    histogram = histogram.astype(float)
    if histogram.sum() > 0:
        histogram /= histogram.sum()
        return entropy(histogram)
    return 0.0

def extract_sharpness(img_gray):
    img_uint8 = (img_gray * 255).astype(np.uint8)
    return cv2.Laplacian(img_uint8, cv2.CV_64F).var()

def extract_edge_density(img_gray):
    img_uint8 = (img_gray * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, 100, 200)
    return np.sum(edges) / edges.size

def extract_brightness_contrast(img_gray):
    return [np.mean(img_gray), np.std(img_gray)]

def extract_color_moments(img_rgb):
    means = np.mean(img_rgb, axis=(0,1))
    stds = np.std(img_rgb, axis=(0,1))
    skews = skew(img_rgb.reshape(-1,3), axis=0)
    return np.concatenate([means, stds, skews])

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
            metadata_features = []
        else:
            raise ValueError("Unsupported image input type.")

        fft_features = extract_fft_features(img_array)

        img_gray = rgb2gray(img_array) / 255.0
        color_hist = extract_color_histogram(img_array)
        entropy_feat = np.array([extract_entropy(img_gray)])
        sharpness = np.array([extract_sharpness(img_gray)])
        edge_density = np.array([extract_edge_density(img_gray)])
        brightness_contrast = np.array(extract_brightness_contrast(img_gray))
        color_moments = extract_color_moments(img_array)

        metadata_features = metadata_features[:50]
        if len(metadata_features) < 50:
            metadata_features += [0.0] * (50 - len(metadata_features))

        features = np.concatenate([
            fft_features,
            metadata_features,
            color_hist,
            entropy_feat,
            sharpness,
            edge_density,
            brightness_contrast,
            color_moments
        ])

        return features

    except Exception as e:
        print(f"[Feature Extraction Error] {e}")
        return np.zeros(1000 + 50 + 48)  # fft(1000) + metadata(50) + other (48)

if __name__ == "__main__":
    test_path = r"C:\Users\YASHASWINI\Downloads\ChatGPT Image Apr 4, 2025, 06_59_17 PM.png"
    features = extract_features(test_path)
    print(f"Extracted {len(features)} features.")
    print(features)
