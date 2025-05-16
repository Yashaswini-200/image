import numpy as np
from skimage import io, color
from skimage.transform import resize
from PIL import Image
from PIL.ExifTags import TAGS

def extract_features(image_path, num_bins=20, size=(256, 256)):
    try:
        print(f"🖼️ Processing: {image_path}")
        img = io.imread(image_path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = color.rgba2rgb(img)
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img = resize(img, size, anti_aliasing=True)

        F = np.fft.fft2(img)
        Fshift = np.fft.fftshift(F)
        log_F = np.log(1 + np.abs(Fshift))

        fft_mean = np.mean(log_F)
        fft_std = np.std(log_F)
        fft_entropy = -np.sum(log_F * np.log(log_F + 1e-10))
        hist_vals, _ = np.histogram(log_F.flatten(), bins=num_bins, density=True)

        # Metadata
        metadata_features = [0, 0, 0, 0, 0]
        try:
            image = Image.open(image_path)
            exif_data = image._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, (str, int, float)):
                        metadata_features.append(len(str(value)))
            metadata_features = metadata_features[:5] if len(metadata_features) >= 5 else metadata_features + [0] * (5 - len(metadata_features))
        except Exception as meta_e:
            print(f"⚠️ Metadata extraction failed: {meta_e}")
            metadata_features = [0, 0, 0, 0, 0]

        features = np.concatenate([[fft_mean, fft_std, fft_entropy], hist_vals, metadata_features])
        if np.any(np.isnan(features)):
            raise ValueError("❌ NaNs found in features!")
        return features

    except Exception as e:
        print(f"🚫 Failed to process {image_path}: {e}")
        return None