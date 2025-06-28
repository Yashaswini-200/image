import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import exifread

# Load ResNet18 model (pretrained)
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier
resnet.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_fft_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum.flatten()[:2048]  # Trim to fixed length

def extract_metadata_features(img_path):
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag='UNDEF', details=False)
    return np.array([hash(str(v)) % 1000 for v in tags.values()][:100])  # Max 100 metadata features

def extract_resnet_features(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = resnet(image)
    return features.view(-1).numpy()  # Flatten the output

def extract_all_features(img_path):
    fft_feat = extract_fft_features(img_path)
    meta_feat = extract_metadata_features(img_path)
    resnet_feat = extract_resnet_features(img_path)
    return np.concatenate([fft_feat, meta_feat, resnet_feat])